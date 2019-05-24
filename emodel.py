#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:23:38 2019

@author: asabater
"""

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"] = ""  

import sys
sys.path.append('keras_yolo3/')
import keras_yolo3.train as ktrain


from yolo3.model import yolo_head, box_iou, DarknetConv2D_BN_Leaky, DarknetConv2D, resblock_body

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D,Conv3D, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

#from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.model import tiny_yolo_body
from yolo3.utils import compose

from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.convolutional_recurrent import ConvLSTM2D


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
			weights_path='model_data/yolo_weights.h5', td_len=None, mode=None, spp=False):
	'''create the training model'''
	K.clear_session() # get a new session
	h, w = input_shape
	num_anchors = len(anchors)
	is_tiny_version = num_anchors==6 # default setting

#	y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
#		num_anchors//3, num_classes+5)) for l in range(3)]
#	print([ (h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
#		num_anchors//3, num_classes+5) for l in range(3) ])

	if is_tiny_version:
		y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
						 num_anchors//2, num_classes+5)) for l in range(2)]
		model_body = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
	elif td_len is not None and mode is not None:
		y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
						 num_anchors//3, num_classes+5)) for l in range(3)]
		image_input = Input(shape=(td_len, None, None, 3))
		model_body = r_yolo_body(image_input, num_anchors//3, num_classes, td_len, mode)
	else:
		y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
						 num_anchors//3, num_classes+5)) for l in range(3)]
		image_input = Input(shape=(None, None, 3))
		model_body = yolo_body(image_input, num_anchors//3, num_classes, spp)
	print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

	if load_pretrained and weights_path != '':
		model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
		print('Load weights {}.'.format(weights_path))
		if freeze_body in [1, 2]:
			# Freeze darknet53 body or freeze all but 3 output layers.
			num = 2 if td_len is not None and mode is not None else \
					(185, len(model_body.layers)-3)[freeze_body-1]
			for i in range(num): model_body.layers[i].trainable = False
			print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
	else:
		print('No freezing, no pretraining')
#	from keras.utils import multi_gpu_model
#	model_body = multi_gpu_model(model_body, gpus=2)
	
	model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
		arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
		[*model_body.output, *y_true])
	model = Model([model_body.input, *y_true], model_loss)

	return model


# =============================================================================
# =============================================================================


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
	'''Return yolo_loss tensor composed by the loss and all its components

	Parameters
	----------
	yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
	y_true: list of array, the output of preprocess_true_boxes
	anchors: array, shape=(N, 2), wh
	num_classes: integer
	ignore_thresh: float, the iou threshold whether to ignore object confidence loss

	Returns
	-------
	loss: tensor, shape=(1,)

	'''
	num_layers = len(anchors)//3 # default setting
	yolo_outputs = args[:num_layers]
	y_true = args[num_layers:]
	anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
	input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
	grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
	loss = 0
	total_xy_loss, total_wh_loss, total_confidence_loss_obj, total_confidence_loss_noobj, total_class_loss = 0, 0, 0, 0, 0
	m = K.shape(yolo_outputs[0])[0] # batch size, tensor
	mf = K.cast(m, K.dtype(yolo_outputs[0]))

	for l in range(num_layers):
		object_mask = y_true[l][..., 4:5]
		true_class_probs = y_true[l][..., 5:]

		grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
			 anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
		pred_box = K.concatenate([pred_xy, pred_wh])

		# Darknet raw box to calculate loss.
		raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
		raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
		raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
		box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

		# Find ignore mask, iterate over each of batch.
		ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
		object_mask_bool = K.cast(object_mask, 'bool')
		def loop_body(b, ignore_mask):
			true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
			iou = box_iou(pred_box[b], true_box)
			best_iou = K.max(iou, axis=-1)
			ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
			return b+1, ignore_mask
		_, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
		ignore_mask = ignore_mask.stack()
		ignore_mask = K.expand_dims(ignore_mask, -1)

		# K.binary_crossentropy is helpful to avoid exp overflow.
		xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
		wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
		confidence_loss_obj = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)
		confidence_loss_noobj = (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
#		confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
#			(1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
		class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

		xy_loss = K.sum(xy_loss) / mf
		wh_loss = K.sum(wh_loss) / mf
		confidence_loss_obj = K.sum(confidence_loss_obj) / mf
		confidence_loss_noobj = K.sum(confidence_loss_noobj) / mf
#		confidence_loss = K.sum(confidence_loss) / mf
		class_loss = K.sum(class_loss) / mf
		
		total_xy_loss += xy_loss
		total_wh_loss += wh_loss
#		total_confidence_loss += confidence_loss
		total_confidence_loss_obj += confidence_loss_obj
		total_confidence_loss_noobj += confidence_loss_noobj
		total_class_loss += class_loss
		
	loss += total_xy_loss + total_wh_loss + total_confidence_loss_obj + total_confidence_loss_noobj + total_class_loss
#	if print_loss:
#		loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
#	return loss
	return tf.convert_to_tensor([loss, 
							  total_xy_loss, total_wh_loss, 
							  total_confidence_loss_obj + total_confidence_loss_noobj,
							  total_confidence_loss_obj, total_confidence_loss_noobj, 
							  total_class_loss], dtype=tf.float32)


def loss(y_true, y_pred): return y_pred[0]
def xy_loss(y_true, y_pred): return y_pred[1]
def wh_loss(y_true, y_pred): return y_pred[2]
def confidence_loss(y_true, y_pred): return y_pred[3]
def confidence_loss_obj(y_true, y_pred): return y_pred[4]
def confidence_loss_noobj(y_true, y_pred): return y_pred[5]
def class_loss(y_true, y_pred): return y_pred[6]
	
	
# =============================================================================
# =============================================================================


def yolo_body(inputs, num_anchors, num_classes, spp=False):
	"""Create YOLO_V3 model CNN body in Keras."""
	darknet = Model(inputs, darknet_body(inputs))
	x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5), spp)

	x = compose(
			DarknetConv2D_BN_Leaky(256, (1,1)),
			UpSampling2D(2))(x)
	x = Concatenate()([x,darknet.layers[152].output])
	x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

	x = compose(
			DarknetConv2D_BN_Leaky(128, (1,1)),
			UpSampling2D(2))(x)
	x = Concatenate()([x,darknet.layers[92].output])
	x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

	return Model(inputs, [y1,y2,y3])


def make_last_layers(x, num_filters, out_filters, spp=False):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    if spp:
#	    x = compose(
##	            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
##	            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
#	            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
#	            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
#	            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
		
	    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), strides=(1,1))(x)
	    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), strides=(1,1))(x)
	    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), strides=(1,1))(x)
	    mp5 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
	    mp9 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(x)
	    mp13 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(x)
	    x = Concatenate()([x, mp13, mp9, mp5])
	    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
	    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
	    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)		
		
    else:
	    x = compose(
	            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
	            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
	            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
	            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
	            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def spp_block(x):
	'''Create SPP block'''
	
#	x = ZeroPadding2D(((1,0),(1,0)))(x)
	x = DarknetConv2D_BN_Leaky(512, (1,1), strides=(1,1))(x)
	x = DarknetConv2D_BN_Leaky(1024, (3,3), strides=(1,1))(x)
	x = DarknetConv2D_BN_Leaky(512, (1,1), strides=(1,1))(x)
	
	mp5 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
	mp9 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(x)
	mp13 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(x)
	x = Concatenate()([x, mp13, mp9, mp5])

#	x = DarknetConv2D_BN_Leaky(512, (1,1), strides=(1,1))(x)
#	x = DarknetConv2D_BN_Leaky(1024, (3,3), strides=(1,1))(x)
#	x = DarknetConv2D_BN_Leaky(512, (1,1), strides=(1,1))(x)
#	x = DarknetConv2D_BN_Leaky(1024, (3,3), strides=(1,1))(x)

	return x
	

def darknet_body(x):
	'''Darknent body having 52 Convolution2D layers'''
#	inpt = x
	x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
#	print(len(Model(inpt, x).layers))
	x = resblock_body(x, 64, 1)
#	print(len(Model(inpt, x).layers))
	x = resblock_body(x, 128, 2)
#	print(len(Model(inpt, x).layers))
	x = resblock_body(x, 256, 8)
#	print(len(Model(inpt, x).layers))
	x = resblock_body(x, 512, 8)
#	print(len(Model(inpt, x).layers))
	x = resblock_body(x, 1024, 4)
#	print(len(Model(inpt, x).layers))
	
	return x

# =============================================================================
# =============================================================================

def r_yolo_body(image_input_td, num_anchors, num_classes, td_len, mode):
	"""Create YOLO_V3 model CNN body in Keras."""
#	image_input_td = Input(shape=(td_len, None, None, 3))
#	darknet = Model(image_input_td, r_darknet_body(inputs, image_input_td))
	darknet, skip_conn = darknet_body_r(image_input_td, td_len, mode)
	darknet = Model(image_input_td, darknet)
#	print(darknet.summary())
	x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))
	
	print('Concatenating:', darknet.layers[skip_conn[0]], darknet.layers[skip_conn[0]].output)
	print('Concatenating:', darknet.layers[skip_conn[1]], darknet.layers[skip_conn[1]].output)

	x = compose(
			DarknetConv2D_BN_Leaky(256, (1,1)),
			UpSampling2D(2))(x)
	print(x.shape)
	x = Concatenate()([x,darknet.layers[skip_conn[1]].output])
	x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))
	
	print('Frist layer concatenated')

	x = compose(
			DarknetConv2D_BN_Leaky(128, (1,1)),
			UpSampling2D(2))(x)
	x = Concatenate()([x,darknet.layers[skip_conn[0]].output])
	x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))
	print('Second layer concatenated')

	return Model(image_input_td, [y1,y2,y3])


def darknet_body_r(image_input_td, td_len, mode):
	
	image_input = Input(shape=(None, None, 3))		# (320, 320, 3)
	skip_conn = []

	x = DarknetConv2D_BN_Leaky(32, (3,3))(image_input)
	print(len(Model(image_input, x).layers))
	x = resblock_body(x, 64, 1)
	print(len(Model(image_input, x).layers))
	x = resblock_body(x, 128, 2)
	print(len(Model(image_input, x).layers))
	x = resblock_body(x, 256, 8)
	print(len(Model(image_input, x).layers))
	x = Model(image_input, x)
	print('-'*20)
	
	x = TimeDistributed(x)(image_input_td)
#	x = TimeDistributed(ZeroPadding2D(((1,0),(1,0))))(x)
 
	if mode == 'lstm':
		x = ConvLSTM2D(256, kernel_size=(3,3), padding='same', activation='relu')(x)		
	elif mode == 'bilstm':
#		x = TimeDistributed(ZeroPadding2D(((1,0),(1,0))))(x)
		x = Bidirectional(ConvLSTM2D(256, kernel_size=(3,3), padding='same', activation='relu'))(x)		
	elif mode == '3d':
		x = Conv3D(256, kernel_size=(td_len,3,3), padding='valid', activation='relu')(x)
		x = Lambda(lambda x: x[:,0,:,:])(x)
		x = ZeroPadding2D(((2,0),(2,0)))(x)
	else: raise ValueError('Recurrent mode not recognized')
	
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)
	print(len(Model(image_input_td, x).layers))
	skip_conn.append(len(Model(image_input_td, x).layers)-1)
	
	x = resblock_body(x, 512, 8)
	print(len(Model(image_input_td, x).layers))
	skip_conn.append(len(Model(image_input_td, x).layers)-1)
	x = resblock_body(x, 1024, 4)
	print(len(Model(image_input_td, x).layers))
	
	return x, skip_conn


# %%

if False:
	# %%

	# concat connections at 92, 152 -> 4, 64
	
	
	td_len = 5 
	img_size = 320
	#image_input = Input(shape=(320, 320, 3))		# (320, 320, 3)
	image_input_td = Input(shape=(td_len, img_size, img_size, 3))
	#r_darknet = Model(image_input_td, r_darknet_body(image_input, image_input_td))
	r_darknet, skip_conn = darknet_body_r(image_input_td, td_len, mode='3d')
	r_darknet = Model(image_input_td, r_darknet)
	
	r_darknet.summary()
	
	
	# %%
	
	img_size = 416
	input_shape = (img_size,img_size)
	num_anchors = 9
	num_classes = 7
	
	K.clear_session() # get a new session
	image_input = Input(shape=(img_size,img_size, 3))		 # (None, None, 3)
	h, w = input_shape
	#num_anchors = len(anchors)
	
	y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
		num_anchors//3, num_classes+5)) for l in range(3)]
		
		
	darknet = Model(image_input, darknet_body(image_input, spp=False))
	darknet_spp = Model(image_input, darknet_body(image_input, spp=True))
	yolo = yolo_body(image_input, num_anchors//3, num_classes, spp=True)
	
#	body_darknet = yolo_body(image_input, num_anchors//3, num_classes)
	
	# %%
	
	print(darknet.layers[152])
	print(darknet_spp.layers[152])
	print(yolo.layers[152])
	print(darknet.layers[92])
	print(darknet_spp.layers[92])
	print(yolo.layers[92])
	
	# %%
	
	from keras.models import model_from_json
	from keras.utils import plot_model
	import json
	
	img_size = 416
	num_anchors = 9
	num_classes = 7
	image_input = Input(shape=(img_size,img_size, 3))		 # (None, None, 3)
	yolo_spp = yolo_body(image_input, num_anchors//3, num_classes, spp=True)
	orig_spp = model_from_json(json.load(open('weights/spp.json', 'r')))
	plot_model(yolo_spp, to_file='yolo_spp.png', show_shapes=True)
	plot_model(orig_spp, to_file='orig_spp.png', show_shapes=True)

	
	# %%
	
	for i in range(len(yolo_spp.layers)):
		if yolo_spp.layers[i].name != orig_spp.layers[i].name:
			print(i, yolo_spp.layers[i].name, orig_spp.layers[i].name)
		elif 'conv' in yolo_spp.layers[i].name:
			l1, lo = yolo_spp.layers[i], orig_spp.layers[i]
			if not (l1.strides == lo.strides and l1.kernel_size == lo.kernel_size and 
		   l1.input.shape[-1] == lo.input.shape[-1] and l1.filters == lo.filters):
				print(i)
				print(i, l1.name, '\t\t', l1.strides,l1.kernel_size, l1.filters, '\t\t', l1.input.shape.as_list())
				print(i, lo.name, '\t\t', lo.strides,lo.kernel_size, lo.filters, '\t\t', lo.input.shape.as_list())
		elif 'pool' in yolo_spp.layers[i].name:
			l1, lo = yolo_spp.layers[i], orig_spp.layers[i]
			print(i)
			print(i, l1.name, l1.pool_size, l1.strides)
			print(i, lo.name, lo.pool_size, lo.strides)
		
			
	# %%
	
	for i,l in enumerate(orig_spp.layers):
		if 'conv' in l.name:
			print(i, l.name, '\t\t', l.strides,l.kernel_size, l.filters, '\t\t', l.input.shape.as_list())
		elif 'add_' in l.name or 'concat' in l.name:
			print(i, l.name, '\t\t\t\t\t\t', [ l.shape.as_list() for l in l.input ])
		else:
			print(i, l.name, '\t\t\t\t\t\t', l.input.shape.as_list())
	
	
	# %%
	
	img_size = 19
	image_input = Input(shape=(img_size,img_size, 3))		 # (None, None, 3)
	x = ZeroPadding2D(((1,0),(1,0)))(image_input)
	x = DarknetConv2D_BN_Leaky(512, (1,1), strides=(1,1))(x)
	x = DarknetConv2D_BN_Leaky(1024, (3,3), strides=(1,1))(x)
	x = DarknetConv2D_BN_Leaky(512, (1,1), strides=(1,1))(x)
	
	mp5 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(x)
	mp9 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(x)
	mp13 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(x)
	x = Concatenate()([x, mp5, mp9, mp13])

	x = DarknetConv2D_BN_Leaky(512, (1,1), strides=(1,1))(x)
	x = DarknetConv2D_BN_Leaky(1024, (3,3), strides=(1,1))(x)
	x = DarknetConv2D_BN_Leaky(512, (1,1), strides=(1,1))(x)
	x = DarknetConv2D_BN_Leaky(1024, (3,3), strides=(1,1))(x)
	x = Model(image_input, x)
	
	
	#%%
	
	#print(r_darknet.summary())
	
	# concat connections at 92, 152 -> 4, 64
	
	skip_conn_r = skip_conn
	#skip_conn_r = [6,66]
	print('darknet	  |||', darknet.layers[92].name, darknet.layers[92].output)
	print('r_darknet	|||', r_darknet.layers[skip_conn_r[0]].name, r_darknet.layers[skip_conn_r[0]].output)
	print('body_darknet |||', body_darknet.layers[skip_conn_r[0]].name, body_darknet.layers[skip_conn_r[0]].output)
	
	print('darknet	  |||', darknet.layers[152].name, darknet.layers[152].output)
	print('r_darknet	|||', r_darknet.layers[skip_conn_r[1]].name, r_darknet.layers[skip_conn_r[1]].output)
	print('body_darknet |||', body_darknet.layers[skip_conn_r[1]].name, body_darknet.layers[skip_conn_r[1]].output)
	
	
	# %%
	
	img_size = None
	num_anchors = 9
	num_classes = 7
	td_len = 5 
	
	K.clear_session() # get a new session
	#image_input = Input(shape=(None,None, 3))		 # (None, None, 3)
	image_input_td = Input(shape=(td_len, img_size,img_size, 3))
	#num_anchors = len(anchors)
	
	#input_shape = (img_size,img_size)
	#h, w = input_shape
	#y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
	#	num_anchors//3, num_classes+5)) for l in range(3)]
		
		
	model = r_yolo_body(image_input_td, num_anchors//3, num_classes, td_len, 'lstm')
	model.summary()
	
	
	# %%
	
	num_classes = 7
	path_anchors = 'base_models/yolo_anchors.txt'
	path_weights = 'base_models/yolo.h5'
	anchors = ktrain.get_anchors(path_anchors)
	img_size = 320
	model = None
	
	td_len = 3
	model = create_model((img_size,img_size), anchors, num_classes, load_pretrained=True, freeze_body=2,
				weights_path=path_weights, td_len=5, mode='bilstm')
	
	
	# %%
	
	for img_size in [320, 416, 608]:
		td_data = np.concatenate([ np.random.rand(1, 1, img_size, img_size, 3) for i in range(td_len) ], axis=1)
		if td_len == 1: td_data = td_data[0,::]
		
	#	out_boxes, out_scores, out_classes = self.sess.run(
	#			[self.boxes, self.scores, self.classes],
	#			feed_dict={
	#				self.yolo_model.input: image_data,
	#				self.input_image_shape: [image.size[1], image.size[0]],
	#				K.learning_phase(): 0
	#			})
		
		pred = model.predict(td_data)
		for p in pred: print(p.shape)
		print('='*20)


# %%

	for i, l in enumerate(model.layers):
		if not l.trainable:
			print(i, l)






