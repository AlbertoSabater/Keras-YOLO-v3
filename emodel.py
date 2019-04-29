#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:23:38 2019

@author: asabater
"""

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"] = ""  

import sys
sys.path.append('keras_yolo3/')
import keras_yolo3.train as ktrain


from yolo3.model import yolo_loss, DarknetConv2D_BN_Leaky, DarknetConv2D, resblock_body, make_last_layers

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
from yolo3.utils import compose

from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.convolutional_recurrent import ConvLSTM2D


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5', td_len=None, mode=None):
    '''create the training model'''
    K.clear_session() # get a new session
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]
#    print([ (h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
#        num_anchors//3, num_classes+5) for l in range(3) ])

    if td_len is not None and mode is not None:
        image_input = Input(shape=(td_len, None, None, 3))
        model_body = r_yolo_body(image_input, num_anchors//3, num_classes, td_len, mode)
    else:
        image_input = Input(shape=(None, None, 3))
        model_body = yolo_body(image_input, num_anchors//3, num_classes)
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
#    from keras.utils import multi_gpu_model
#    model_body = multi_gpu_model(model_body, gpus=2)
    
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)



    return model


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

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


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
#    inpt = x
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
#    print(len(Model(inpt, x).layers))
    x = resblock_body(x, 64, 1)
#    print(len(Model(inpt, x).layers))
    x = resblock_body(x, 128, 2)
#    print(len(Model(inpt, x).layers))
    x = resblock_body(x, 256, 8)
#    print(len(Model(inpt, x).layers))
    x = resblock_body(x, 512, 8)
#    print(len(Model(inpt, x).layers))
    x = resblock_body(x, 1024, 4)
#    print(len(Model(inpt, x).layers))
    return x

# =============================================================================
# =============================================================================

def r_yolo_body(image_input_td, num_anchors, num_classes, td_len, mode):
    """Create YOLO_V3 model CNN body in Keras."""
#    image_input_td = Input(shape=(td_len, None, None, 3))
#    darknet = Model(image_input_td, r_darknet_body(inputs, image_input_td))
    darknet, skip_conn = darknet_body_r(image_input_td, td_len, mode)
    darknet = Model(image_input_td, darknet)
#    print(darknet.summary())
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
    
    image_input = Input(shape=(None, None, 3))        # (320, 320, 3)
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
#    x = TimeDistributed(ZeroPadding2D(((1,0),(1,0))))(x)
 
    if mode == 'lstm':
        x = ConvLSTM2D(256, kernel_size=(3,3), padding='same', activation='relu')(x)        
    elif mode == 'bilstm':
#        x = TimeDistributed(ZeroPadding2D(((1,0),(1,0))))(x)
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
    #image_input = Input(shape=(320, 320, 3))        # (320, 320, 3)
    image_input_td = Input(shape=(td_len, img_size, img_size, 3))
    #r_darknet = Model(image_input_td, r_darknet_body(image_input, image_input_td))
    r_darknet, skip_conn = darknet_body_r(image_input_td, td_len, mode='3d')
    r_darknet = Model(image_input_td, r_darknet)
    
    r_darknet.summary()
    
    
    # %%
    
    img_size = 320
    input_shape = (img_size,img_size)
    num_anchors = 9
    num_classes = 7
    
    K.clear_session() # get a new session
    image_input = Input(shape=(img_size,img_size, 3))         # (None, None, 3)
    h, w = input_shape
    #num_anchors = len(anchors)
    
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]
        
        
    darknet = Model(image_input, darknet_body(image_input))
    
    
    body_darknet = yolo_body(image_input, num_anchors//3, num_classes)
    
    
    
    #%%
    
    #print(r_darknet.summary())
    
    # concat connections at 92, 152 -> 4, 64
    
    skip_conn_r = skip_conn
    #skip_conn_r = [6,66]
    print('darknet      |||', darknet.layers[92].name, darknet.layers[92].output)
    print('r_darknet    |||', r_darknet.layers[skip_conn_r[0]].name, r_darknet.layers[skip_conn_r[0]].output)
    print('body_darknet |||', body_darknet.layers[skip_conn_r[0]].name, body_darknet.layers[skip_conn_r[0]].output)
    
    print('darknet      |||', darknet.layers[152].name, darknet.layers[152].output)
    print('r_darknet    |||', r_darknet.layers[skip_conn_r[1]].name, r_darknet.layers[skip_conn_r[1]].output)
    print('body_darknet |||', body_darknet.layers[skip_conn_r[1]].name, body_darknet.layers[skip_conn_r[1]].output)
    
    
    # %%
    
    img_size = None
    num_anchors = 9
    num_classes = 7
    td_len = 5 
    
    K.clear_session() # get a new session
    #image_input = Input(shape=(None,None, 3))         # (None, None, 3)
    image_input_td = Input(shape=(td_len, img_size,img_size, 3))
    #num_anchors = len(anchors)
    
    #input_shape = (img_size,img_size)
    #h, w = input_shape
    #y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
    #    num_anchors//3, num_classes+5)) for l in range(3)]
        
        
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
        
    #    out_boxes, out_scores, out_classes = self.sess.run(
    #            [self.boxes, self.scores, self.classes],
    #            feed_dict={
    #                self.yolo_model.input: image_data,
    #                self.input_image_shape: [image.size[1], image.size[0]],
    #                K.learning_phase(): 0
    #            })
        
        pred = model.predict(td_data)
        for p in pred: print(p.shape)
        print('='*20)


# %%

    for i, l in enumerate(model.layers):
        if not l.trainable:
            print(i, l)





