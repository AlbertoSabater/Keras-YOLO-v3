#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:03:59 2019

@author: asabater
"""

import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"  


import sys
sys.path.append('keras_yolo3/')

import keras_yolo3.train as ktrain
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from tensorflow.python.client import device_lib

import train_utils
import numpy as np
import json

from data_factory import data_generator_wrapper_default, data_generator_wrapper_custom
from emodel import create_model, loss, xy_loss, wh_loss, confidence_loss, confidence_loss_obj, confidence_loss_noobj, class_loss

import evaluate_model
import time
import pyperclip

# Iniciar la stage 2 con un lr más bajo

# tensorboard --logdir /media/asabater/hdd/egocentric/adl

path_results = '/mnt/hdd/egocentric_results/'
print_indnt = 12
print_line = 100
num_gpu = len([x for x in device_lib.list_local_devices() if x.device_type == 'GPU'])


path_anchors = 'base_models/yolo_anchors.txt'
dataset_name = 'adl'
spp = False
mode = None					 # lstm/bilstm/3d

#path_weights, freeze_body, path_anchors = 'base_models/yolo_tiny.h5', \
#									2, 'base_models/tiny_yolo_anchors.txt' 	# TinyYolo + pretraining
path_weights, freeze_body = 'base_models/yolo.h5', 2 	 	 	 	 	 	# COCO pretraining
#path_weights, freeze_body, spp = 'base_models/yolov3-spp.h5', 2, True 	 	# SPP pretraining
#path_weights, freeze_body, spp = 'base_models/darknet53.h5', 1, True 	 	# SPP backbone pretraining
#path_weights, freeze_body, spp = '', 0, True 	 	 	 	 	 	 	 	# SPP no pretraining 	
#path_weights, freeze_body = 'base_models/darknet53.h5', 1 		 	 	 	# Darknet pretraining
#path_weights, freeze_body = '', 0 	 	 	 	 	 	 	 	 	 	 	# No pretraining
#path_weights, freeze_body = train_utils.get_best_weights(train_utils.get_model_path(
#		path_results, 'kitchen', 1)), 2 	 	 	 	 	 	 	 	 	# Kitchen pretraining
#path_weights = 'base_models/darknet53.h5'
#freeze_body = 2				 # freeze_body = 1 -> freeze feature extractor
#								 # freeze_body = 2 -> freeze all but 3 output layers
#								 # freeze_body = otro -> don't freeze
input_shape = (416,416)		 # multiple of 32, hw


# TODO: Pretrain con un modelo de KTICHEN


#val_split = 0.1
if freeze_body == 0: frozen_epochs = 0
else: frozen_epochs = 15			   # 50



title = 'Remove null trainings'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
train_utils.remove_null_trainings(path_results, dataset_name)
print('='*print_line)



path_dataset = ''
size_suffix = ''
version = -1
mode = None					 # lstm/bilstm/3d

# Get dataset annotations, classes and anchors
if dataset_name == 'adl':
#	path_dataset = '/mnt/hdd/datasets/adl_dataset/ADL_frames/'
	version = '_v2_27'		# _v2_27 , _v3_8
	
	size_suffix = ''		 # '_416'
	input_shape = [416,416]
	
	path_dataset = '/home/asabater/projects/ADL_dataset/'
	path_annotations = ['./dataset_scripts/adl/annotations_adl_train{}{}.txt'.format(size_suffix, version),
						'./dataset_scripts/adl/annotations_adl_val{}{}.txt'.format(size_suffix, version)]
	
#	r_suffix = ','.join([ str(f) for f in [5,10,20] ])	# 5 | 10 | 5,10 | 10,20 | 5,10,15 | 5,10,20
#	path_annotations, mode = ['./dataset_scripts/adl/annotations_adl_train{}_r_fd|{}|.txt'.format(version, r_suffix),
#							  './dataset_scripts/adl/annotations_adl_val{}_r_fd|{}|.txt'.format(version, r_suffix)], \
#							  'bilstm'
#	path_annotations = ['./dataset_scripts/adl/annotations_adl_train{}{}.txt'.format(version, size_suffix),
#						'./dataset_scripts/adl/annotations_adl_val{}{}.txt'.format(version, size_suffix)]

	path_classes = './dataset_scripts/adl/adl_classes{}.txt'.format(version)
#	path_anchors = './dataset_scripts/adl/anchors_adl{}{}.txt'.format(size_suffix, version)
#	path_anchors = './dataset_scripts/adl/anchors_adl{}_pr{}.txt'.format(version, input_shape[0])

elif dataset_name == 'kitchen':
	
	version = '_cv2_18' 			# '', _v1_15, _v2_25, _v3_35, _cv1_17, _cv2_18
	input_shape = (416,416)
	path_dataset = ''
	path_annotations = ['./dataset_scripts/kitchen/annotations_kitchen_train{}.txt'.format(version),
						 './dataset_scripts/kitchen/annotations_kitchen_val{}.txt'.format(version)]
	path_classes = './dataset_scripts/kitchen/kitchen_classes{}.txt'.format(version)
	
	
elif dataset_name == 'coco':	
	# 117266 train images
	# 4952 val images
	# 80/12 categories
	
	version = '_super'
	input_shape = (416,416)
	path_dataset = '/mnt/hdd/datasets/coco/'
	
	path_annotations = ['./dataset_scripts/coco/annotations_coco_train{}.txt'.format(version),
					'./dataset_scripts/coco/annotations_coco_val{}.txt'.format(version)]
	path_classes = './dataset_scripts/coco/coco_classes{}.txt'.format(version)
	
elif dataset_name == 'voc':
#	path_dataset = '/mnt/hdd/datasets/adl_dataset/ADL_frames/'
	input_shape = (416,416)
	version = ''
	size_suffix = ''
	path_dataset = '/mnt/hdd/datasets/VOC/'
	path_annotations = ['./dataset_scripts/voc/annotations_voc_train.txt',
						'./dataset_scripts/voc/annotations_voc_val.txt']
#	path_annotations = ['/home/asabater/projects/ADL_dataset/annotations_adl_train.txt',
#						'/home/asabater/projects/ADL_dataset/annotations_adl_val.txt']
	path_classes = './dataset_scripts/voc/voc_classes.txt'
	path_anchors = './dataset_scripts/voc/anchors_voc{}{}.txt'.format(size_suffix, version)

#elif dataset_name == 'epic':
#	path_annotations = '/home/asabater/projects/epic_dataset/annotations_epic_train.txt'
#	path_classes = '/home/asabater/projects/epic_dataset/epic_classes.txt'
#	
#elif dataset_name == 'imagenet':
#	path_annotations = '/media/asabater/hdd/datasets/imagenet_vid/annotations_train.txt'
#	path_classes = '/media/asabater/hdd/datasets/imagenet_vid/imagenet_vid_classes.txt'
	
else: raise ValueError('Dataset not recognized')

# Load dataset classes and anchors
class_names = ktrain.get_classes(path_classes)
num_classes = len(class_names)
anchors = ktrain.get_anchors(path_anchors)



# Train/Val split
np.random.seed(10101)
if type(path_annotations) == list:
	with open(path_annotations[0]) as f: lines_train = f.readlines()
	with open(path_annotations[1]) as f: lines_val = f.readlines()
	num_train, num_val = len(lines_train), len(lines_val)
else:
	with open(path_annotations) as f: lines = f.readlines()
	np.random.shuffle(lines)
	num_val, num_train = int(len(lines)*val_split); len(lines) - num_val
	lines_train = lines[:num_train]; lines_val = lines[num_train:]
np.random.shuffle(lines_train), np.random.shuffle(lines_val)
np.random.seed(None)

lines_train = [ ','.join([ path_dataset+img for img in ann.split(' ')[0].split(',') ]) \
					+ ' ' + ' '.join(ann.split(' ')[1:]) for ann in lines_train ]
lines_val = [ ','.join([ path_dataset+img for img in ann.split(' ')[0].split(',') ]) \
					+ ' ' + ' '.join(ann.split(' ')[1:]) for ann in lines_val ]

td_len = None if mode is None else len(lines_train[0].split(' ')[0].split(','))


if mode is None:
	batch_size_frozen = 32		  # 32
	batch_size_unfrozen = 4		 # note that more GPU memory is required after unfreezing the body
else:
	batch_size_frozen = 8		  # 32
	batch_size_unfrozen = 2		 # note that more GPU memory is required after unfreezing the body


title = 'Create and get model folders'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
# Create and get model folders
path_model = train_utils.create_model_folder(path_results, dataset_name)
print(path_model)
print('='*print_line)


print(num_train, num_val)
#num_train = batch_size_frozen * 10
#num_train, num_val = 250, 250
#num_train = 200
print(num_train, num_val)


## %%

# =============================================================================
# Create model
# =============================================================================

title = 'Create Keras model'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
print('Num. GPUs:', num_gpu)
#model = ktrain.create_model(input_shape, anchors, num_classes,
#			freeze_body = freeze_body, 
#			weights_path = path_weights) # make sure you know what you freeze
model = create_model(input_shape, anchors, num_classes, 
					 freeze_body=freeze_body,
					 weights_path=path_weights, td_len=td_len, mode=mode, 
					 spp=spp)
print('='*print_line)



# Train callbacks
logging = TensorBoard(log_dir = path_model)
checkpoint = ModelCheckpoint(path_model + 'weights/' + 'ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5',
							 monitor='val_loss', 
							 save_weights_only=True, 
							 save_best_only=True, 
							 period=1)
reduce_lr_1 = ReduceLROnPlateau(monitor='loss', min_delta=0.5, factor=0.1, patience=4, verbose=1)
reduce_lr_2 = ReduceLROnPlateau(monitor='val_loss', min_delta=0, factor=0.1, patience=4, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
	


title = 'Storing train params and model architecture'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
train_params = {
				'dataset_name': dataset_name,
				'freeze_body': freeze_body,
				'input_shape': input_shape,
#				'val_split': val_split,
				'batch_size_frozen': batch_size_frozen,
				'batch_size_unfrozen': batch_size_unfrozen,
				'frozen_epochs': frozen_epochs,
				'path_anchors': path_anchors,
				'path_annotations': path_annotations,
				'path_classes': path_classes,
				'path_dataset': path_dataset,
				'path_weights': path_weights,
				'num_val': num_val,
				'num_train': num_train,
				'size_suffix': size_suffix, 
				'version': version,
				'td_len': td_len,
				'mode': mode,
				'spp': spp
				}
print(train_params)
with open(path_model + 'train_params.json', 'w') as f:
	json.dump(train_params, f)
	
model_architecture = model.to_json()
with open(path_model + 'architecture.json', 'w') as f:
	json.dump(model_architecture, f)
print('='*print_line)


title = 'Excel params'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
print('|{path_model}|\t|{version}|\t|{input_shape}|\t|{annotations}|\t|{anchors}|\t|{pretraining}|\t|{frozen}|'.format(
		path_model = path_model, version = version, input_shape = input_shape,
		annotations = path_annotations[1].split('/')[-1],
		anchors = path_anchors.split('/')[-1],
		pretraining = path_weights.split('/')[-1],
		frozen = freeze_body
		))
print('='*print_line)

excel_resume = evaluate_model.get_excel_resume_full(path_model, train_params, '', '', 
						  {}, {}, '', None)
print(excel_resume)		
pyperclip.copy(excel_resume)


## %%

# =============================================================================
# Define train metrics
# =============================================================================


#metrics = [train_utils.get_lr_metric(optimizer)]

#import loss_metrics

#loss_metrics.anchors = anchors
#loss_metrics.num_classes = num_classes
#
#metrics = [
#			train_utils.get_lr_metric(optimizer),
#			loss_metrics.xy_metric
#		]

#def xy_loss(y_true, y_pred): return y_pred[1]



# %%

# =============================================================================
# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
# =============================================================================

title = 'Train first stage'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
import datetime
print(print(datetime.datetime.now()))
if True:
	optimizer = Adam(lr=1e-3)
#	model.compile(optimizer = optimizer,
#				  loss = {'yolo_loss': lambda y_true, y_pred: y_pred},		# use custom yolo_loss Lambda layer.
#				  metrics = [train_utils.get_lr_metric(optimizer)]	
#				  )
	model.compile(optimizer = optimizer,
				  loss = {'yolo_loss': loss},		# use custom yolo_loss Lambda layer.
				  metrics = [
							 xy_loss, wh_loss, confidence_loss, 
							 confidence_loss_obj, confidence_loss_noobj, class_loss,
							 train_utils.get_lr_metric(optimizer),
							 ]	
				  )

	print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_frozen))
	hist_1 = model.fit_generator(
			data_generator_wrapper_custom(lines_train, batch_size_frozen, input_shape, anchors, num_classes, random=True),
			steps_per_epoch = max(1, num_train//batch_size_frozen),
			validation_data = data_generator_wrapper_custom(lines_val, batch_size_frozen, input_shape, anchors, num_classes, random=False),
			validation_steps = max(1, num_val//batch_size_frozen),
			epochs = frozen_epochs,
			initial_epoch = 0,
			callbacks=[logging, 
#					   reduce_lr_1,
					   checkpoint
					   ])
	model.save_weights(path_model + 'weights/trained_weights_stage_1.h5')
print('='*print_line)


## %%

# =============================================================================
# Unfreeze and continue training, to fine-tune.
# Train longer if the result is not good.
# =============================================================================

title = 'Train second stage'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
if True:
	for i in range(len(model.layers)):
		model.layers[i].trainable = True
	print('Unfreeze all of the layers.')

	optimizer = Adam(lr=1e-4)
#	model.compile(optimizer = optimizer,
#				  loss = {'yolo_loss': lambda y_true, y_pred: y_pred},		# use custom yolo_loss Lambda layer.
#				  metrics = [train_utils.get_lr_metric(optimizer),
##							 loss_metrics.get_loss_metric(anchors, num_classes, .5)
#							 ]
#				  )

	model.compile(optimizer = optimizer,
				  loss = {'yolo_loss': loss},		# use custom yolo_loss Lambda layer.
				  metrics = [
							 xy_loss, wh_loss, confidence_loss, 
							 confidence_loss_obj, confidence_loss_noobj, class_loss,
							 train_utils.get_lr_metric(optimizer),
							 ]	
				  )

	print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_unfrozen))
	hist_2 = model.fit_generator(
		data_generator_wrapper_custom(lines_train, batch_size_unfrozen, input_shape, anchors, num_classes, random=True),
		steps_per_epoch = max(1, num_train//batch_size_unfrozen),
		validation_data = data_generator_wrapper_custom(lines_val, batch_size_unfrozen, input_shape, anchors, num_classes, random=False),
		validation_steps = max(1, num_val//batch_size_unfrozen),
		epochs = 500,
		initial_epoch = frozen_epochs,
		callbacks = [logging, checkpoint, 
					   reduce_lr_2,
#					   reduce_lr
						early_stopping])
	model.save_weights(path_model + 'weights/trained_weights_final.h5')
print('='*print_line)
	
	
## %%

train_utils.remove_worst_weights(path_model)


## %%

best_weights = train_utils.get_best_weights(path_model)
model.load_weights(best_weights)

	
# %%

model_num = int(path_model.split('/')[-2].split('_')[-1])
#dataset_name, path_results = 'kitchen', '/mnt/hdd/egocentric_results/'
#evaluate_model.main(path_results, dataset_name, model_num, score, iou, num_annotation_file, plot=True, full=True)

iou = 0.5
if dataset_name == 'kitchen':
	annotation_files = [(0.005, 1), (0.005, 0)]
else:
	annotation_files = [(0, 1), (evaluate_model.MIN_SCORE, 0)]
times = [ None for i in range(max([ af for _,af in annotation_files ])+1) ]
eval_stats_arr = [ {} for i in range(max([ af for _,af in annotation_files ])+1) ]
for score, num_annotation_file in annotation_files:		 # ,0

	print('='*80)
	print('dataset = {}, num_ann = {}, model = {}'.format(
			dataset_name, num_annotation_file, model_num))	
	print('='*80)
	
	times[num_annotation_file] = time.time()

	_, _, _, _, _, eval_stats_f, _, loss_f = evaluate_model.main(
			path_results, dataset_name, model_num, score, iou, num_annotation_file, 
			plot=True, full=True, best_weights=False)
	
	_, _, _, _, resume, eval_stats_t, _, loss_t = evaluate_model.main(
			path_results, dataset_name, model_num, score, iou, num_annotation_file, 
			plot=True, full=True, best_weights=True)
	
	
	if eval_stats_t['total'][1] >= eval_stats_f['total'][1]:
		print('Model {} con best_weights'.format(model_num))
		eval_stats_arr[num_annotation_file] = eval_stats_t
		best_weights = True
	else:
		print('Model {} con stage_2'.format(model_num))
		eval_stats_arr[num_annotation_file] = eval_stats_f
		best_weights = False
				
				
#	eval_stats_arr[num_annotation_file] = eval_stats
	
#	if num_annotation_file == 1:
#		pyperclip.copy(resume)
		

	times[num_annotation_file] = (time.time() - times[num_annotation_file])/60

eval_stats_train, eval_stats_val = eval_stats_arr
full_resume = evaluate_model.get_excel_resume_full(
				model_folder = resume.split('\t')[0], 
				train_params = train_params, 
				train_loss = resume.split('\t')[7], 
				val_loss = resume.split('\t')[8], 
				eval_stats_train = eval_stats_train, 
				eval_stats_val = eval_stats_val, 
				train_diff = resume.split('\t')[6], 
				best_weights = best_weights)

print('='*80)
print(full_resume)		
pyperclip.copy(full_resume)


#%%

#for name in dir():
#    if not name.startswith('_'):
#        del globals()[name]
