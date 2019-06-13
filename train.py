#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:38:36 2019

@author: asabater
"""

import sys
sys.path.append('keras_yolo3/')

import keras_yolo3.train as ktrain
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

import train_utils
import numpy as np
import json

from data_factory import data_generator_wrapper_custom
from emodel import create_model, loss, xy_loss, wh_loss, confidence_loss, confidence_loss_obj, confidence_loss_noobj, class_loss

import evaluate_model
import time
import pyperclip
import datetime
import argparse


print_indnt = 12
print_line = 100


# %%

def log(msg):
	with open("log.txt", "a") as log_file:
		   log_file.write('{} | {}\n'.format(str(datetime.datetime.now()), msg))


def get_train_params_from_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("path_results", help="path where the store the training results")
	parser.add_argument("dataset_name", help="subfolder where to store the training results")
	parser.add_argument("--path_dataset", help="path to each training image if not specified in annotations file", default='', type=str)
	parser.add_argument("path_classes", help="dataset classes file")
	parser.add_argument("path_anchors", help="anchors file")
	parser.add_argument("path_annotations_train", help="train annotations file")
	parser.add_argument("path_annotations_val", help="validation annotations file")
	
	parser.add_argument("path_weights", help="path to pretrained weights")
	parser.add_argument("freeze_body", help="0 to not freezing\n1 to freeze backbone\n2 to freeze all the model")
	parser.add_argument("--frozen_epochs", help="number of frozen training epochs", type=int, default=15)
	parser.add_argument("--input_shape", help="training/validation input image shape. Must be a multiple of 32", type=int, default=416)
	parser.add_argument("--spp", help="use Spatial Pyramid Pooling", action='store_true')
	parser.add_argument("--multi_scale", help="use multi-scale training", action='store_true')
	
	args = parser.parse_args()

	train_params = {
			'path_results': args.path_results,
			'dataset_name': args.dataset_name,
			'path_dataset': args.path_dataset,
			'path_classes': args.path_classes,
			'path_anchors': args.path_anchors,
			'path_annotations': [args.path_annotations_train, args.path_annotations_val],
			'path_weights': args.path_weights,
			'freeze_body': int(args.freeze_body),
			'frozen_epochs': int(args.frozen_epochs),
			'input_shape': [int(args.input_shape), int(args.input_shape)],
			'spp': args.spp,
			'multi_scale': args.multi_scale,
			'size_suffix': '', 'version': '',
			'mode': None,
			'loss_percs': {}, 		# Use this parameter to weight loss components
			}
	
	return train_params


def load_data_and_initialize_training(path_results, dataset_name, path_dataset, 
								   path_annotations, mode, **kwargs):
	# Remove folders of non finished training
	title = 'Remove null trainings'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
	train_utils.remove_null_trainings(path_results, dataset_name)
	print('='*print_line)

	# Load train and val annotations
	np.random.seed(10101)
	with open(path_annotations[0]) as f: lines_train = f.readlines()
	with open(path_annotations[1]) as f: lines_val = f.readlines()
	num_train, num_val = len(lines_train), len(lines_val)
	np.random.shuffle(lines_train), np.random.shuffle(lines_val)
	np.random.seed(None)

	lines_train = [ ','.join([ path_dataset+img for img in ann.split(' ')[0].split(',') ]) \
						+ ' ' + ' '.join(ann.split(' ')[1:]) for ann in lines_train ]
	lines_val = [ ','.join([ path_dataset+img for img in ann.split(' ')[0].split(',') ]) \
						+ ' ' + ' '.join(ann.split(' ')[1:]) for ann in lines_val ]
	
	# If model use recurrent layers, calculate the recurrence lenght
	td_len = None if mode is None else len(lines_train[0].split(' ')[0].split(','))

	# Set batch size according to the model type
	if mode is None:
		batch_size_frozen = 32		  # 32
		batch_size_unfrozen = 4		 # note that more GPU memory is required after unfreezing the body
	else:
		batch_size_frozen = 8		  # 32
		batch_size_unfrozen = 2		 # note that more GPU memory is required after unfreezing the body


	# Initialize model folder
	title = 'Create and get model folders'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
	path_model = train_utils.create_model_folder(path_results, dataset_name)
	model_num = int(path_model.split('/')[-2].split('_')[-1])
	print(path_model)
	print('='*print_line)
	log('NEW TRAIN {}'.format(model_num))
	
	return lines_train, lines_val, \
			{'batch_size_frozen': batch_size_frozen, 'batch_size_unfrozen': batch_size_unfrozen,
			  'num_val': num_val, 'num_train': num_train, 'td_len': td_len, 'model_num': model_num,
			  'path_model': path_model}


def store_train_params(train_params):
	log('TRAIN PARAMS: {}'.format(train_params))
	print(train_params)
	with open(train_params['path_model'] + 'train_params.json', 'w') as f:
		json.dump(train_params, f)
	print("train_params stored as json")

	title = 'Excel params'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
	excel_resume = evaluate_model.get_excel_resume_full(train_params['path_model'], train_params, '', '', 
						  {}, {}, '', None)
	print(excel_resume)		
	pyperclip.copy(excel_resume)
	

def initialize_model(path_classes, path_anchors, path_model, input_shape, freeze_body, 
				  path_weights, td_len, mode, spp, loss_percs, **kwargs):

	class_names = ktrain.get_classes(path_classes)
	num_classes = len(class_names)
	anchors = ktrain.get_anchors(path_anchors)
	
	# Create model
	title = 'Create Keras model'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
	model = create_model(input_shape, anchors, num_classes, 
						 freeze_body=freeze_body,
						 weights_path=path_weights, td_len=td_len, mode=mode, 
						 spp=spp, loss_percs=loss_percs)
	
	# Store model architecture
	model_architecture = model.to_json()
	with open(path_model + 'architecture.json', 'w') as f:
		json.dump(model_architecture, f)
	print("Model architecture stored as json")
	print('='*print_line)
	
	# Training callbacks
	callbacks = {
			'logging': TensorBoard(log_dir = path_model),
			'checkpoint': ModelCheckpoint(path_model + 'weights/' + 'ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5',
										 monitor='val_loss', 
										 save_weights_only=True, 
										 save_best_only=True, 
										 period=1),
			'reduce_lr_1': ReduceLROnPlateau(monitor='loss', min_delta=0.5, factor=0.1, patience=4, verbose=1),
			'reduce_lr_2': ReduceLROnPlateau(monitor='val_loss', min_delta=0, factor=0.1, patience=4, verbose=1),
			'early_stopping': EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
		}

	log('MODEL CREATED')

	return model, callbacks, anchors, num_classes


# Train with frozen layers first, to get a stable loss.
def train_frozen_stage(model, callbacks, lines_train, lines_val, anchors, num_classes, 
					   path_model, num_train, num_val, input_shape, batch_size_frozen,
					   frozen_epochs, multi_scale, **kwargs):
	
	title = 'Train first stage'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
	log('TRAIN STAGE 1')
	optimizer = Adam(lr=1e-3)
	model.compile(optimizer = optimizer,
				  loss = {'yolo_loss': loss},		# use custom yolo_loss Lambda layer.
				  metrics = [
							 xy_loss, wh_loss, confidence_loss, 
							 confidence_loss_obj, confidence_loss_noobj, class_loss,
							 train_utils.get_lr_metric(optimizer),
							 ]	
				  )

	print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_frozen))
	model.fit_generator(
			data_generator_wrapper_custom(lines_train, batch_size_frozen, input_shape, 
								 anchors, num_classes, random=True, multi_scale=multi_scale),
			steps_per_epoch = max(1, num_train//batch_size_frozen),
			validation_data = data_generator_wrapper_custom(lines_val, batch_size_frozen, 
								   input_shape, anchors, num_classes, random=False, multi_scale=False),
			validation_steps = max(1, num_val//batch_size_frozen),
			epochs = frozen_epochs,
			initial_epoch = 0,
			callbacks=[callbacks['logging'], callbacks['checkpoint']])
	model.save_weights(path_model + 'weights/trained_weights_stage_1.h5')
	print('='*print_line)


# Unfreeze and continue training, to fine-tune.
def train_final_stage(model, callbacks, lines_train, lines_val, anchors, num_classes, 
					   path_model, num_train, num_val, input_shape, batch_size_unfrozen,
					   frozen_epochs, multi_scale, **kwargs):
	
	title = 'Train second stage'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
	log('TRAIN STAGE 2')
	# Unfreeze layers
	for i in range(len(model.layers)):
		model.layers[i].trainable = True
	print('Unfreeze all of the layers.')

	optimizer = Adam(lr=1e-4)
	model.compile(optimizer = optimizer,
				  loss = {'yolo_loss': loss},		# use custom yolo_loss Lambda layer.
				  metrics = [
							 xy_loss, wh_loss, confidence_loss, 
							 confidence_loss_obj, confidence_loss_noobj, class_loss,
							 train_utils.get_lr_metric(optimizer),
							 ]	
				  )

	print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_unfrozen))
	model.fit_generator(
		data_generator_wrapper_custom(lines_train, batch_size_unfrozen, input_shape, 
								anchors, num_classes, random=True, multi_scale=multi_scale),
		steps_per_epoch = max(1, num_train//batch_size_unfrozen),
		validation_data = data_generator_wrapper_custom(lines_val, batch_size_unfrozen, 
								input_shape, anchors, num_classes, random=False, multi_scale=False),
		validation_steps = max(1, num_val//batch_size_unfrozen),
		epochs = 500,
		initial_epoch = frozen_epochs,
		callbacks = [callbacks['logging'], callbacks['checkpoint'], 
					   callbacks['reduce_lr_2'], callbacks['early_stopping']])
	model.save_weights(path_model + 'weights/trained_weights_final.h5')
	print('='*print_line)
	

# Evaluate trained model
# If best_weights is True, evaluates with the model weights that get the lower mAP
# If best_weights is False, evaluates with the last stored model weights
# If best_weights is Nonw, evaluates with the kind of weights that get the highest mAP
# The reference metric is mAP@50
# score_train and score_val specify the minimum score to filter out predictions before evaluation
# 	increase this value for large datasets
def evaluate_training(train_params, best_weights, score_train=evaluate_model.MIN_SCORE, 
					  score_val=0, iou=0.5, **kwargs):
	log('EVALUATING')
	
	model_num = train_params['model_num']
	dataset_name = train_params['dataset_name']
	path_results = train_params['path_results']

	annotation_files = [(score_val, 1), (score_train, 0)]
	
	times = [ None for i in range(max([ af for _,af in annotation_files ])+1) ]
	eval_stats_arr = [ {} for i in range(max([ af for _,af in annotation_files ])+1) ]
	
	for score, num_annotation_file in annotation_files:
	
		print('='*80)
		print('dataset = {}, num_ann = {}, model = {}'.format(
				dataset_name, num_annotation_file, model_num))	
		print('='*80)
		
		times[num_annotation_file] = time.time()
	
		if best_weights is None:
			_, _, _, _, _, eval_stats_f, _, loss_f = evaluate_model.main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file, 
					plot=True, full=True, best_weights=False)
			
			_, _, _, _, resume, eval_stats_t, _, loss_t = evaluate_model.main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file, 
					plot=True, full=True, best_weights=True)
	
	
			if eval_stats_t['total'][1] >= eval_stats_f['total'][1]:
				print('Model {} con best_weights: {:.2f}'.format(model_num, eval_stats_t['total'][1]*100))
				eval_stats_arr[num_annotation_file] = eval_stats_t
				best_weights = True
			else:
				print('Model {} con stage_2: {:.2f}'.format(model_num, eval_stats_f['total'][1]*100))
				eval_stats_arr[num_annotation_file] = eval_stats_f
				best_weights = False
		else:
			_, _, _, _, resume, eval_stats, _, loss_t = evaluate_model.main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file, 
					plot=True, full=True, best_weights=best_weights)
			
			eval_stats_arr[num_annotation_file] = eval_stats
					
		times[num_annotation_file] = (time.time() - times[num_annotation_file])/60
	
	
	eval_stats_train, eval_stats_val = eval_stats_arr

	print('Training evaluation time: {:.2f}'.format(times[0]))
	print('Validation evaluation time: {:.2f}'.format(times[1]))
	print('Train mAP@50: {:.2f}'.format(eval_stats_train['total'][1]))
	print('Validation mAP@50: {:.2f}'.format(eval_stats_val['total'][1]))
	
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
	print("Model_folder	Version	Input Shape	Annotations	Anchors	Pretraining	Frozen	Mode	Training Time	Train Loss	Val Loss	mAP	mAP@50	R@10	mAP	mAP@50	mAP@75	R@10	mAP S	mAP M	mAP L")
	print(full_resume)		
	pyperclip.copy(full_resume)
	
	log(full_resume)
	log('PIPELINE ENDED\n')


# %%

def main(train_params):
	
	lines_train, lines_val, tp = load_data_and_initialize_training(**train_params)
	train_params.update(tp)
	store_train_params(train_params)
	
	model, callbacks, anchors, num_classes = initialize_model(**train_params)
	
	train_frozen_stage(model, callbacks, lines_train, lines_val, anchors, num_classes, **train_params)
	train_final_stage(model, callbacks, lines_train, lines_val, anchors, num_classes, **train_params)

	train_utils.remove_worst_weights(train_params['path_model'])
	
	evaluate_training(train_params, best_weights=None)



if __name__ == "__main__":
	train_params = get_train_params_from_args()
	main(train_params)






