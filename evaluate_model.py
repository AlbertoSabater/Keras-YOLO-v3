#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:53:31 2019

@author: asabater
"""

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
#os.environ["CUDA_VISIBLE_DEVICES"] = "0";  

import sys
sys.path.append('keras_yolo3/')
sys.path.append('keras_yolo3/yolo3/')

import json
import os
import numpy as np
import pandas as pd
import keras_yolo3.train as ktrain
#from keras.models import model_from_json
import train_utils

from tensorboard.backend.event_processing import event_accumulator
import datetime
import time

import cv2
from PIL import Image
from tqdm import tqdm

from eyolo import EYOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pyperclip
import matplotlib.pyplot as plt
import time


MIN_SCORE = 0.00005


# Returns training time, best train loss and val loss
def get_train_resume(model_folder):
	tb_files = [ model_folder + f for f in os.listdir(model_folder) if f.startswith('events.out.tfevents') ]
	
	train_losses, val_losses = [], []
	times = []
	for tbf in tb_files:
#		print(tbf)
		try:
			ea = event_accumulator.EventAccumulator(tbf).Reload()
			train_losses += [ e.value for e in ea.Scalars('loss') ]
			val_losses += [ e.value for e in ea.Scalars('val_loss') ]
			times += [ e.wall_time for e in ea.Scalars('val_loss') ]
		except: continue
	
	val_loss = min(val_losses)
	train_loss = train_losses[val_losses.index(val_loss) ]
	
	train_init, train_end = min(times), max(times)
	
	train_init = datetime.datetime.fromtimestamp(train_init)
	train_end = datetime.datetime.fromtimestamp(train_end)
	
	train_diff = (train_end - train_init)
	train_diff = '{}d {:05.2f}h'.format(train_diff.days, train_diff.seconds/3600)
	
	return train_diff, train_loss, val_loss


# Stores an json file with the predictions calculated by the given annotations_file
# Uses best weights or last calculated weights depending on best_weights
def predict_and_store_from_annotations(model_folder, train_params, annotations_file, 
									   preds_filename, score, iou, best_weights=True):

	if os.path.isfile(preds_filename): return None, -1
	
	if best_weights:
		model_path = train_utils.get_best_weights(model_folder)
	else:
		model_path = model_folder + 'weights/trained_weights_final.h5'
	
	model = EYOLO(
					model_image_size = tuple(train_params['input_shape']),
					model_path = model_path,
					classes_path = train_params['path_classes'],
					anchors_path = train_params['path_anchors'],
					score = score,	# 0.3
					iou = iou,	  # 0.5
					td_len = train_params.get('td_len', None),
					mode = train_params.get('mode', None),
					spp = train_params.get('spp', False)
				)
	
	
	# Create pred file
	with open(annotations_file, 'r') as f: annotations = f.read().splitlines()
	preds = []
	
	t = time.time()
	total = len(annotations)
	for ann in tqdm(annotations[:total], total=total, file=sys.stdout):
		img = ann.split()[0]
		
#		image = cv2.imread(train_params['path_dataset'] + img)
#		image = Image.fromarray(image)
		images = [ Image.open(train_params['path_dataset'] + img) for img in img.split(',') ]
		img_size = images[0].size
#		images = images[0] if len(images) == 1 else np.stack(images)
		boxes, scores, classes = model.get_prediction(images)
		
		for i in range(len(boxes)):
			left, bottom, right, top = [ int(b) for b in boxes[i].tolist() ]
			left, bottom = max(0, left), max(0, bottom)
			right, top = min(img_size[1], right), min(img_size[0], top)
			width = top-bottom
			height = right-left
			preds.append({
							'image_id': img[:-4],
							'category_id': int(classes[i]),
#							'bbox': [ x_min, y_min, width, height ],
							'bbox': [ bottom, left, width, height ],
							'score': float(scores[i]),
						})
		
	fps = len(annotations)/(time.time()-t)
	json.dump(preds, open(preds_filename, 'w'))
	
	return model, fps


# Perform mAP evaluation given the preddictions file and the groundtruth filename
# Evaluation is performed to all the datset, pero class and per subdataset (if exists)
def get_full_evaluation(annotations_file, preds_filename, eval_filename, class_names, full):
	
	eval_filename_full = eval_filename.replace('.json', '_full.json')
	ann = annotations_file[:-4] + '_coco.json'
	
	if full and os.path.isfile(eval_filename_full):
		# If full exists return full
		print(' * Loading:', eval_filename_full)
		eval_stats =  json.load(open(eval_filename_full, 'r'))
		videos = [ k for k in eval_stats.keys() if not k.startswith('cat_') and not k.startswith('total') ]
		return eval_stats, videos
	elif os.path.isfile(eval_filename):
		# If regular eval exists load id
		print(' * Loading:', eval_filename)
		eval_stats =  json.load(open(eval_filename, 'r'))
		
		if not full:
			videos = [ k for k in eval_stats.keys() if not k.startswith('cat_') and not k.startswith('total') ]
			return eval_stats, videos	  
	else:
		# if no eval has been performed, initialize the dictionary and COCO
		eval_stats = {}

	cocoGt = COCO(ann)
	cocoDt = cocoGt.loadRes(preds_filename)
	cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')		
	
	if 'total' not in eval_stats:
		print(' * Evaluating total')
		cocoEval.evaluate()
		cocoEval.accumulate()
		cocoEval.summarize()
		eval_stats['total'] = cocoEval.stats.tolist()
		json.dump(eval_stats, open(eval_filename, 'w'))
		
	if not full:
		# if not full return regular eval if not, continue the evaluation
		videos = [ k for k in eval_stats.keys() if not k.startswith('cat_') and not k.startswith('total') ]
		return eval_stats, videos  
	
	else:
	
		params = cocoEval.params
		imgIds = params.imgIds
		catIds = params.catIds
		
		# Evaluate each category
		for c in range(len(class_names)):
			print(' * Evaluationg the category |{}|'.format(class_names[c]))
			params.catIds = [c]
			cocoEval.evaluate()
			cocoEval.accumulate(params)
			cocoEval.summarize()
			eval_stats['cat_{}'.format(c)] = cocoEval.stats.tolist()
			params.catIds = catIds
		
		
		# Evaluate each video and each category per video
		videos = sorted(list(set([ i.split('/')[-2] for i in imgIds ])))
		for v in videos:
			print(' * Evaluating the video |{}|'.format(v))
			eval_stats[v] = {}
			v_imgIds = [ i for i in imgIds if v in i ]
			params.imgIds = v_imgIds
			cocoEval.evaluate()
			cocoEval.accumulate(params)
			cocoEval.summarize()
			eval_stats[v]['total'] = cocoEval.stats.tolist()
			
			for c in range(len(class_names)):
				print(' * Evaluationg the category |{}| in the video |{}|'.format(class_names[c], v))
				params.catIds = [c]
				cocoEval.evaluate()
				cocoEval.accumulate(params)
				cocoEval.summarize()
				eval_stats[v]['cat_{}'.format(c)] = cocoEval.stats.tolist()
				params.catIds = catIds
			
			params.imgIds = imgIds
			params.catIds = catIds
			
		json.dump(eval_stats, open(eval_filename_full, 'w'))

	return eval_stats, videos


def get_excel_resume(model_folder, train_params, train_loss, val_loss, eval_stats, train_diff, fps, score, iou):
	result = '{model_folder}\t{input_shape}\t{annotations}\t{anchors}\t{pretraining}\t{frozen_training:d}\t{training_time}'.format(
				model_folder = '/'.join(model_folder.split('/')[-2:]), 
				input_shape = train_params['input_shape'],
				annotations = train_params['path_annotations'],
				anchors = train_params['path_anchors'],
				pretraining = train_params['path_weights'],
				frozen_training = train_params['freeze_body'],
				training_time = train_diff
			)
	
	result += '\t{train_loss:.5f}\t{val_loss:.5f}'.format(train_loss=train_loss, val_loss=val_loss).replace('.', ',')
	result += '\t{score:.5f}\t{iou:.2f}'.format(score=score, iou=iou).replace('.', ',')
	
	result += '\t{mAP}\t{mAP50}\t{mAP75}\t{mAPS}\t{mAPM}\t{mAPL}'.format(
				mAP=eval_stats['total'][0]*100, mAP50=eval_stats['total'][1]*100, 
				mAP75=eval_stats['total'][2]*100, mAPS=eval_stats['total'][3]*100, 
				mAPM=eval_stats['total'][4]*100, mAPL=eval_stats['total'][5]*100, 
			).replace('.', ',')
	
	result += '\t{fps:.2f}'.format(fps=fps)

	return result


def get_excel_resume_full(model_folder, train_params, train_loss, val_loss, 
						  eval_stats_train, eval_stats_val, train_diff, best_weights):
	
	if 'egocentric_results' in train_params['path_weights']:
		path_weights = '/'.join(train_params['path_weights'].split('/')[4:6])
	else:
		path_weights = train_params['path_weights'] 
	
	tiny_version = len(ktrain.get_anchors(train_params['path_anchors'])) == 6
	if tiny_version: model = 'tiny'
	elif train_params.get('spp', False): model = 'spp'
	else: model = 'yolo'
	
	mode = '{} | {} | {}'.format(
			'bw' if best_weights else 's2',
			model,
			train_params['mode'] if train_params.get('mode', None) is not None else '-')
		
	result = '{model_folder}\t{version}\t{input_shape}\t{annotations}\t{anchors}\t{pretraining}\t{frozen_training:d}\t{mode}\t{training_time}'.format(
				model_folder = '/'.join(model_folder.split('/')[-2:]), 
				version = train_params.get('version', ''),
				input_shape = 'multi_scale' if train_params.get('multi_scale', False) else train_params['input_shape'],
				annotations = train_params['path_annotations'],
				anchors = train_params['path_anchors'],
				pretraining = path_weights,
				frozen_training = train_params['freeze_body'],
				mode = mode,
				training_time = train_diff
			)
	
	result += '\t{train_loss}\t{val_loss}'.format(
				train_loss=train_loss, val_loss=val_loss).replace('.', ',')

	result += '\t{mAPtrain:.5f}\t{mAP50train:.5f}\t{R100train:.5f}'.format(
				mAPtrain = eval_stats_train['total'][0]*100 if 'total' in eval_stats_train else 0,
				mAP50train = eval_stats_train['total'][1]*100 if 'total' in eval_stats_train else 0,
				R100train = eval_stats_train['total'][7]*100 if 'total' in eval_stats_train else 0,
			).replace('.', ',')
	result += '\t{mAPval:.5f}\t{mAP50val:.5f}\t{mAP75val:.5f}\t{R100val:.5f}'.format(
				mAPval = eval_stats_val['total'][0]*100 if 'total' in eval_stats_val else 0,
				mAP50val = eval_stats_val['total'][1]*100 if 'total' in eval_stats_val else 0,
				mAP75val = eval_stats_val['total'][2]*100 if 'total' in eval_stats_val else 0,
				R100val = eval_stats_val['total'][7]*100 if 'total' in eval_stats_val else 0,
			).replace('.', ',')

	result += '\t{mAPS}\t{mAPM}\t{mAPL}'.format(
				mAPS=eval_stats_val['total'][3]*100 if 'total' in eval_stats_val else 0, 
				mAPM=eval_stats_val['total'][4]*100 if 'total' in eval_stats_val else 0, 
				mAPL=eval_stats_val['total'][5]*100 if 'total' in eval_stats_val else 0, 
			).replace('.', ',')
	
	return result


def plot_prediction_resume(eval_stats, videos, class_names, by, annotations_file, model_num, plot):
	occurrences = {}
	for v in videos:
		v_res = {}
		for c in range(len(class_names)):
			v_res[class_names[c]] = eval_stats[v]['cat_{}'.format(c)][1]
		occurrences[v] = v_res
		
	
	occurrences = pd.DataFrame.from_dict(occurrences, orient='index')
	if by == 'video': occurrences = occurrences = occurrences.transpose()
	occurrences = occurrences.replace({-1: np.nan})
	
	meds = occurrences.mean()
	meds = meds.sort_values(ascending=False)
	occurrences = occurrences[meds.index]
	
	ann = 'Train' if 'train' in annotations_file else 'Val'
	
	if plot:
		if by == 'video':
			keys = meds.index
			bars = [ eval_stats[k]['total'][1] for k in keys ]
		else:
			keys = [ 'cat_{}'.format(class_names.index(k)) for k in meds.index ]
			bars = [ eval_stats[k][1] for k in keys ]
		occurrences.boxplot(figsize=(20,7));
		plt.bar(range(1, len(bars)+1), bars, alpha=0.4)
		plt.axhline(eval_stats['total'][1], c='y', label="Total mAP@50")
		plt.xticks(rotation=50);
		plt.suptitle('Model {} | {} | Boxcox by: {}'.format(model_num, ann, by), fontsize=20);
		plt.show();
	
	return occurrences



def main(path_results, dataset_name, model_num, score, iou, num_annotation_file=1, 
		 plot=True, full=True, best_weights=True):
	model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
	train_params = json.load(open(model_folder + 'train_params.json', 'r'))
	class_names = ktrain.get_classes(train_params['path_classes'])

	annotations_file = train_params['path_annotations'][num_annotation_file]
	if 'adl' in annotations_file and train_params.get('size_suffix', '') != '':
		annotations_file = annotations_file.replace(train_params.get('size_suffix', ''), '')
#	annotations_file = annotations_file.replace('.txt', '_pr416.txt')
		
	print(' * Exploring:', annotations_file)
	
	if best_weights:
		preds_filename = '{}preds_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
		eval_filename = '{}stats_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
	else:
		preds_filename = '{}preds_stage2_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
		eval_filename = '{}stats_stage2_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)

#	print(preds_filename)
	print('='*80)
	
	train_diff, train_loss, val_loss = get_train_resume(model_folder)
	model, fps = predict_and_store_from_annotations(model_folder, train_params, annotations_file, preds_filename, score, iou, best_weights=best_weights)
	occurrences = None
	eval_stats, videos = get_full_evaluation(annotations_file, preds_filename, eval_filename, class_names, full)
	
	resume = get_excel_resume(model_folder, train_params, train_loss, val_loss, eval_stats, train_diff, fps, score, iou)
	plot_prediction_resume(eval_stats, videos, class_names, 'video', annotations_file, model_num, plot);
	occurrences = plot_prediction_resume(eval_stats, videos, class_names, 'class', annotations_file, model_num, plot)
#	print(resume)
	
	return model, class_names, videos, occurrences, resume, eval_stats, train_params, (train_loss, val_loss)


# %%

def main_evaluation():
	
	path_results = '/mnt/hdd/egocentric_results/'
	dataset_name = 'adl'
	#score = 
	iou = 0.5
	plot = True
	full = True
#	best_weights = False
	
	model_num = 76

	if dataset_name == 'kitchen':
		annotation_files = [(0.005, 1), (0.005, 0)]
	else:
		annotation_files = [(0, 1), (MIN_SCORE, 0)]
	
	times = [ None for i in range(max([ af for _,af in annotation_files ])+1) ]
	eval_stats_arr = [ None for i in range(max([ af for _,af in annotation_files ])+1) ]
	videos_arr = [ None for i in range(max([ af for _,af in annotation_files ])+1) ]
	
	for score, num_annotation_file in annotation_files:		 # ,0
	
		print('='*80)
		print('dataset = {}, num_ann = {}, model = {}'.format(
				dataset_name, num_annotation_file, model_num))	
		print('='*80)
		
		times[num_annotation_file] = time.time()
	

		if num_annotation_file == 1:
			_, _, _, _, _, eval_stats_t, _, loss_t = main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file,
					plot=False, full=True, best_weights=True)
			
			_, _, videos, _, resume, eval_stats_f, tp, loss_f = main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file,
					plot=False, full=True, best_weights=False)
		
			print('Val. best_weigths mAP: {:.3f}'.format(eval_stats_t['total'][1]*100))
			print('Val. stage_2 mAP: {:.3f}'.format(eval_stats_f['total'][1]*100))
		
			if eval_stats_t['total'][1] >= eval_stats_f['total'][1]:
				eval_stats_arr[num_annotation_file] = eval_stats_t
				best_weights = True
			else:
				eval_stats_arr[num_annotation_file] = eval_stats_f
				best_weights = False

			videos_arr[num_annotation_file] = videos
			
			if num_annotation_file == 1:
				pyperclip.copy(resume)
		
		else:
			model, class_names, videos, occurrences, resume, eval_stats, train_params, loss = main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file,
					plot, full, best_weights=best_weights)	
			
			eval_stats_arr[num_annotation_file] = eval_stats
			videos_arr[num_annotation_file] = videos
			
			if num_annotation_file == 1:
				pyperclip.copy(resume)
	
		times[num_annotation_file] = (time.time() - times[num_annotation_file])/60
	
	eval_stats_train, eval_stats_val = eval_stats_arr
	videos_train, videos_val = videos_arr
	full_resume = get_excel_resume_full(resume.split('\t')[0], train_params, 
								 resume.split('\t')[7], resume.split('\t')[8], 
								 eval_stats_train, eval_stats_val, resume.split('\t')[6],
								 best_weights)
	
	print('='*80)
	print(full_resume)
	pyperclip.copy(full_resume)
	print('='*80)
	print('Val mAP@50: {:.3f}'.format(eval_stats_val['total'][1]*100))

	
if __name__ == "__main__": main_evaluation()

