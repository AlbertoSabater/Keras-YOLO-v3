#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:28:33 2019

@author: asabater
"""

import os
os.chdir(os.getcwd() + '/..')

# %%

import train_utils
import json
import pandas as pd
from playground.test_annotations import print_annotations
#from playground.check_predictions import print_annotations
import colorsys
import cv2
import numpy as np
from evaluate_model import MIN_SCORE

import sys
sys.path.append('keras_yolo3/')
sys.path.append('keras_yolo3/yolo3/')
import keras_yolo3.train as ktrain


path_results = '/mnt/hdd/egocentric_results/'
dataset_name = 'adl'
path_dataset = '/home/asabater/projects/ADL_dataset/'
#num_ann, score = 0, MIN_SCORE
num_ann, score = 1, 0
min_score = 0.3

model_nums = {45:'320', 56:'416', 44:'608'}
#model_nums = {18:'320', 57:'416', 17:'608'}
preds_num = {}

for model_num, label in model_nums.items():
	
	model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
	train_params = json.load(open(model_folder + 'train_params.json', 'r'))
	annotations_file = train_params['path_annotations'][num_ann]

	preds_filename = '{}preds_stage2_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, 0.5)
	preds = json.load(open(preds_filename, 'r'))
	preds = [ p for p in preds if p['score'] >= min_score ]
	preds_num[label] = pd.DataFrame(preds)


# %%

#preds_count = { k:pd.DataFrame(v.groupby('image_id').score.sum() / v.groupby('image_id').score.size()) for k,v in preds_num.items() }
#for k,v in preds_count.items(): preds_count[k].columns = [k]
preds_count = { k:pd.DataFrame(v.groupby('image_id').size(), columns=[k]) for k,v in preds_num.items() }

preds_count = preds_count['320'].join(preds_count['416'], how='outer').join(preds_count['608'], how='outer')
preds_count = preds_count.fillna(0)
preds_count['score'] = preds_count.apply(lambda r: (r['416']-r['320']) + (r['608']-r['416']) + (r['608']-r['320']), axis=1)
preds_count = preds_count.sort_values('score', ascending=False)


# %%

def get_preds(image_id):
	results = []
	for model_num, label in model_nums.items():
		
		print('='*80)
		print(preds_num[label][preds_num[label]['image_id'] == image_id][['bbox', 'score']])
		boxes = preds_num[label][preds_num[label]['image_id'] == image_id]
#		boxes = boxes.apply(lambda r: ','.join([ str(b) for b in r['bbox'] ]) + ',' + str(r['category_id']), axis=1).tolist()
		
		sample = '{}{}.jpg'.format(path_dataset, image_id)
		for i,r in boxes.iterrows():
			bbox = r['bbox']
			sample += ' {},{},{},{},{}'.format(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], r['category_id'])
	
		result = print_annotations(sample, class_names, colors, perc=0.4)
#		result = print_annotations(sample, perc=0.4, class_names=class_names)
		results.append(result)
	return results

class_names = ktrain.get_classes(train_params['path_classes'])


hsv_tuples = [(x / len(class_names), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))


preds_count = preds_count[preds_count['320'] != 0]
#image_id = preds_count.index[1]

for image_id in preds_count.index:
	print('='*80)
	print('='*80)
	print(image_id)

	results = get_preds(image_id)
	
	cv2.imshow("result", np.hstack(results))
	cv2.waitKey(1)
	input('next')


# %%

path_results = '/mnt/hdd/egocentric_results/figuras_05.2019/'

#image_id = 'ADL_frames/P_18/00033414'
#image_id = 'ADL_frames/P_19/00096786'
#image_id = 'ADL_frames/P_19/00028290'
#image_id = 'ADL_frames/P_19/00096906'
#image_id = 'ADL_frames/P_20/00097566'


#image_id = 'ADL_frames/P_18/00096186'
image_id = 'ADL_frames/P_19/00028290'

results = get_preds(image_id)
for img, label in zip(results, model_nums.values()):
#	img = Image.fromarray(img)
#	img.save('{}{}_{}.png'.format(path_results, label, image_id.replace('/', '_')))
	cv2.imshow(label, img)
	cv2.waitKey(1)
	cv2.imwrite('{}img_size_comp/{}_{}.png'.format(path_results, label, image_id.replace('/', '_')), img)

