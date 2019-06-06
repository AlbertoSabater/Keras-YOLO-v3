#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:18:24 2019

@author: asabater
"""

import os
os.chdir(os.getcwd() + '/..')


# %%

from eyolo import EYOLO
from PIL import Image
import cv2
from tqdm import tqdm
import train_utils
import json
import sys
import numpy as np
import pickle


def get_best_weights(model_folder, train_params, score=0, iou=0.5):
	annotations_file = train_params['path_annotations'][1]

	eval_t = '{}stats_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
	eval_f = '{}stats_stage2_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
	eval_t = json.load(open(eval_t, 'r'))
	eval_f = json.load(open(eval_f, 'r'))
	
	if eval_t['total'][1] > eval_f['total'][1]:
		# best_weights
		return train_utils.get_best_weights(model_folder)
	else:
		# Final weights
		return model_folder + 'weights/trained_weights_final.h5'


#video_file = '/mnt/hdd/datasets/home_videos/2019_06_04.MP4'
video_file = '/mnt/hdd/datasets/home_videos/2019_06_04_censored.avi'


#model_nums, dataset_name, score = [64, 66, 62], 'adl', 0
model_nums, dataset_name, score = [56, 52, 60], 'adl', 0
#model_nums, dataset_name, score = [0,1], 'kitchen', 0.005
models = {}
preds = {}

for model_num in model_nums:
	
	preds_file = '/'.join(video_file.split('/')[:-1]) + '/predictions/' + \
					video_file.split('/')[-1][:-4] + \
					'_model{}_ms{}'.format(model_num, score) + '.pckl'

	
	model_folder = train_utils.get_model_path('/mnt/hdd/egocentric_results/', dataset_name, model_num)
	train_params = json.load(open(model_folder + 'train_params.json', 'r'))
	classes_path = train_params['path_classes']
	anchors_path = train_params['path_anchors']
	model_image_size = train_params['input_shape']
	model_path = get_best_weights(model_folder, train_params, score=score)
	
	model = EYOLO(
                model_image_size = (416, 416),
                model_path = model_path,
                anchors_path = anchors_path,
                classes_path = classes_path,
                score = score,
                iou = 0.5,
#                gpu_num = 2
                td_len = train_params['td_len'],
                mode = train_params['mode'],
				spp = train_params.get('spp', False)
            )
	
	
	if os.path.isfile(preds_file):
		preds[model_num] = pickle.load(open(preds_file, 'rb'))
		print('*', preds_file + ' loaded')

	else:	
		cap = cv2.VideoCapture(video_file)
		num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	
	#	num_frames = 50
		preds[model_num] = []
		for i in tqdm(range(num_frames), total=num_frames, file=sys.stdout):
			ret, frame = cap.read()
			if ret:
				frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
				boxes, scores, classes = model.get_prediction(frame)
				preds[model_num].append((i, boxes, scores, classes))
			else: break
		
		pickle.dump(preds[model_num], open(preds_file, 'wb'))
		print('*', preds_file + ' stored')
		cap.release()

	model.yolo_model = None
	models[model_num] = model


# %%

from skvideo.io import FFmpegWriter


perc = 0.5
frame_gap = 10
min_score = 0.2
output_filename = '{}_{}_{}_ms{}.mp4'.format(video_file[:-4], dataset_name,
				   '|'.join([ str(mn) for mn in model_nums ]), min_score)

print('='*80)
print('Storing video: ' + output_filename)

writer = None
cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#num_frames = 30
for i in tqdm(range(num_frames), total=num_frames, file=sys.stdout):
	ret, frame = cap.read()
	frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	
	frame_set = []
	for model_num in model_nums:
		_, boxes, scores, classes = preds[model_num][i]
		mask = [ True if score >= min_score else False for score in scores ]
		boxes = [ r for r,m in zip(boxes, mask) if m ]
		scores = [ r for r,m in zip(scores, mask) if m ]
		classes = [ r for r,m in zip(classes, mask) if m ]
		
		frame_set.append(models[model_num].print_boxes(frame.copy(), boxes, classes, scores))

	frame_set = [ f.resize((int(f.size[0]*perc), int(f.size[1]*perc)), Image.ANTIALIAS) for f in frame_set ]
	frame_set = [ np.array(f) for f in frame_set ]
	f_shape = frame_set[0].shape
	frame = np.zeros((f_shape[0], f_shape[1]*len(frame_set) + (len(frame_set)-1)*frame_gap, 3), 
				  np.uint8)
	
	for num_f, f in enumerate(frame_set):
		init_w = f_shape[1] * num_f + (num_f)*frame_gap
		end_w = init_w + f_shape[1]
		frame[:, init_w:end_w, :] = f
	
#	frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	
	if writer is None:
		writer = FFmpegWriter(output_filename,
								   inputdict={'-r': str(fps)},
								   outputdict={'-r': str(fps)})

	writer.writeFrame(frame)

writer.close()
print(output_filename + ' video stored')
	
#	cv2.imwrite('res.png', frame)
#	break

#	cv2.imshow('result', frame)
#	if cv2.waitKey(1) & 0xFF == ord('q'):
#            break


