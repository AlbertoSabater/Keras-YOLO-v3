#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 10:18:24 2019

@author: asabater
"""

import os
os.chdir(os.getcwd() + '/..')


# %%

import os
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
#video_file = '/mnt/hdd/datasets/home_videos/2019_06_07_head.avi'
#video_file = '/mnt/hdd/datasets/home_videos/P_18_12.MP4'
#video_file = '/mnt/hdd/datasets/home_videos/P_19_12.MP4'
#video_file = '/mnt/hdd/datasets/home_videos/P_20_12.MP4'


model_nums, dataset_name, score = {64: 'original', 66: 'v2', 62: 'v3'}, 'adl', 0
#model_nums, dataset_name, score = {56:'v2 old', 52: 'v2', 60: 'v2 | xy: 5, wh: 5, conf_obj: 2.3'}, 'adl', 0
#model_nums, dataset_name, score = {57:'v3 old', 53: 'v3 new'}, 'adl', 0
#model_nums, dataset_name, score = {0: 'cv1_17', 1: 'cv2_18'}, 'kitchen', 0.005
models = {}
preds = {}

for model_num, _ in model_nums.items():
	
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
from PIL import Image, ImageFont, ImageDraw


font_size = 70
height = 720
num_rows, num_cols = 3,1
font = ImageFont.truetype(font='keras_yolo3/font/FiraMono-Medium.otf', size=font_size)

#perc = 0.5
frame_gap = 10
min_score = 0.2
output_filename = '{}_{}_{}_ms{}_r{}_c{}.mp4'.format(video_file[:-4], dataset_name,
				   '|'.join([ str(mn) for mn, _ in model_nums.items() ]), min_score,
				   num_rows, num_cols)
#output_filename = 'aaa.mp4'

save_frames, write_frame = [(104, 104)], True
#save_frames, write_frame = [], False

print('='*80)
print('Storing video: ' + output_filename)

writer = None
cap = cv2.VideoCapture(video_file)
fps = cap.get(cv2.CAP_PROP_FPS)

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#num_frames = 30
for i in tqdm(range(num_frames), total=num_frames, file=sys.stdout):
	ret, frame = cap.read()
	if not ret:
		print('not ret'); continue
	
#	if write_frame and i not in save_frames: continue
#	if write_frame and (i < 90 or i > 120): continue
	if write_frame and not any([ True if i >= min_f and i <= max_f else False for min_f, max_f in save_frames ]): continue
	
	frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	
	frame_set = []
	for model_num, label in model_nums.items():
		_, boxes, scores, classes = preds[model_num][i]
		mask = [ True if score >= min_score else False for score in scores ]
		boxes = [ r for r,m in zip(boxes, mask) if m ]
		scores = [ r for r,m in zip(scores, mask) if m ]
		classes = [ r for r,m in zip(classes, mask) if m ]
		frame_model = models[model_num].print_boxes(frame.copy(), boxes, classes, 
					  scores, label_size=4)
		draw = ImageDraw.Draw(frame_model)
		draw.text((0,0), label, fill=(0, 0, 0), font=font)
		
		frame_set.append(frame_model)

	frame_set = [ f.resize((int(f.size[0]*height/f.size[1]), height), Image.ANTIALIAS) for f in frame_set ]
	frame_set = [ np.array(f) for f in frame_set ]
	f_shape = frame_set[0].shape
	frame = np.full((f_shape[0]*num_rows + (num_rows-1)*frame_gap, 
				  f_shape[1]*num_cols + (num_cols-1)*frame_gap, 
				  3), 255, np.uint8)
	
	for num_f, f in enumerate(frame_set):
		cell_w, cell_h = num_f % num_cols, num_f//num_cols % num_rows
		init_w = f_shape[1] * cell_w + cell_w*frame_gap
		end_w = init_w + f_shape[1]
		init_h = f_shape[0] * cell_h + cell_h*frame_gap
		end_h = init_h + f_shape[0]
		frame[init_h:end_h, init_w:end_w, :] = f
	
	
	if write_frame:
		img = Image.fromarray(frame)
		img.save("{}.jpeg".format(i))
		print("{}.jpeg".format(i), " saved")
		continue

	frame = Image.fromarray(frame)
	draw = ImageDraw.Draw(frame)
	draw.text((height,0), str(i), fill=(0, 0, 0), font=font)
	frame = np.array(frame)
	
	if writer is None:
		writer = FFmpegWriter(output_filename,
								   inputdict={'-r': str(fps)},
								   outputdict={'-r': str(fps)})
	writer.writeFrame(frame)

if not write_frame: writer.close()
print(output_filename + ' video stored')
	
#	cv2.imwrite('res.png', frame)
#	break

#	cv2.imshow('result', frame)
#	if cv2.waitKey(1) & 0xFF == ord('q'):
#            break


