#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:06:07 2019

@author: asabater
"""

import json
import numpy as np

#from eyolo import EYOLO
#from pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
#import pyperclip
#import matplotlib.pyplot as plt



import random
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import colorsys


def print_annotations(sample, perc, class_names):
	sample = [ s for s in sample.split(' ') if s != '' ]
		
#		'/media/asabater/hdd/datasets/imagenet_vid/ILSVRC2015/Data/VID/val/' + 
#	image = cv2.imread(sample[0])
#	image = cv2.resize(image,(int(image.shape[1]*perc),int(image.shape[0]*perc)))
#	image = Image.fromarray(image)
	image = sample[0].split(',')
	image = image[len(image)//2]
#	image = Image.open('/home/asabater/projects/ADL_dataset/' + image)
	image = Image.open(image)
	image = image.resize((int(image.size[0]*perc),int(image.size[1]*perc)), Image.ANTIALIAS)

	
	draw = ImageDraw.Draw(image)
	thickness = (image.size[0] + image.size[1]) // 600

	font = ImageFont.truetype(font='keras_yolo3/font/FiraMono-Medium.otf',
				size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
	
	if len(sample) > 1:
		for box in sample[1:]:
			for i in range(thickness):

				left, bottom, right, top, c = box.split(',')
				left, bottom, right, top, c = int(left)*perc, int(bottom)*perc, int(right)*perc, int(top)*perc, int(c)
#				color = colors[c]
				color = (250,0,0)
#					print(left, bottom, right, top, c)
				
				draw.rectangle(
						[int(left) + i, int(top) + i, int(right) - i, int(bottom) - i],
						outline = color)
				
				c = class_names[c]
#				c = 'clase'
				label_size = draw.textsize(c, font)
				if top - label_size[1] >= 0:
					text_origin = np.array([left, top - label_size[1]])
				else:
					text_origin = np.array([left, top + 1])
				draw.rectangle(
						[tuple(text_origin), tuple(text_origin + label_size)], 
						fill = (255,255,255),
						outline = color)
				draw.text(text_origin, c, fill=(0, 0, 0), font=font)
	
	
	result = np.asarray(image)
	result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
#	cv2.putText(result, text=sample[0].split('/')[-2], org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#					fontScale=0.50, color=(255, 0, 0), thickness=2)
		
#	cv2.imshow("result", result)
	return result



def main():
	
	import os
	os.chdir(os.getcwd() + '/..')
	
	# %%
	
	import train_utils
	import pandas as pd

	import sys
	sys.path.append('keras_yolo3/')
	sys.path.append('keras_yolo3/yolo3/')
	import keras_yolo3.train as ktrain

	

	##path_dataset = '/mnt/hdd/datasets/VOC/'
	#path_results = '/mnt/hdd/egocentric_results/'
	##path_classes = './dataset_scripts/coco/coco_classes.txt'
	#path_dataset = '/mnt/hdd/datasets/coco/'
	#path_classes = './dataset_scripts/adl/adl_classes_v3_8.txt'
	#
	##gt_filename = '/home/asabater/projects/Egocentric-object-detection/dataset_scripts/coco/annotations_coco_val_coco.json'
	##gt = json.load(open(gt_filename, 'r'))['annotations']
	#
	#score = 0.4
	#gt_filename = path_results + 'default/voc_yolo_model_0/preds_annotations_coco_val_score{}_iou.json'.format(score)
	##gt_filename = path_results + 'adl/0414_2252_model_20/preds_annotations_adl_val_v3_8_r_fd10_fsn1_score0_iou0.5.json'
	##gt_filename = path_results + 'adl/0414_2252_model_20/preds_annotations_adl_train_v3_8_r_fd10_fsn1_score5e-05_iou0.5.json'
	#gt = json.load(open(gt_filename, 'r'))
	
	#path_classes = './dataset_scripts/voc/voc_classes.txt'
	#path_dataset = '/mnt/hdd/datasets/VOC/'
	#score = 0.4
	#gt_filename = path_results + 'voc/0326_1706_model_0/preds_annotations_voc_val_score{}_iou.json'.format(score)
	#gt = json.load(open(gt_filename, 'r'))
		
	
	
	path_results = '/mnt/hdd/egocentric_results/'
	dataset_name = 'adl'
	path_dataset = '/home/asabater/projects/ADL_dataset/'
	model_num = 62
	
	model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
	train_params = json.load(open(model_folder + 'train_params.json', 'r'))
	annotations_file = train_params['path_annotations'][1]
	preds_filename = '{}preds_stage2_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], 0, 0.5)
	gt = json.load(open(preds_filename, 'r'))
	
	
	# %%
	
	preds = pd.DataFrame([ p for p in gt if p['score'] >= 0.3 ])
	
	preds = [ {'image_id': i, 'bboxes': g.bbox.tolist(), 'category_id': g.category_id.tolist()} 
			for i,g in preds.groupby('image_id') ]
	
	# %%
	
	from playground.test_annotations import print_annotations

	class_names = ktrain.get_classes(train_params['path_classes'])

	hsv_tuples = [(x / len(class_names), 1., 1.)
	              for x in range(len(class_names))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

	
	preds = preds[random.randint(0, len(preds)):]
	
	for pred in preds:
		
		sample = '{}{}.jpg'.format(path_dataset, pred['image_id'])
		for bbox, cat in zip(pred['bboxes'], pred['category_id']):
			sample += ' {},{},{},{},{}'.format(bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], cat)
		
		print(sample)
		
		
		pass
	
#		result = print_annotations(sample, 0.7, class_names)
		result = print_annotations(sample, class_names, colors, perc=0.7)
		cv2.imshow("result", result)
		
		if cv2.waitKey(200) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		
		
if __name__ == "__main__": main()



# %%


