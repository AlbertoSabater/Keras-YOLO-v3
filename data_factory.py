#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:20:48 2019

@author: asabater
"""

# https://github.com/zxwxz/keras-yolo3/blob/51d55bbe4c680a7f8ae2bf995dd4929809bb77dc/train.py

#from multiprocessing.dummy import Pool as ThreadPool
import threading
import numpy as np
import itertools
import time
from multiprocessing.dummy import Pool as ThreadPool

import sys
sys.path.append('keras_yolo3/')

from yolo3.model import preprocess_true_boxes
#from keras_yolo3.yolo3.utils import get_random_data



def data_generator_default(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
	'''data generator for fit_generator'''
	n = len(annotation_lines)
	i = 0
	while True:
		image_data = []
		box_data = []
		for b in range(batch_size):
			if i==0:
				np.random.shuffle(annotation_lines)
			image, box = get_random_data(annotation_lines[i], input_shape, random=random)
			image_data.append(image)
			box_data.append(box)
			i = (i+1) % n
		image_data = np.array(image_data)
		box_data = np.array(box_data)
		y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
		yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper_default(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
	n = len(annotation_lines)
	if n==0 or batch_size<=0: return None
	return data_generator_default(annotation_lines, batch_size, input_shape, anchors, num_classes, random)



# %%
###############################################################################
	
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('keras_yolo3/')
from keras_yolo3.yolo3.utils import get_random_data as get_random_data_old
import keras_yolo3.train as ktrain
from yolo3.model import preprocess_true_boxes

from tqdm import tqdm


def rand(a=0, b=1):
	return np.random.rand()*(b-a) + a

def get_random_data_custom(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
	'''random preprocessing for real-time data augmentation'''
	line = annotation_line.split()
	images = [ Image.open(i) for i in line[0].split(',') ]
#	image = Image.open(line[0])
	iw, ih = images[0].size
	h, w = input_shape
	box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

	if not random:
		# resize image
		scale = min(w/iw, h/ih)
		nw = int(iw*scale)
		nh = int(ih*scale)
		dx = (w-nw)//2
		dy = (h-nh)//2
		images_data = [ 0 for i in images ]
		if proc_img:
			images = [ img.resize((nw,nh), Image.BICUBIC) for img in images ]
			new_images = [ Image.new('RGB', (w,h), (128,128,128)) for i in images ]
			for i in range(len(images)): new_images[i].paste(images[i], (dx, dy))
			images_data = [ np.array(new_image)/255. for new_image in new_images ]

		# correct boxes
		box_data = np.zeros((max_boxes,5))
		if len(box)>0:
			np.random.shuffle(box)
			if len(box)>max_boxes: box = box[:max_boxes]
			box[:, [0,2]] = box[:, [0,2]]*scale + dx
			box[:, [1,3]] = box[:, [1,3]]*scale + dy
			box_data[:len(box)] = box

		images_data = images_data[0] if len(images_data) == 1 else np.stack(images_data)
		return images_data, box_data

	# resize image
	new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
	scale = rand(.25, 2)
	if new_ar < 1:
		nh = int(scale*h)
		nw = int(nh*new_ar)
	else:
		nw = int(scale*w)
		nh = int(nw/new_ar)
	images = [ image.resize((nw,nh), Image.BICUBIC) for image in images ]

	# place image
	dx = int(rand(0, w-nw))
	dy = int(rand(0, h-nh))
	new_images = [ Image.new('RGB', (w,h), (128,128,128)) for i in images ]
	for i in range(len(images)): new_images[i].paste(images[i], (dx, dy))
	images = new_images

	# flip image or not
	flip = rand()<.5
	if flip: 
		for i in range(len(images)):
			images[i] = images[i].transpose(Image.FLIP_LEFT_RIGHT)

	# distort image
	hue = rand(-hue, hue)
	sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
	val = rand(1, val) if rand()<.5 else 1/rand(1, val)
	images_data = []
	for i in range(len(images)):
		x = rgb_to_hsv(np.array(images[i])/255.)
		x[..., 0] += hue
		x[..., 0][x[..., 0]>1] -= 1
		x[..., 0][x[..., 0]<0] += 1
		x[..., 1] *= sat
		x[..., 2] *= val
		x[x>1] = 1
		x[x<0] = 0
		images_data.append(hsv_to_rgb(x)) # numpy array, 0 to 1

	# correct boxes
	box_data = np.zeros((max_boxes,5))
	if len(box)>0:
		np.random.shuffle(box)
		box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
		box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
		if flip: box[:, [0,2]] = w - box[:, [2,0]]
		box[:, 0:2][box[:, 0:2]<0] = 0
		box[:, 2][box[:, 2]>w] = w
		box[:, 3][box[:, 3]>h] = h
		box_w = box[:, 2] - box[:, 0]
		box_h = box[:, 3] - box[:, 1]
		box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
		if len(box)>max_boxes: box = box[:max_boxes]
		box_data[:len(box)] = box

	images_data = images_data[0] if len(images_data) == 1 else np.stack(images_data)
	return np.stack(images_data), box_data


import cv2

def get_random_data_cv2(annotation_line, input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
	'''random preprocessing for real-time data augmentation'''
	line = annotation_line.split()
#	print(line)

	# numpy array: BGR, 0-255
#	image = cv2.imread(line[0])
	images = [ cv2.imread(i) for i in line[0].split(',') ]
	# height, width, channel
	ih, iw, _ = images[0].shape
	h, w = input_shape
	box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

	if not random:
		# resize image
		scale = min(w/iw, h/ih)
		nw = int(iw*scale)
		nh = int(ih*scale)
		dx = (w-nw)//2
		dy = (h-nh)//2
#		image_data=0
		if proc_img:
			# resize
			new_images, images_data = [None]*len(images), [None]*len(images)
			for i in range(len(images)):
				images[i] = cv2.resize(images[i], (nw, nh), interpolation=cv2.INTER_AREA)
				# convert into PIL Image object
				images[i] = Image.fromarray(images[i][:, :, ::-1])
				new_images[i] = Image.new('RGB', (w,h), (128,128,128))
				new_images[i].paste(images[i], (dx, dy))
				# convert into numpy array: RGB, 0-1
				images_data[i] = np.array(new_images[i])/255.

		# correct boxes
		box_data = np.zeros((max_boxes,5))
		if len(box)>0:
			np.random.shuffle(box)
			if len(box)>max_boxes: box = box[:max_boxes]
			box[:, [0,2]] = box[:, [0,2]]*scale + dx
			box[:, [1,3]] = box[:, [1,3]]*scale + dy
			box_data[:len(box)] = box

		images_data = images_data[0] if len(images_data) == 1 else np.stack(images_data)
		return images_data, box_data

	# resize image
	new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
	scale = rand(.25, 2)
	if new_ar < 1:
		nh = int(scale*h)
		nw = int(nh*new_ar)
	else:
		nw = int(scale*w)
		nh = int(nw/new_ar)

	# resize
	images = [ cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA) for image in images ]
	# convert into PIL Image object
	images = [ Image.fromarray(image[:, :, ::-1]) for image in images ]

	# place image
	dx = int(rand(0, w-nw))
	dy = int(rand(0, h-nh))
	new_images = [ Image.new('RGB', (w,h), (128,128,128)) for i in range(len(images)) ]
#	new_image.paste(image, (dx, dy))
	for i in range(len(images)): new_images[i].paste(images[i], (dx, dy))
	# convert into numpy array: BGR, 0-255
	images = [ np.asarray(new_image)[:, :, ::-1] for new_image in new_images ]

	# horizontal flip (faster than cv2.flip())
	h_flip = rand() < 0.5
	if h_flip:
		images = [ image[:, ::-1] for image in images ]

#	# vertical flip
#	v_flip = rand() < 0.5
#	if v_flip:
#		image = image[::-1]

#	# rotation augment
#	is_rot = False
#	if is_rot:
#		right = rand() < 0.5
#		if right:
#			image = image.transpose(1, 0, 2)[:, ::-1]
#		else:
#			image = image.transpose(1, 0, 2)[::-1]

	# distort image
	hue = rand(-hue, hue) * 179
	sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
	val = rand(1, val) if rand()<.5 else 1/rand(1, val)

	images_data = []
	for i in range(len(images)):
		img_hsv = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)
		H = img_hsv[:, :, 0].astype(np.float32)
		S = img_hsv[:, :, 1].astype(np.float32)
		V = img_hsv[:, :, 2].astype(np.float32)
	
		H += hue
		np.clip(H, a_min=0, a_max=179, out=H)
	
		S *= sat
		np.clip(S, a_min=0, a_max=255, out=S)
	
		V *= val
		np.clip(V, a_min=0, a_max=255, out=V)
	
		img_hsv[:, :, 0] = H.astype(np.uint8)
		img_hsv[:, :, 1] = S.astype(np.uint8)
		img_hsv[:, :, 2] = V.astype(np.uint8)

		# convert into numpy array: RGB, 0-1
		images_data.append(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB) / 255.0)

	# correct boxes
	box_data = np.zeros((max_boxes,5))
	if len(box)>0:
		np.random.shuffle(box)
		box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
		box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
		if h_flip:
			box[:, [0,2]] = w - box[:, [2,0]]
#		if v_flip:
#			box[:, [1,3]] = h - box[:, [3,1]]
#		if is_rot:
#			if right:
#				tmp = box[:, [0, 2]]
#				box[:, [0,2]] = h - box[:, [3,1]]
#				box[:, [1,3]] = tmp
#			else:
#				tmp = box[:, [2, 0]]
#				box[:, [0,2]] = box[:, [1,3]]
#				box[:, [1,3]] = w - tmp

		box[:, 0:2][box[:, 0:2]<0] = 0
		box[:, 2][box[:, 2]>w] = w
		box[:, 3][box[:, 3]>h] = h
		box_w = box[:, 2] - box[:, 0]
		box_h = box[:, 3] - box[:, 1]
		box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
		if len(box)>max_boxes: box = box[:max_boxes]
		box_data[:len(box)] = box

	images_data = images_data[0] if len(images_data) == 1 else np.stack(images_data)
	return images_data, box_data



def data_generator_custom(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
	'''data generator for fit_generator'''
	n = len(annotation_lines)
	i = 0
	while True:
		image_data = []
		box_data = []
		for b in range(batch_size):
			if i==0:
				np.random.shuffle(annotation_lines)
#			images, box = get_random_data_cv2_v2(annotation_lines[i], input_shape, random=random)
			images, box = get_random_data_cv2(annotation_lines[i], input_shape, random=random)
#			images, box = get_random_data_custom(annotation_lines[i], input_shape, random=random)
#			images, box = f(annotation_lines[i], input_shape, random=random)
			image_data.append(images)
			box_data.append(box)
			i = (i+1) % n
		image_data = np.array(image_data)
		box_data = np.array(box_data)
		y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
		yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper_custom(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
	n = len(annotation_lines)
	if n==0 or batch_size<=0: return None
	return data_generator_custom(annotation_lines, batch_size, input_shape, anchors, num_classes, random)


# %%
if False:
	# %%

#	path_dataset = '/home/asabater/projects/ADL_dataset/'
##	annotations_r = './dataset_scripts/adl/annotations_adl_val_r_fd10_fsn1.txt'
#	annotations_r = './dataset_scripts/adl/annotations_adl_val.txt'
	path_dataset = ''
#	annotations_r = './dataset_scripts/adl/annotations_adl_val_r_fd10_fsn1.txt'
	annotations_r = './dataset_scripts/kitchen/annotations_kitchen_val_v2_25.txt'
	
	with open(annotations_r, 'r') as f: annotation_lines = f.read().splitlines()
	annotation_lines = [ ','.join([ path_dataset+img for img in ann.split(' ')[0].split(',') ]) \
					+ ' ' + ' '.join(ann.split(' ')[1:])
					for ann in annotation_lines ]
	
	path_anchors = 'base_models/yolo_anchors.txt'
	anchors = ktrain.get_anchors(path_anchors)
	input_shape = [416,416]
	num_classes = 25
	
	generator = data_generator_wrapper_custom(annotation_lines[2000:], 32, input_shape, 
										   anchors, num_classes, random=False)
	
	for [img_arr, out_1, out_2, out_3], zero in tqdm(generator, total=len(annotation_lines)):
		continue
		if len(img_arr.shape) == 5:
			for i in range(img_arr.shape[1]):
				plt.imshow(img_arr[0,i,::], interpolation='nearest')
				plt.show()
		else:
			plt.imshow(img_arr[0,::], interpolation='nearest')
			plt.show()
		
	#	time.sleep(0.2)
		
		break
	
	
	# %%
	
	images, box = get_random_data_old(annotation_lines[100], input_shape, random=True)
#	images, box = get_random_data_custom(annotation_lines[500], input_shape, random=True)
	
	if type(images) is not list: images = [images]
	for img in images:
		plt.imshow(img, interpolation='nearest')
		plt.show()


# %%

	annotations = './dataset_scripts/kitchen/annotations_kitchen_train_v2_25.txt'
	with open(annotations, 'r') as f: annotation_lines = f.read().splitlines()
	annotation_lines = [ ','.join([ path_dataset+img for img in ann.split(' ')[0].split(',') ]) \
					+ ' ' + ' '.join(ann.split(' ')[1:])
					for ann in annotation_lines ]	
	boxes_train = []
	for ann in tqdm(annotation_lines, total=len(annotation_lines)):
		images, box = get_random_data_cv2(ann, input_shape, random=False, proc_img=True)
		boxes_train.append(box)
	
	
	# %%
	
	import time 
	from tqdm import tqdm
	
	functs = [ get_random_data_custom, get_random_data_cv2, get_random_data_cv2_v2 ]
	times = [None] * len(functs)
	num_samples = 2500
	batch_size = 32
	
	for i in range(len(functs)):
		times[i] = time.time()
		print(functs[i])
		generator = data_generator_wrapper_custom(functs[i], annotation_lines[:num_samples], batch_size, 
							  input_shape, anchors, num_classes, True)
		count = 0
		total = num_samples//batch_size
		for _ in tqdm(generator, total=total): 
#			_ = generator.next()
			count += 1
			if count >= total: break
			pass
#		while data_generator_custom(f, annotation_lines[:num_samples], 32, 
#							  input_shape, anchors, num_classes, True): continue
#		for l in tqdm(annotation_lines[:num_samples], total=num_samples):
#			_ = functs[i](l, input_shape, random=True)
		times[i] = time.time() - times[i]
	print('\n')
	for i in zip(times, functs): print(i)
	
	
	
	
	
	
