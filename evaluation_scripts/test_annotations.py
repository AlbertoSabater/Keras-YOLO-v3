#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:03:13 2019

@author: asabater
"""

import random
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import colorsys

import os


def print_annotations(sample, classes, colors, perc=1):
    sample = [ s for s in sample.split(' ') if s != '' ]
        
#        '/media/asabater/hdd/datasets/imagenet_vid/ILSVRC2015/Data/VID/val/' + 
    image = cv2.imread(sample[0])
    image = cv2.resize(image,(int(image.shape[1]*perc),int(image.shape[0]*perc)))
    image = Image.fromarray(image)
#    image = Image.open(sample[0])
    
    draw = ImageDraw.Draw(image)
    thickness = (image.size[0] + image.size[1]) // 600

    font = ImageFont.truetype(font='keras_yolo3/font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 3.5).astype('int32'))
    
    if len(sample) > 1:
        for box in sample[1:]:
            for i in range(thickness):

                left, bottom, right, top, c = box.split(',')
                left, bottom, right, top, c = int(left)*perc, int(bottom)*perc, int(right)*perc, int(top)*perc, int(c)
                color = colors[c]
#                    print(left, bottom, right, top, c)
                
                draw.rectangle(
                        [int(left) + i, int(top) + i, int(right) - i, int(bottom) - i],
                        outline = color)
                
                c = classes[c]
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

    cv2.putText(result, text=sample[0].split('/')[-2], org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        
    return result


def get_random_init(samples):
    return samples[random.randint(0, len(samples)):]


def get_samples_containing_classes(samples, filter_classes):
    return [ s for s in samples if any([ c in [ int(sc.split(',') [-1]) for sc in s.split()[1:] ] for c in filter_classes ]) ]


        
# %%


def main():
	
	os.chdir(os.getcwd() + '/..')

	perc = 0.5
	
	#annotations_file = '/media/asabater/hdd/datasets/imagenet_vid/annotations_train.txt'
	#classes = '/media/asabater/hdd/datasets/imagenet_vid/imagenet_vid_classes.txt'
	
	path_dataset = '/home/asabater/projects/ADL_dataset/'
	suffix = ''
	path_annotations = ['./dataset_scripts/adl/annotations_adl_train{}.txt'.format(suffix),
	                    './dataset_scripts/adl/annotations_adl_val{}.txt'.format(suffix)]
	classes = './dataset_scripts/adl/adl_classes.txt'
	
	#path_annotations = ['./dataset_scripts/adl/annotations_adl_train_608_v2_27.txt',
	#                    './dataset_scripts/adl/annotations_adl_val_608_v2_27.txt']
	#classes = './dataset_scripts/adl/adl_classes_v2_27.txt'
	
	#path_dataset = '/home/asabater/projects/ADL_dataset/'
	#path_annotations = ['./dataset_scripts/adl/annotations_adl_train_v2_27_pr416.txt',
	#                    './dataset_scripts/adl/annotations_adl_val_v2_27_pr416.txt']
	#classes = './dataset_scripts/adl/adl_classes_v2_27.txt'
	
	
#	path_dataset = '' 
#	version = '_cv1_17' 		# v1_15
#	path_annotations = ['./dataset_scripts/kitchen/annotations_kitchen_train{}.txt'.format(version),
#	                    './dataset_scripts/kitchen/annotations_kitchen_val{}.txt'.format(version)]
#	classes = './dataset_scripts/kitchen/kitchen_classes{}.txt'.format(version)
	
	
	#path_dataset = '/mnt/hdd/datasets/VOC/'
	#path_annotations = ['./dataset_scripts/voc/annotations_voc_train.txt',
	#                    './dataset_scripts/voc/annotations_voc_val.txt']
	#classes = './dataset_scripts/voc/voc_classes.txt'
	
	#path_dataset = '/mnt/hdd/datasets/coco/'
	#path_annotations = ['./dataset_scripts/coco/annotations_coco_train.txt',
	#                    './dataset_scripts/coco/annotations_coco_val.txt']
	#classes = './dataset_scripts/coco/coco_classes.txt'
	#path_annotations = ['./dataset_scripts/coco/annotations_coco_train_super.txt',
	#                    './dataset_scripts/coco/annotations_coco_val_super.txt']
	#classes = './dataset_scripts/coco/coco_classes_super.txt'
	
	
	#annotations_file = '/home/asabater/projects/epic_dataset/annotations_epic_train.txt'
	#classes = '/home/asabater/projects/epic_dataset/epic_classes.txt'
	#perc = 0.8; wait_time=400
	
	with open(classes, 'r') as f:
	    classes = f.read().splitlines()
	    
	
	hsv_tuples = [(x / len(classes), 1., 1.)
	              for x in range(len(classes))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
	    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
	        colors))
	np.random.seed(10101)  # Fixed seed for consistent colors across runs.
	np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	np.random.seed(None)  # Reset seed to default.
	       
	
	with open(path_annotations[0]) as f: lines_train = [ path_dataset + l for l in f.readlines() ]
	with open(path_annotations[1]) as f: lines_val = [ path_dataset + l for l in f.readlines() ]
	annotations = lines_train + lines_val

	wait_time = 300
	
#	samples = annotations
	samples = get_random_init(annotations)
#	samples = get_samples_containing_classes(annotations, 
#				 [classes.index(c) for c in 'container'.split() ])
	
	print('samples: {} | {} | {}'.format(len(samples), 
		  len(set([ s.split('/')[-3] for s in samples])),
		  len(set([ '/'.join(s.split('/')[-3:-1]) for s in samples])),
		  ))
#	for sample in samples[0::12]:
	for sample in samples:
	    print(sample.split()[0])
	    
	#    print(sample)
	    result = print_annotations(sample, classes, colors, perc)
	    cv2.imshow("result", result)
	
	    if cv2.waitKey() & 0xFF == ord('q'):
	#        cv2.destroyAllWindows()
	        break
	#cv2.destroyAllWindows()


		

if __name__ == "__main__": main()

'''
trash_can
P_01/00009798

basket
P_09/00022167
P_13/00003480
P_13/00044190
P_15/00010338
P_18/00020946

P_02/00018000
P_02/00026970
P_02/00046110
P_02/00060750

large_container
P_09/00034368
P_19/00031224

bottle
P_02/00045120

container
P_03/00001260
P_07/00032700
'''

'''
look_change
P_01/00000228
P_01/00000288

occlusion
P_13/00002520
P_13/00002550
P_13/00002580

P_03/00017250
P_03/00017370
P_03/00017520

blurr
P_05/00029364
'''

frames = {
			'look_change': ['P_01/00000228', 'P_01/00000288'],
			'occlusion': ['P_13/00002520', 'P_13/00002550', 'P_13/00002580', 
				 'P_03/00017250', 'P_03/00017370', 'P_03/00017520'],
			'blurr': ['P_05/00029364'],
		}


# %%

if False:
	# %%

	perc = 0.5
	
	path_dataset = '/home/asabater/projects/ADL_dataset/'
	suffix = ''
	path_annotations = ['./dataset_scripts/adl/annotations_adl_train{}.txt'.format(suffix),
	                    './dataset_scripts/adl/annotations_adl_val{}.txt'.format(suffix)]
	classes = './dataset_scripts/adl/adl_classes.txt'
	
	with open(classes, 'r') as f:
	    classes = f.read().splitlines()
	    
	
	hsv_tuples = [(x / len(classes), 1., 1.)
	              for x in range(len(classes))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
	    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
	        colors))
	np.random.seed(10101)  # Fixed seed for consistent colors across runs.
	np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	np.random.seed(None)  # Reset seed to default.
	       
	
	with open(path_annotations[0]) as f: lines_train = [ path_dataset + l for l in f.readlines() ]
	with open(path_annotations[1]) as f: lines_val = [ path_dataset + l for l in f.readlines() ]
	annotations = lines_train + lines_val

	wait_time = 300
	
#	samples = annotations
#	samples = get_random_init(annotations)
#	samples = get_samples_containing_classes(annotations, 
#				 [classes.index(c) for c in 'container'.split() ])
	
#	print('samples: {} | {} | {}'.format(len(samples), 
#		  len(set([ s.split('/')[-3] for s in samples])),
#		  len(set([ '/'.join(s.split('/')[-3:-1]) for s in samples])),
#		  ))

	
#	frames, folder = {
#				'trash_can': ['P_01/00009798'],
#				'basket': ['P_09/00022167', 'P_13/00003480', 'P_13/00044190', 'P_15/00010338', 'P_18/00020946', 'P_02/00018000', 'P_02/00026970', 'P_02/00046110', 'P_02/00060750'],
#				'large_container': ['P_09/00034368', 'P_19/00031224'],
#				'bottle': ['P_02/00045120'],
#				'container': ['P_03/00001260', 'P_07/00032700']
#			}, 'inconsistencies'
	
	frames, folder = {
			'look_change': ['P_01/00000228', 'P_01/00000288'],
			'occlusion': ['P_13/00002520', 'P_13/00002550', 'P_13/00002580', 
				 'P_03/00017250', 'P_03/00017370', 'P_03/00017520'],
			'blurr': ['P_05/00029364'],
		}, 'intro'


	path_results = '/mnt/hdd/egocentric_results/figuras_05.2019/'

	for cat, l in frames.items():
		for image_id in l:
			
			frame = [ ann for ann in annotations if image_id in ann ][0]
			result = print_annotations(frame, classes, colors, perc)
#			cv2.imwrite('{}{}/{}_{}.png'.format(path_results, folder, cat, image_id.replace('/', '_')), result)
			cv2.imwrite('{}{}/{}_{}.png'.format(path_results, folder, cat, image_id.replace('/', '_')), result)


# %%

if False:
	# %%
	
# =============================================================================
# 	Check boundaries
# =============================================================================
	
	from tqdm import tqdm
	import sys
	import numpy as np
	from PIL import Image, ImageFont, ImageDraw

	
	version = 'v0_-1'
	path_dataset = ''
	path_annotations = './dataset_scripts/kitchen/annotations_kitchen_val_{}.txt'.format(version)
	with open(path_annotations) as f: annotations = [ '' + l for l in f.readlines() ]

	
	widths, heights, image_sizes = [], [], []
	
	def downscale(coord, img_size):
		return int(coord*416/img_size)
		
	
	for ann in tqdm(annotations, total=len(annotations), file=sys.stdout):
		
		ann = ann.split()
		img = Image.open(path_dataset + ann[0])
		img_size = img.size
		image_sizes.append(img_size)
		bbs = [ [ int(b) for b in bb.split(',') ] for bb in ann[1:] ]
		if len(bbs) == 0: print('len == 0')
		
		for bb in bbs:
			
			width = bb[2] - bb[0]
			height = bb[3] - bb[1]
			widths.append(width); heights.append(height)
			
			if width <= 0: print('width <= 0 | ', '/'.join(ann[0].split('/')[-3:]), bb, width, img_size)
			if height <= 0: print('height <= 0 | ', '/'.join(ann[0].split('/')[-3:]), bb, height, img_size)
			
			if bb[0] < 0: print('x_min < 0')
			if bb[1] < 0: print('y_min < 0')
			if bb[2] > img_size[0]: print('x_max < 0', '|', '/'.join(ann[0].split('/')[-3:]), bb, bb[2], img_size[0])
			if bb[3] > img_size[1]: print('y_max < 0', '|', '/'.join(ann[0].split('/')[-3:]), bb, bb[3], img_size[1])

#			if bb[0] == bb[2]: print('bb[0] == bb[2] | ', '/'.join(ann[0].split('/')[-3:]), bb)
#			if bb[1] == bb[3]: print('bb[1] == bb[3] | ', '/'.join(ann[0].split('/')[-3:]), bb)
			
			if downscale(bb[0], img_size[0]) == downscale(bb[2], img_size[0]): print('bb[0] == bb[2] | ', '/'.join(ann[0].split('/')[-3:]), bb)
			if downscale(bb[1], img_size[1]) == downscale(bb[3], img_size[1]): print('bb[0] == bb[2] | ', '/'.join(ann[0].split('/')[-3:]), bb)

	print(min(widths), max(widths))
	print(min(heights), max(heights))
	print(min([ w for w,h in image_sizes]), max([ w for w,h in image_sizes]))
	print(min([ h for w,h in image_sizes]), max([ h for w,h in image_sizes]))
	
