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


perc = 1

annotations_file = '/media/asabater/hdd/datasets/imagenet_vid/annotations_train.txt'
#classes = '/media/asabater/hdd/datasets/imagenet_vid/imagenet_vid_classes.txt'

#path_dataset = '/home/asabater/projects/ADL_dataset/'
#suffix = '_608'
#path_annotations = ['./dataset_scripts/adl/annotations_adl_train{}.txt'.format(suffix),
#                    './dataset_scripts/adl/annotations_adl_val{}.txt'.format(suffix)]
#classes = './dataset_scripts/adl/adl_classes.txt'

#path_annotations = ['./dataset_scripts/adl/annotations_adl_train_608_v2_27.txt',
#                    './dataset_scripts/adl/annotations_adl_val_608_v2_27.txt']
#classes = './dataset_scripts/adl/adl_classes_v2_27.txt'

#path_dataset = '/mnt/hdd/datasets/VOC/'
#path_annotations = ['./dataset_scripts/voc/annotations_voc_train.txt',
#                    './dataset_scripts/voc/annotations_voc_val.txt']
#classes = './dataset_scripts/voc/voc_classes.txt'

path_dataset = '/mnt/hdd/datasets/coco/'
path_annotations = ['./dataset_scripts/coco/annotations_coco_train.txt',
                    './dataset_scripts/coco/annotations_coco_val.txt']
classes = './dataset_scripts/coco/coco_classes.txt'
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


# %%

def print_annotations(sample):
    sample = [ s for s in sample.split(' ') if s != '' ]
        
#        '/media/asabater/hdd/datasets/imagenet_vid/ILSVRC2015/Data/VID/val/' + 
    image = cv2.imread(sample[0])
    image = cv2.resize(image,(int(image.shape[1]*perc),int(image.shape[0]*perc)))
    image = Image.fromarray(image)
    
    draw = ImageDraw.Draw(image)
    thickness = (image.size[0] + image.size[1]) // 600

    font = ImageFont.truetype(font='keras_yolo3/font/FiraMono-Medium.otf',
                size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    
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
        
    cv2.imshow("result", result)


def get_random_init(samples):
    return samples[random.randint(0, len(samples)):]


def get_samples_containing_classes(samples, filter_classes):
    return [ s for s in samples if any([ c in [ int(sc.split(',') [-1]) for sc in s.split()[1:] ] for c in filter_classes ]) ]


        
# %%
  
wait_time = 600

samples = get_random_init(annotations)
#samples = get_samples_containing_classes(annotations, [classes.index('cloth')])

for sample in samples[0::10]:
    
    print(sample)
    print_annotations(sample)

    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
        