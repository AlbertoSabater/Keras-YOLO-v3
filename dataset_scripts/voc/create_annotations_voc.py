#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:09:26 2019

@author: asabater
"""

import os
from tqdm import tqdm
import xml.etree.ElementTree
import random


classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
           "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# %%

#annotations_train = []
#annotations_val = []
annotations = []

for voc_set in  ['2007', '2012']:
    
#    with open('/mnt/hdd/datasets/VOC/{}_train.txt'.format(voc_set)) as f: 
#        frames_train = [ '/'.join(l.split('/')[-4:])  for l in f.read().splitlines() ]
#    with open('/mnt/hdd/datasets/VOC/{}_val.txt'.format(voc_set)) as f:
#        frames_val = [ '/'.join(l.split('/')[-4:])  for l in f.read().splitlines() ]

    base_path = 'VOCdevkit/VOC{}/JPEGImages/'.format(voc_set)
    annotations_path = '/mnt/hdd/datasets/VOC/VOCdevkit/VOC{}/Annotations/'.format(voc_set)
    frames = [ annotations_path + f for f in os.listdir(annotations_path) ]

    for fr in tqdm(frames, total=len(frames)):
        root = xml.etree.ElementTree.parse(fr).getroot()
        
        fr_name = base_path + root.find('filename').text
        
        objs = root.findall('object')
        if len(objs) == 0: 
            print('len==0')
            continue
        
        boxes = []
        for obj in objs:
            obj_name = obj.find('name').text
            bbx = obj.find('bndbox')
            xmin = int(float(bbx.find('xmin').text))
            ymin = int(float(bbx.find('ymin').text))
            xmax = int(float(bbx.find('xmax').text))
            ymax = int(float(bbx.find('ymax').text))
            boxes.append('{},{},{},{},{}'.format(xmin, ymin, xmax, ymax, classes.index(obj_name)))    
            
#        if fr_name in frames_train:
#            annotations_train.append(fr_name + ' ' + ' '.join(boxes))
#        elif fr_name in frames_val:
#            annotations_train.append(fr_name + ' ' + ' '.join(boxes))
#        else:
#            raise ValueError(fr_name)

        annotations.append(fr_name + ' ' + ' '.join(boxes))


# %%
        
random.shuffle(annotations)
val_perc = 0.2
annotations_train = annotations[int(len(annotations)*val_perc):]
annotations_val = annotations[:int(len(annotations)*val_perc)]


with open('annotations_voc_train.txt', 'w') as f:
    for l in annotations_train:
        f.write(l + '\n')
        
with open('annotations_voc_val.txt', 'w') as f:
    for l in annotations_val:
        f.write(l + '\n')
        

with open('voc_classes.txt', 'w') as f:
    for l in classes:
        f.write(l + '\n')


