#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 13:27:15 2019

@author: asabater
"""

import sys
sys.path.append('keras_yolo3/')

import keras_yolo3.train as ktrain
import train_utils




path_dataset = '/home/asabater/projects/ADL_dataset/'
annotations_file = './dataset_scripts/adl/annotations_adl_val_416.txt'
with open(annotations_file) as f: lines_val = [ path_dataset + l for l in f.readlines() ]

path_anchors = 'base_models/yolo_anchors.txt'
anchors = ktrain.get_anchors(path_anchors)

path_classes = './dataset_scripts/adl/adl_classes.txt'
class_names = ktrain.get_classes(path_classes)
num_classes = len(class_names)

batch_size = 1
input_shape = (416, 416)


# %%

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import time


#lines = lines_val[3:3+batch_size]
lines = lines_val
generator = train_utils.data_generator_wrapper(lines, batch_size, input_shape, anchors, num_classes, random=True)

for [img, out_1, out_2, out_3], zero in tqdm(generator, total=len(lines)):

    img = np.asarray(img[0])

    plt.imshow(img, interpolation='nearest')
    plt.show()
    
#    time.sleep(0.2)
    
    break
    

