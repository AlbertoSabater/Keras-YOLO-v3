#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:04:15 2019

@author: asabater
"""

import os
os.chdir(os.getcwd() + '/..')


# %%

import cv2
import sys
from tqdm import tqdm

path_dataset = '/home/asabater/projects/ADL_dataset/'
path_annotations = ['./dataset_scripts/adl/annotations_adl_train.txt'.format(),
						'./dataset_scripts/adl/annotations_adl_val.txt'.format()]


with open(path_annotations[1]) as f: lines = f.readlines()

bbx_area = []
img_area = []
for l in tqdm(lines, total=len(lines), file=sys.stdout):
	l = l.split()
	
	img = cv2.imread(path_dataset + l[0])
	img_area.append(img.shape[0] * img.shape[1])
	
	for bb in l[1:]:
		bb = [ int(b) for b in bb.split(',') ]
		bbx_area.append((bb[2]-bb[0])*(bb[3]-bb[1]))


print('bbx:', sum(bbx_area))
print('img:', sum(img_area))
print('perc:', sum(bbx_area)/sum(img_area))
print('perc:', sum(img_area)/sum(bbx_area))


# %%

