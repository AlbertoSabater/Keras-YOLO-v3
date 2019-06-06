#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:26:28 2019

@author: asabater
"""

import sys
sys.path.append('keras_yolo3/')

import keras_yolo3.train as ktrain
import pandas as pd


path_dataset = '/home/asabater/projects/ADL_dataset/'
#path_annotations = ['./dataset_scripts/adl/annotations_adl_train_608_v2_27.txt',
#                    './dataset_scripts/adl/annotations_adl_val_608_v2_27.txt']
#path_classes = './dataset_scripts/adl/adl_classes_v2_27.txt'
path_annotations = ['./dataset_scripts/adl/annotations_adl_train.txt',
                    './dataset_scripts/adl/annotations_adl_val.txt']
path_classes = './dataset_scripts/adl/adl_classes.txt'

# Load dataset classes and anchors
class_names = ktrain.get_classes(path_classes)
num_classes = len(class_names)

with open(path_annotations[0]) as f: lines_train = [ path_dataset + l for l in f.readlines() ]
with open(path_annotations[1]) as f: lines_val = [ path_dataset + l for l in f.readlines() ]
annotations = lines_train + lines_val


res = []

for ann in annotations:
    ann = ann.split()
    img = ann[0]
    boxes = ann[1:]

    img = img.split('/')
    video = img[-2]
    frame = img[-1]
	
    added_cats = []
    for box in boxes:
        box = box.split(',')
        cat = int(box[-1])
        cat_name = class_names[cat]
        if cat in added_cats: continue

        res.append({
                    'cat': cat,
                    'cat_name': cat_name,
                    'video': video,
                    'frame': frame
                })
        added_cats.append(cat)

res = pd.DataFrame(res)


# %%

# Sum of each class

from collections import Counter
import matplotlib.pyplot as plt

counter = Counter(res['cat_name'])
counter = sorted(counter.items(), key = lambda x: x[1], reverse=True)

plt.figure(figsize=(20,5))
plt.bar([ k for k,v in counter ], [ v for k,v in counter ]);
plt.xticks(rotation=50);
plt.show()


# %%

# Mean of each class and dispersion -> box plot

grouped = res[['cat_name', 'video']].groupby(['video', 'cat_name']).size().reset_index()
grouped = grouped.groupby('cat_name')

videos = res.video.drop_duplicates().tolist()
occurrences = { col:[ g[g.video==v][0].values[0] if v in g.video.values else 0 for v in videos ]
                    for col,g in grouped }

occurrences = pd.DataFrame(occurrences, index=videos)
meds = occurrences.mean()
meds = meds.sort_values(ascending=False)

occurrences = occurrences[meds.index]
occurrences.boxplot(figsize=(20,7));
plt.xticks(rotation=50);
plt.show()


# %%

import numpy as np

occurrences = occurrences.replace({0: np.nan})


# For each video print each class



# Number of videos where each class apper
# Number of classes that appear in each video










