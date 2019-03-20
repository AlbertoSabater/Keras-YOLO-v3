#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:53:31 2019

@author: asabater
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";  

import sys
sys.path.append('keras_yolo3/')
sys.path.append('keras_yolo3/yolo3/')

import json
import os
import numpy as np
import keras_yolo3.train as ktrain
#from keras.models import model_from_json
import train_utils

from tensorboard.backend.event_processing import event_accumulator
import datetime
import time

# TODO: imprimir resumen de la evaluación por pantalla/portapapeles para pegarlo en excel
# TODO: sacar métricas con diferentes IoU thresholds


path_results = '/mnt/hdd/egocentric_results/'
dataset_name = 'adl'
model_num = 13
score = 0.15
iou = 0.5


model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
train_params = json.load(open(model_folder + 'train_params.json', 'r'))

class_names = ktrain.get_classes(train_params['path_classes'])
num_classes = len(class_names)
anchors = ktrain.get_anchors(train_params['path_anchors'])

#model_weights = model_folder + [ 'weights/' + f for f in os.listdir(model_folder + '/weights/') if f.endswith('.h5') and not f.startswith('trained_weights') ][0]

path_dataset = train_params['path_dataset']
annotations_file = train_params['path_annotations'][1]


# %%

tb_files = [ model_folder + f for f in os.listdir(model_folder) if f.startswith('events.out.tfevents') ]


train_losses, val_losses = [], []
times = []
for tbf in tb_files:
    ea = event_accumulator.EventAccumulator(tbf).Reload()
    train_losses += [ e.value for e in ea.Scalars('loss') ]
    val_losses += [ e.value for e in ea.Scalars('val_loss') ]
    times += [ e.wall_time for e in ea.Scalars('val_loss') ]


num_epochs = len(train_losses)
val_loss = min(val_losses)
train_loss = train_losses[val_losses.index(val_loss) ]

train_init, train_end = min(times), max(times)

train_init = datetime.datetime.fromtimestamp(train_init)
train_end = datetime.datetime.fromtimestamp(train_end)

train_diff = (train_end - train_init)
train_diff = '{}d {:05.2f}h'.format(train_diff.days, train_diff.seconds/3600)


# %%

model_weights = train_utils.get_best_weights(model_folder)


# %%

from eyolo import EYOLO
#import tensorflow as tf
#from keras import backend as K

#model = None
#with K.device('/cpu:0'):

model = EYOLO(
                model_image_size = tuple(train_params['input_shape']),
#                model_path = 'base_models/yolo.h5',
#                classes_path = 'base_models/coco_classes.txt',
#                anchors_path = 'base_models/yolo_anchors.txt',
                model_path = model_weights,
                classes_path = train_params['path_classes'],
                anchors_path = train_params['path_anchors'],
                score = score,    # 0.3
                iou = iou,      # 0.5
            )

print(model)


# %%

import cv2
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm


# Create pred file

#path_dataset = '/home/asabater/projects/ADL_dataset/ADL_frames_416/'
#annotations_file = '/home/asabater/projects/ADL_dataset/annotations_adl_val.txt'

with open(annotations_file, 'r') as f: annotations = f.read().splitlines()
preds = []

t = time.time()
total = len(annotations)
for ann in tqdm(annotations[:total], total=total):
    img = ann.split()[0]
    
    image = cv2.imread(path_dataset + img)
    image = Image.fromarray(image)
    boxes, scores, classes = model.get_prediction(image)
    
    for i in range(len(boxes)):
        preds.append({
                    'image_id': img[:-4],
                    'category_id': int(classes[i]),
                    'bbox': [ int(b) for b in boxes[i].tolist() ],
    #                'score': 1,
                    'score': float(scores[i]),
                })
	
fps = len(annotations)/(time.time()-t)
preds_filename = '{}preds_{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4])
json.dump(preds, open(preds_filename, 'w'))

#    break
    
    
# %%

#import random
#import cv2
#from PIL import Image, ImageFont, ImageDraw
#
##base_imgs = '/home/asabater/projects/ADL_dataset/ADL_frames_416/'
#base_imgs = '/mnt/hdd/datasets/imagenet_vid/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000/'
#base_imgs += random.choice(os.listdir(base_imgs)) + '/'
#img = base_imgs + random.choice(os.listdir(base_imgs))
#print(img)
#img = cv2.imread(img)
#img = Image.fromarray(img)
#boxes, scores, classes = model.get_prediction(img)


# %%

#preds_filename = '{}preds_{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4])
#preds = json.load(open(preds_filename, 'r+'))




# %%

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ann = annotations_file[:-4] + '_coco.json'

eval_stats = {}

cocoGt = COCO(ann)
cocoDt = cocoGt.loadRes(preds_filename)

cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

cocoEval.params.catIds = [1]
    
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

eval_stats['total'] = cocoEval.stats.tolist()

for c in range(0, len(class_names)+1):
    cocoEval.params.catIds = [1]
    cocoEval.evaluate()
    cocoEval.accumulate()
#    cocoEval.summarize()
    eval_stats['cat_{}'.format(c)] = cocoEval.stats.tolist()


eval_filename = '{}stats_{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4])
json.dump(eval_stats, open(eval_filename, 'w'))


# %%

import pyperclip

result = '{model_folder}\t{input_shape}\t{annotations}\t{anchors}\t{pretraining}\t{frozen_training:.2f}\t{training_time}'.format(
            model_folder = '/'.join(model_folder.split('/')[-2:]), 
            input_shape = train_params['input_shape'],
            annotations = train_params['path_annotations'],
            anchors = train_params['path_anchors'],
            pretraining = train_params['path_weights'],
            frozen_training = train_params['freeze_body'],
            training_time = train_diff
        )

result += '\t{train_loss:.5f}\t{val_loss:.5f}'.format(train_loss=train_loss, val_loss=val_loss).replace('.', ',')
result += '\t{score:.5f}\t{iou:.2f}'.format(score=score, iou=iou).replace('.', ',')

result += '\t{mAP}\t{mAP50}\t{mAP75}\t{mAPS}\t{mAPM}\t{mAPL}'.format(
            mAP=eval_stats['total'][0], mAP50=eval_stats['total'][1], mAP75=eval_stats['total'][2], 
            mAPS=eval_stats['total'][3], mAPM=eval_stats['total'][4], mAPL=eval_stats['total'][5], 
        ).replace('.', ',')

result += '\t{fps:.2f}'.format(fps=fps)

print(result)
pyperclip.copy(result)
