#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:40:46 2019

@author: asabater
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  

import train
import time


# %%

version = '_v2_27'

train_params = {
			'path_results': '/mnt/hdd/egocentric_results/',
			'dataset_name': 'adl',
			'path_dataset': '/home/asabater/projects/ADL_dataset/',
			'path_classes': './dataset_scripts/adl/adl_classes{}.txt'.format(version),
			'path_anchors': 'base_models/yolo_anchors.txt',
			'path_annotations': ['./dataset_scripts/adl/annotations_adl_train{}.txt'.format(version),
								'./dataset_scripts/adl/annotations_adl_val{}.txt'.format(version)],
			'path_weights': 'base_models/yolov3-spp.h5',
			'input_shape': [416,416],
			'size_suffix': '', 'version': version,
			'mode': None,
			'spp': True,
			'freeze_body': 2,
			'multi_scale': True,
			'frozen_epochs': 15,
			'loss_percs': {'xy': 0.5, 'wh': 0.0625, 'confidence_noobj': 4, 'confidence_obj': 4, 'class': 0.0625},
			}


train.main(train_params)

print('='*70)
print('='*70)
print('='*70)
print('='*70)

time.sleep(15*10)

