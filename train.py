#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:03:59 2019

@author: asabater
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  


import sys
sys.path.append('keras_yolo3/')

import keras_yolo3.train as ktrain
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from tensorflow.python.client import device_lib

import train_utils
import numpy as np
import json

from data_factory import data_generator_wrapper_default


# TODO: train/val/test split en función del dataset

# tensorboard --logdir /media/asabater/hdd/egocentric

path_results = '/mnt/hdd/egocentric_results/'
print_indnt = 12
print_line = 100
num_gpu = len([x for x in device_lib.list_local_devices() if x.device_type == 'GPU'])


dataset_name = 'adl'
path_weights = 'base_models/yolo.h5'
#path_weights = 'base_models/darknet53.h5'
freeze_body = 2                 # freeze_body = 1 -> freeze feature extractor
                                # freeze_body = 2 -> freeze all but 3 output layers
                                # freeze_body = otro -> don't freeze
input_shape = (416,416)         # multiple of 32, hw

#val_split = 0.1
batch_size_frozen = 32          # 32
batch_size_unfrozen = 4         # note that more GPU memory is required after unfreezing the body
frozen_epochs = 15               # 50



title = 'Remove null trainings'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
train_utils.remove_null_trainings(path_results, dataset_name)
print('='*print_line)



path_anchors = 'base_models/yolo_anchors.txt'
path_dataset = ''
version = -1

# Get dataset annotations, classes and anchors
if dataset_name == 'adl':
#    path_dataset = '/mnt/hdd/datasets/adl_dataset/ADL_frames/'
    version = '_v2_27'        # _v2_27
    size_suffix = ''         # '_416'
    input_shape = (416,416)
    path_dataset = '/home/asabater/projects/ADL_dataset/'
    path_annotations = ['./dataset_scripts/adl/annotations_adl_train{}{}.txt'.format(size_suffix, version),
                        './dataset_scripts/adl/annotations_adl_val{}{}.txt'.format(size_suffix, version)]
#    path_annotations = ['/home/asabater/projects/ADL_dataset/annotations_adl_train.txt',
#                        '/home/asabater/projects/ADL_dataset/annotations_adl_val.txt']
    path_classes = './dataset_scripts/adl/adl_classes{}.txt'.format(version)
#    path_anchors = './dataset_scripts/adl/anchors_adl{}{}.txt'.format(size_suffix, version)

elif dataset_name == 'coco':    
    # 117266 train images
    # 4952 val images
    # 80/12 categories
    
    version = '_super'
    input_shape = (416,416)
    path_dataset = '/mnt/hdd/datasets/coco/'
    
    
    path_annotations = ['./dataset_scripts/coco/annotations_coco_train{}.txt'.format(version),
                    './dataset_scripts/coco/annotations_coco_val{}.txt'.format(version)]
    path_classes = './dataset_scripts/coco/coco_classes{}.txt'.format(version)
    
elif dataset_name == 'voc':
#    path_dataset = '/mnt/hdd/datasets/adl_dataset/ADL_frames/'
    input_shape = (416,416)
    version = ''
    size_suffix = ''
    path_dataset = '/mnt/hdd/datasets/VOC/'
    path_annotations = ['./dataset_scripts/voc/annotations_voc_train.txt',
                        './dataset_scripts/voc/annotations_voc_val.txt']
#    path_annotations = ['/home/asabater/projects/ADL_dataset/annotations_adl_train.txt',
#                        '/home/asabater/projects/ADL_dataset/annotations_adl_val.txt']
    path_classes = './dataset_scripts/voc/voc_classes.txt'
    path_anchors = './dataset_scripts/voc/anchors_voc{}{}.txt'.format(size_suffix, version)

elif dataset_name == 'epic':
    path_annotations = '/home/asabater/projects/epic_dataset/annotations_epic_train.txt'
    path_classes = '/home/asabater/projects/epic_dataset/epic_classes.txt'
    
elif dataset_name == 'imagenet':
    path_annotations = '/media/asabater/hdd/datasets/imagenet_vid/annotations_train.txt'
    path_classes = '/media/asabater/hdd/datasets/imagenet_vid/imagenet_vid_classes.txt'
    
else: raise ValueError('Dataset not recognized')

# Load dataset classes and anchors
class_names = ktrain.get_classes(path_classes)
num_classes = len(class_names)
anchors = ktrain.get_anchors(path_anchors)



# Train/Val split
np.random.seed(10101)
if type(path_annotations) == list:
    with open(path_annotations[0]) as f: lines_train = [ path_dataset + l for l in f.readlines() ]
    with open(path_annotations[1]) as f: lines_val = [ path_dataset + l for l in f.readlines() ]
    num_train, num_val = len(lines_train), len(lines_val)
else:
    with open(path_annotations) as f: lines = [ path_dataset + l for l in f.readlines() ]
    np.random.shuffle(lines)
    num_val, num_train = int(len(lines)*val_split); len(lines) - num_val
    lines_train = lines[:num_train]; lines_val = lines[num_train:]
np.random.shuffle(lines_train), np.random.shuffle(lines_val)
np.random.seed(None)


title = 'Create and get model folders'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
# Create and get model folders
path_model = train_utils.create_model_folder(path_results, dataset_name)
print(path_model)
print('='*print_line)


# %%

# =============================================================================
# Create model
# =============================================================================

title = 'Create Keras model'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
print('Num. GPUs:', num_gpu)
model = ktrain.create_model(input_shape, anchors, num_classes,
            freeze_body = freeze_body, 
            weights_path = path_weights) # make sure you know what you freeze
print('='*print_line)



# Train callbacks
logging = TensorBoard(log_dir = path_model)
checkpoint = ModelCheckpoint(path_model + 'weights/' + 'ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5',
                             monitor='val_loss', 
                             save_weights_only=True, 
                             save_best_only=True, 
                             period=1)
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0, factor=0.1, patience=3, verbose=1)
reduce_lr_1 = ReduceLROnPlateau(monitor='loss', min_delta=0.75, factor=0.1, patience=3, verbose=1)
reduce_lr_2 = ReduceLROnPlateau(monitor='val_loss', min_delta=0, factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    


title = 'Storing train params and model architecture'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
train_params = {
                'dataset_name': dataset_name,
                'freeze_body': freeze_body,
                'input_shape': input_shape,
#                'val_split': val_split,
                'batch_size_frozen': batch_size_frozen,
                'batch_size_unfrozen': batch_size_unfrozen,
                'frozen_epochs': frozen_epochs,
                'path_anchors': path_anchors,
                'path_annotations': path_annotations,
                'path_classes': path_classes,
                'path_dataset': path_dataset,
                'path_weights': path_weights,
                'num_val': num_val,
                'num_train': num_train,
                'version': version,
                }
print(train_params)
with open(path_model + 'train_params.json', 'w') as f:
    json.dump(train_params, f)
    
model_architecture = model.to_json()
with open(path_model + 'architecture.json', 'w') as f:
    json.dump(model_architecture, f)
print('='*print_line)


# %%

# =============================================================================
# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
# =============================================================================

title = 'Train first stage'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
import datetime
print(print(datetime.datetime.now()))
if True:
    optimizer = Adam(lr=1e-3)
    model.compile(optimizer = optimizer,
                  loss = {'yolo_loss': lambda y_true, y_pred: y_pred},        # use custom yolo_loss Lambda layer.
                  metrics = [train_utils.get_lr_metric(optimizer)]
                  )

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_frozen))
    hist_1 = model.fit_generator(
#            ktrain.data_generator_wrapper(lines_train, batch_size_frozen, input_shape, anchors, num_classes),
#            train_utils.data_generator_wrapper(lines_train, batch_size_frozen, input_shape, anchors, num_classes, random=True),
#            data_generator_wrapper_factory(train_data_factory),
#            train_generator,
            data_generator_wrapper_default(lines_train, batch_size_frozen, input_shape, anchors, num_classes, random=True),
            steps_per_epoch = max(1, num_train//batch_size_frozen),
#            validation_data = ktrain.data_generator_wrapper(lines_val, batch_size_frozen, input_shape, anchors, num_classes),
#            validation_data = train_utils.data_generator_wrapper(lines_val, batch_size_frozen, input_shape, anchors, num_classes, random=True),
#            validation_data=data_generator_wrapper_factory(valid_data_factory),
#            validation_data = val_generator,
            validation_data = data_generator_wrapper_default(lines_val, batch_size_frozen, input_shape, anchors, num_classes, random=False),
            validation_steps = max(1, num_val//batch_size_frozen),
            epochs = frozen_epochs,
            initial_epoch = 0,
            callbacks=[logging, 
#                       checkpoint, 
                       reduce_lr_1
#                       reduce_lr
                        ])
    model.save_weights(path_model + 'weights/trained_weights_stage_1.h5')
print('='*print_line)


# %%

# TODO: TEST
# =============================================================================
# Load best weights from stage 1
# =============================================================================

#best_weights_stage_1 = train_utils.get_best_weights(path_model)
#title = 'Loading weights: ' + best_weights_stage_1; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
#model.load_weights(best_weights_stage_1)
#print('='*print_line)


# %%

# =============================================================================
# Unfreeze and continue training, to fine-tune.
# Train longer if the result is not good.
# =============================================================================

title = 'Train second stage'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
if True:
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    print('Unfreeze all of the layers.')

    optimizer = Adam(lr=1e-4)
    model.compile(optimizer = optimizer,
                  loss = {'yolo_loss': lambda y_true, y_pred: y_pred},        # use custom yolo_loss Lambda layer.
                  metrics = [train_utils.get_lr_metric(optimizer)]
                  )


    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_unfrozen))
    hist_2 = model.fit_generator(
#        ktrain.data_generator_wrapper(lines_train, batch_size_unfrozen, input_shape, anchors, num_classes),
#        train_utils.data_generator_wrapper(lines_train, batch_size_unfrozen, input_shape, anchors, num_classes, random=True),
        data_generator_wrapper_default(lines_train, batch_size_unfrozen, input_shape, anchors, num_classes, random=True),
        steps_per_epoch = max(1, num_train//batch_size_unfrozen),
#        validation_data = ktrain.data_generator_wrapper(lines_val, batch_size_unfrozen, input_shape, anchors, num_classes),
#        validation_data = train_utils.data_generator_wrapper(lines_val, batch_size_unfrozen, input_shape, anchors, num_classes, random=True),
        validation_data = data_generator_wrapper_default(lines_val, batch_size_unfrozen, input_shape, anchors, num_classes, random=False),
        validation_steps = max(1, num_val//batch_size_unfrozen),
        epochs = 500,
        initial_epoch = frozen_epochs,
        callbacks = [logging, checkpoint, 
                       reduce_lr_2,
#                       reduce_lr
                        early_stopping])
    model.save_weights(path_model + 'weights/trained_weights_final.h5')
print('='*print_line)
    
    
# %%

train_utils.remove_worst_weights(path_model)



    
    
