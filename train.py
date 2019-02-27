#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:03:59 2019

@author: asabater
"""

import sys
sys.path.append('keras_yolo3/')

import keras_yolo3.train as ktrain
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

import train_utils
import numpy as np
import json


# TODO: train/val/test split en función del dataset

# tensorboard --logdir /media/asabater/hdd/egocentric

path_results = '/mnt/hdd/egocentric_results/'
print_indnt = 12
print_line = 100


dataset_name = 'adl'
freeze_body = 1     # freeze_body = 1 -> freeze prertained layers
                    # freeze_body = 2 -> freeze all but 3 output layers
                    # freeze_body = otro -> don't freeze
input_shape = (416,416) # multiple of 32, hw
val_split = 0.1
batch_size_frozen = 32          # 32
batch_size_unfrozen = 32        # note that more GPU memory is required after unfreezing the body
frozen_epochs = 1               # 50


anchors_path = 'base_models/yolo_anchors.txt'

# Get dataset annotations, classes and anchors
if dataset_name == 'adl':
    path_dataset = '/home/asabater/projects/ADL_dataset/ADL_frames/'
    path_annotations = '/home/asabater/projects/ADL_dataset/annotations_adl.txt'
    path_classes = '/home/asabater/projects/ADL_dataset/adl_classes.txt'
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
anchors = ktrain.get_anchors(anchors_path)



title = 'Create and get model folders'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
# Create and get model folders
path_log, path_model = train_utils.create_model_folder(path_results, dataset_name)
print(path_model)
print('='*print_line)



# =============================================================================
# Create model
# =============================================================================

title = 'Create Keras model'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
model = ktrain.create_model(input_shape, anchors, num_classes,
            freeze_body = freeze_body, 
            weights_path='base_models/yolo.h5') # make sure you know what you freeze
print('='*print_line)



# Train callbacks
logging = TensorBoard(log_dir = path_model)
checkpoint = ModelCheckpoint(path_model + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                             monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    


# Train/Val split
with open(path_annotations) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val



title = 'Storing train params and model architecture'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
train_params = {
                'dataset_name': dataset_name,
                'freeze_body': freeze_body,
                'input_shape': input_shape,
                'val_split': val_split,
                'batch_size_frozen': batch_size_frozen,
                'batch_size_unfrozen': batch_size_unfrozen,
                'frozen_epochs': frozen_epochs,
                'anchors_path': anchors_path,
                'path_annotations': path_annotations,
                'path_classes': path_classes,
                'num_val': num_val,
                'num_train': num_train,
                }
print(train_params)
with open(path_model + 'train_params.json', 'w') as f:
    json.dump(train_params, f)
    
model_architecture = model.to_json()
with open(path_model + 'architecture.json', 'w') as f:
    json.dump(model_architecture, f)
print('='*print_line)



# =============================================================================
# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
# =============================================================================

title = 'Train first stage'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
if True:
    model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_frozen))
    model.fit_generator(
            ktrain.data_generator_wrapper(lines[:num_train], batch_size_frozen, input_shape, anchors, num_classes),
            steps_per_epoch = max(1, num_train//batch_size_frozen),
            validation_data = ktrain.data_generator_wrapper(lines[num_train:], batch_size_frozen, input_shape, anchors, num_classes),
            validation_steps = max(1, num_val//batch_size_frozen),
            epochs = frozen_epochs,
            initial_epoch = 0,
            callbacks=[logging, checkpoint])
    model.save_weights(path_model + 'trained_weights_stage_1.h5')
print('='*print_line)



# =============================================================================
# Unfreeze and continue training, to fine-tune.
# Train longer if the result is not good.
# =============================================================================

title = 'Train second stage'; print('{} {} {}'.format('='*print_indnt, title, '='*(print_line-2 - len(title)-print_indnt)))
if True:
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    print('Unfreeze all of the layers.')

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size_unfrozen))
    model.fit_generator(
        ktrain.data_generator_wrapper(lines[:num_train], batch_size_unfrozen, input_shape, anchors, num_classes),
        steps_per_epoch = max(1, num_train//batch_size_unfrozen),
        validation_data = ktrain.data_generator_wrapper(lines[num_train:], batch_size_unfrozen, input_shape, anchors, num_classes),
        validation_steps = max(1, num_val//batch_size_unfrozen),
        epochs = 500,
        initial_epoch = 50,
        callbacks = [logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(path_model + 'trained_weights_final.h5')
print('='*print_line)
    
    

