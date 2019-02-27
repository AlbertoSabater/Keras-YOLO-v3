#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 15:37:08 2019

@author: asabater
"""

import os
import datetime


def create_folder_if_not_exists(folder_path):
    if not os.path.isdir(folder_path): os.makedirs(folder_path)
    

def create_model_folder(path_results, dataset_name):
    
    path_log = path_results + dataset_name + '/logs/'
    path_model = path_results + dataset_name + '/models/'
    
    # Create logs and model main folder
    create_folder_if_not_exists(path_log)
    create_folder_if_not_exists(path_model)

    model_name = get_model_name(path_results + dataset_name + '/models/')
    path_log += model_name; path_model += model_name
    create_folder_if_not_exists(path_log)
    create_folder_if_not_exists(path_model)
    
    return path_log, path_model


def get_model_name(model_dir):
    folder_num = len(os.listdir(model_dir))
    return '{}_model_{}/'.format(datetime.datetime.today().strftime('%m%d_%H%M'), folder_num)


# TODO
def remove_empty_folders(path_results, dataset_name):
    pass





