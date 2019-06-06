#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:41:19 2019

@author: asabater
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = ""; 

os.chdir(os.getcwd() + '/..')

import tensorflow as tf
import keras.backend as K
from keras.layers import Input
from emodel import yolo_body, tiny_yolo_body
import pyperclip


# %%
	
#img_size = 416
#num_anchors = 9
#num_classes = 20
#spp = True
#is_tiny_version = num_anchors==6


def get_tradeoff(model_data, num_classes):
	run_meta = tf.RunMetadata()

	with tf.Session(graph=tf.Graph()) as sess:
		
		K.set_session(sess)
	
		image_input = Input(shape=(model_data['inpt'], model_data['inpt'], 3))
		
		if model_data['arch'] == 'tiny-yolo':
			model = tiny_yolo_body(Input(shape=(None,None,3)), 3, num_classes)
		else:
			model = yolo_body(image_input, 3, num_classes, model_data['arch']=='SPP')
			
		model(tf.placeholder('float32', shape=(1,model_data['inpt'],model_data['inpt'],3)))
		
		opts = tf.profiler.ProfileOptionBuilder.float_operation()	
		flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
	
		opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()	
		params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
	
		params = params.total_parameters
		flops = flops.total_float_ops
		print("Trainable params: {:,}".format(params))
		print("FLOPs/sample: {:,}".format(flops))
		
		return flops, params
		


# %%

num_classes = 27
models = [
			{'arch': 'base', 'inpt': 320, 'v2': 36.771, 'v3': 48.326}, 
			{'arch': 'base', 'inpt': 416, 'v2': 38.794, 'v3': 51.145}, 
			{'arch': 'base', 'inpt': 608, 'v2': 39.188, 'v3': 51.692},
			{'arch': 'SPP', 'inpt': 320, 'v2': 36.417, 'v3': 48.304}, 
			{'arch': 'SPP', 'inpt': 416, 'v2': 39.496, 'v3': 51.514}, 
			{'arch': 'SPP', 'inpt': 608, 'v2': 38.850, 'v3': 50.705},
#			{'arch': 'tiny-yolo', 'inpt': 320, 'v2': 0, 'v3': 0}, 
			{'arch': 'tiny-yolo', 'inpt': 416, 'v2': 26.989, 'v3': 42.798}, 
#			{'arch': 'tiny-yolo', 'inpt': 608, 'v2': 0, 'v3': 0},
		]


for i, model_data in enumerate(models):
	flops, params = get_tradeoff(model_data, num_classes)
	models[i]['flops'] = flops
	models[i]['params'] = params


# %%

table_data = ''
for model_data in models:
	table_data += '{:<10} & {} x {} & {:.2f} Bn & {:.2f} M & {} & {} \\\\ \hline \n'.format(
				model_data['arch'], model_data['inpt'], model_data['inpt'],
				model_data['flops']/1000000000, model_data['params']/1000000, 
				model_data['v2'], model_data['v3']
			)


#table_data = '\\begin{table}[!htb]\n\centering\n' + \
#				'\\begin{tabular}{|l|l|l|l|l|}' + \
#				'\hline\n' + table_data + \
#				'\end{tabular}\n\caption[Caption for LOF]{}\n\label{tab:}\n\end{table}'

print(table_data)
pyperclip.copy(table_data)

