#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:06:30 2019

@author: asabater
"""

import os
os.chdir(os.getcwd() + '/..')


# %%

from evaluate_model import main
from tensorboard.backend.event_processing import event_accumulator
import train_utils
import numpy as np

# =============================================================================
# Loss plot
# =============================================================================

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

path_results = '/mnt/hdd/egocentric_results/'
dataset_name = 'adl'
model_num = 55

_, _, videos, _, resume, eval_stats_f, tp, loss_f = main(
	path_results, dataset_name, model_num=model_num, score=0, iou=0.5, num_annotation_file=1,
	plot=False, full=True, best_weights=False)


metrics = ['class_loss', 'confidence_loss_noobj', 'confidence_loss_obj', 'wh_loss', 'xy_loss'] #, 'confidence_loss'
linestyles = ['-', '--', '-.', ':']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
init_epoch = 4

#	model_nums = [(38, 'darknet'), (42, 'k cv1')]
model_nums = [(model_num, '')]
results = {}
train = True
metric_score = {}
#	metric_score = {'class_loss': 0.13, 'confidence_loss_obj': 0.04, 
#				'confidence_loss_noobj': 0.12} 	 	# Model 52
#	metric_score = {'class_loss': 1, 'confidence_loss_obj': 1, 
#				'confidence_loss_noobj': 0.25} 	 	# Model 55


plt.figure(figsize=(12,6))
for model_num, _ in model_nums:
	model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
	tb_files = [ model_folder + f for f in os.listdir(model_folder) if f.startswith('events.out.tfevents') ]

	metrics_train = { k:[] for k in metrics}
	metrics_val = { k:[] for k in metrics}
	for tbf in tb_files:
		try:
			ea = event_accumulator.EventAccumulator(tbf).Reload()
			train_loss, val_loss = [], []
			for k in metrics:
				metrics_train[k] += [ e.value for e in ea.Scalars('{}'.format(k)) ]
				metrics_val[k] += [ e.value for e in ea.Scalars('val_{}'.format(k)) ]
				train_loss += [ e.value for e in ea.Scalars('loss'.format(k)) ]
				val_loss += [ e.value for e in ea.Scalars('val_loss'.format(k)) ]
		except: continue
	
	results[model_num] = {'metrics_train': metrics_train,
							'metrics_val': metrics_val,
							'train_loss': train_loss,
							'val_loss': val_loss}
	

for model_num, _ in model_nums:
	for k,v in metric_score.items():
		print(k,v)
		results[model_num]['metrics_train'][k] = list(np.array(results[model_num]['metrics_train'][k])*v)
		results[model_num]['metrics_val'][k] = list(np.array(results[model_num]['metrics_val'][k])*v)


for (model_num,model), linestyle in zip(model_nums, linestyles):
	
	metrics_train, metrics_val, train_loss, val_loss = results[model_num].values()
	num_epochs = len(train_loss)
	for k, color in zip(metrics, colors):
		plt_mtrs = metrics_train if train else metrics_val
		plt.plot(range(init_epoch,len(plt_mtrs[k])), plt_mtrs[k][init_epoch:num_epochs], 
					linestyle=linestyle, color=color)
	plt.scatter([len(plt_mtrs[k])-10]*len(metrics), [plt_mtrs[k][-10] for k in metrics], marker='x')
		
	print('{:<8} | Train loss: {:.2f}, Val loss: {:.2f}'.format(model, min(train_loss), min(val_loss)))
	
plt.legend(handles = [Line2D([0], [0], color='black', linewidth=3, linestyle=ls, label=model) 
							for (_,model), ls in zip(model_nums, linestyles)] +
						[Line2D([0], [0], marker='o', markersize=12, color=c, linewidth=3, label=metric) 
							for metric, c in zip(metrics, colors)], bbox_to_anchor=(1, 1));



ref = np.array(results[model_num]['metrics_train']['wh_loss'][init_epoch:])
mtr = np.array(results[model_num]['metrics_train']['class_loss'][init_epoch:])

#	plt.plot(mtr/ref)


# %%

#plt.plot(np.array(results[52]['metrics_train']['xy_loss'][init_epoch:]))
#plt.plot(np.array(results[52]['metrics_train']['wh_loss'][init_epoch:]))
#plt.plot(np.array(results[52]['metrics_train']['class_loss'][init_epoch:])*0.13)
#plt.plot(np.array(results[52]['metrics_train']['confidence_loss_obj'][init_epoch:])*0.04)
#plt.plot(np.array(results[52]['metrics_train']['confidence_loss_noobj'][init_epoch:])*0.12)


