#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:14:23 2019

@author: asabater
"""

import os
os.chdir(os.getcwd() + '/..')

import sys
sys.path.append('keras_yolo3/')
sys.path.append('keras_yolo3/yolo3/')


# %%

from evaluate_model import main, MIN_SCORE
import keras_yolo3.train as ktrain
import matplotlib.pyplot as plt
import pyperclip
import numpy as np


# Given a video, plot its mAP for each category and the total category mAP
def plot_video_performance(video, class_names, eval_stats):
	plt.figure(figsize=(20,7))
	cats = [ int(k[4:]) for k in eval_stats.keys() if k.startswith('cat_') ]
	bars_video = [ eval_stats[video]['cat_{}'.format(c)][1] for c in cats ]
	
	plt.xticks(range(len(cats)), [ class_names[c] for c in cats ], rotation=45)
	for i in range(len(bars_video)): 
		if bars_video[i] == -1: plt.gca().get_xticklabels()[i].set_color("red")
	bars_video = [ b if b!= -1 else 0 for b in bars_video]
	plt.bar(np.arange(0, len(cats)), bars_video, alpha=0.5, width=0.35)
	
	bars_cat = [ eval_stats['cat_{}'.format(c)][1] for c in cats ]
	plt.bar(np.arange(0.35, len(cats)), bars_cat, color='y', alpha=0.5, width=0.35)
   
	plt.axhline(eval_stats[video]['total'][1])
	plt.axhline(eval_stats['total'][1], c='y')
	plt.suptitle('Performance of the video: ' + video, fontsize=20)
	plt.legend(["{} mAP@50".format(video), "Total mAP@50", 'video-cat mAP', 'cat mAP'])
	plt.show()
	

# Given a category, plot its mAP for each video and the total video mAP
def plot_category_performance(cat, class_names, eval_stats, videos):
	plt.figure(figsize=(20,7))
	cat_ind = class_names.index(cat)
	bars = [ eval_stats[v]['cat_{}'.format(cat_ind)][1] for v in videos ]
	
	plt.xticks(range(len(videos)), [ v for v in videos ], rotation=45)
	for i in range(len(bars)): 
		if bars[i] == -1: plt.gca().get_xticklabels()[i].set_color("red")
	bars = [ b if b!= -1 else 0 for b in bars]
	
	plt.bar(np.arange(0, len(videos)), bars, alpha=0.5, width=0.35)
	
	bars_video = [ eval_stats[v]['total'][1] for v in videos ]
	plt.bar(np.arange(0.35, len(videos)), bars_video, color='y', alpha=0.5, width=0.35)
	
	plt.axhline(eval_stats['cat_{}'.format(cat_ind)][1], label="'{}' mAP@50".format(cat))
	plt.axhline(eval_stats['total'][1], c='y', label="Total mAP@50")
#	plt.xticks(range(len(videos)), videos)
	plt.suptitle('Performance of the category: ' + cat, fontsize=20)
	plt.legend(["{} mAP@50".format(cat), "Total mAP@50", 'video-cat mAP', 'video mAP'])
	plt.show()
	
	
# Bar plot of mAP@50 for the selected models
# models -> {num_model: bar_label}
def model_comparision(models, train, path_results, dataset_name, iou=0.5, 
					  plot_loss=True, table=['data','arch','prtr','inp','map50','loss']):
	mAPs, losses, train_params = {}, {}, {}
	if train: 
		score, num_annotation_file = MIN_SCORE, 0
	else:
		score, num_annotation_file = 0, 1
	
	for model_num, label in models.items():
		if model_num < 0:
			mAPs[model_num] = 0
			losses[model_num] = 0
		else:
			# model, class_names, videos, occurrences, resume, eval_stats, train_params, loss
			_, _, _, _, _, eval_stats_t, _, loss_t = main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file,
					plot=False, full=True, best_weights=True)
			
			_, _, _, _, _, eval_stats_f, tp, loss_f = main(
					path_results, dataset_name, model_num, score, iou, num_annotation_file,
					plot=False, full=True, best_weights=False)
			
			train_params[model_num] = tp
			
			if eval_stats_t['total'][1] >= eval_stats_f['total'][1]:
				print('Model {} con best_weights: {:.2f}'.format(model_num, eval_stats_t['total'][1]))
				mAPs[model_num] = eval_stats_t['total'][1]
				losses[model_num] = loss_t[1]
			else:
				print('Model {} con stage_2: {:.2f}'.format(model_num, eval_stats_f['total'][1]))
				mAPs[model_num] = eval_stats_f['total'][1]
				losses[model_num] = loss_f[1]
			
	if not table:
		fig, ax1 = plt.subplots()
		ax1.bar(list(models.values()), list(mAPs.values()), alpha=0.6, color='b');
	
		if plot_loss:
			ax1.set_ylabel('mAP', color='b')
			ax1.tick_params(axis='y', labelcolor='b')
			ax2 = ax1.twinx()
			ax2.plot(list(models.values()), list(losses.values()), color='g');
			ax2.set_ylabel('loss', color='g')
			ax2.tick_params(axis='y', labelcolor='g')
		else:
			ax1.set_ylabel('mAP')
	
		fig.tight_layout()
		
		return fig
	
	else:
		['arch','inp','map50','loss']
		table_data = []
		if 'num' in table: table_data.append('model\_num')
		if 'data' in table: table_data.append('dataset')
		if 'arch' in table: table_data.append('architecture')
		if 'prtr' in table: table_data.append('pretraining')
		if 'inp' in table: table_data.append('input size')
		if 'loss' in table: table_data.append('Loss')
		if 'map50' in table: table_data.append('mAP\\textsubscript{50}')
		table_data = ' & '.join(table_data) + ' \\\\ \hline\n'
		
		
		for model_num, label in models.items():
			if model_num < 0: 
				row = ['{} {}'.format(model_num, label)] + ['--']*(len(table)-1)
			else:
				row = []
				if 'num' in table: row.append(str(model_num))
				if 'data' in table: 
					if 'v2' in train_params[model_num]['version']: row.append('v2')
					elif 'v3' in train_params[model_num]['version']: row.append('v3')
					else: row.append('raw')
				if 'arch' in table: 
					tiny_version = len(ktrain.get_anchors(train_params[model_num]['path_anchors'])) == 6
					if tiny_version: row.append('tiny')
					elif train_params[model_num].get('spp', False): row.append('spp')
					else: row.append('base')			
				if 'prtr' in table:
					if 'darknet' in train_params[model_num]['path_weights']: row.append('backbone')
					elif train_params[model_num]['path_weights'] == '': row.append('--')
					elif 'model_0' in train_params[model_num]['path_weights']: row.apend('kitchen 17')
					elif 'model_1' in train_params[model_num]['path_weights']: row.apend('kitchen 18')
					else: row.append('coco')
				if 'inp' in table: row.append(str(train_params[model_num]['input_shape'][0]))
				if 'loss' in table: row.append('{:.2f}'.format(losses[model_num]))
				if 'map50' in table: row.append('{:.3f}'.format(mAPs[model_num]*100))
			
			table_data += ' & '.join(row) + ' \\\\ \hline\n'
			
#		table_data = '\\begin{table}[!htb]\n\centering\n' + \
#						'\\begin{tabular}{|' + '{}'.format('l|'*len(table)) + '}\n' + \
#						'\hline\n' + table_data + \
#						'\end{tabular}\n\caption[Caption for LOF]{}\n\label{tab:}\n\end{table}'
		return table_data
	
	

# %%

# =============================================================================
# Comparación de modelos
# =============================================================================

path_results = '/mnt/hdd/egocentric_results/'
dataset_name = 'adl'
plot_loss = False
#	table=['data','arch','prtr','inp','map50','loss']
table = False

# Por version
#	models, fig_name = {15: 'raw', 13: 'v2_27', 16: 'v3_8'}, 'versions_mAP' 	 

# v2 Por tamaño  
#	models, fig_name = {45: '320 x 320', 13: '416 x 416', 44: '608 x 608'}, 'image_size_v2'
 	
# v3 Por tamaño
#	models, fig_name = {18: '320 x 320', 16: '416 x 416', 17: '608 x 608'}, 'image_size_v3'

# v2 por pretraining
#	models, fig_name, plot_loss = {35: 'no pretraining', 38: 'darknet', 13: 'yolo_coco',
#			42: 'kitchen 17', 43: 'kitchen 18'}, 'pretraining_v2', True

# v3 por pretraining
#	models, fig_name, plot_loss = {30: 'no pretraining', 34: 'darknet', 16: 'yolo_coco', 
#		   31: 'kitchen 17', 32: 'kitchen 18'}, 'pretraining_v3', True

# TODO: v2 Por modelo
#	models, fig_name = {46: 'tiny', 13: 'yolo', 47: 'yolo SPP'}, 'model_v2'

# v3 Por modelo
#	models, fig_name = {39: 'tiny', 16: 'yolo', 37: 'yolo SPP'}, 'model_v3'

# TODO: v2 SPP
#	models, fig_name = {45: 'yolo 320', 49: 'spp 320',
#					 13: 'yolo 416', 47: 'spp 416', 
#					 48: 'spp 608', 44: 'yolo 608'}, 'spp_v2' 			# , 41: 'spp 416 backbone'

## TODO: v3 SPP
#	models, fig_name = {18: 'yolo 320', -10: 'spp 320',
#					 16: 'yolo 416', 37: 'spp 416', 
#					 17: 'spp 608', 40: 'yolo 608'}, 'spp_v3' 			# , 41: 'spp 416 backbone'

models, _, table = {15: 'raw', 51: 'spp',
  # v2 
  13: 'base', 35: 'no_pretraining', 38: 'darknet',
  45: '320', 44: '608', 49: 'spp 320', 47: 'spp 416', 48: 'spp 608',
  46: 'tiny',
  # v3
  16: 'base', 30: 'no_pretraining', 34: 'darknet',
  18: '320', 17: '608', 50: 'spp 320', 37: 'spp 416', 40: 'spp: 608',
  39: 'tiny'}, '', ['data','arch','prtr','inp','map50','loss']

res = model_comparision(models, False, path_results, dataset_name, plot_loss=plot_loss, table=table)
if not table:
	plt.grid()
	plt.savefig('{}figuras_05.2019/{}.png'.format(path_results, fig_name))
	print(fig_name + ' guardado')
else:
	print(res)
	pyperclip.copy(res)


# %%

if False:
	# %%
	
	model_num = 52
	
	model, class_names, videos, occurrences, resume, eval_stats, train_params, loss = main(
					path_results, dataset_name, model_num, score=0, iou=0.5, num_annotation_file=1,
					plot=False, full=True, best_weights=False)	
	
	plot_category_performance('laptop', class_names, eval_stats, videos)
	plot_video_performance('P_20', class_names, eval_stats)
	
	# %%
	
	def get_cat_map(eval_stats, classes, class_name):
		return eval_stats['cat_{}'.format(classes.index(class_name))][1]
	
	score = '{:.2f}'.format(get_cat_map(eval_stats, class_names, 'bottle')*100)
	print(score)
	pyperclip.copy(score)




