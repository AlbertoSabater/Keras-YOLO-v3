#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 16:53:38 2019

@author: asabater
"""

import sys
sys.path.append('keras_yolo3/')

import keras_yolo3.train as ktrain
from dataset_scripts.annotations_to_coco import annotations_to_coco
from dataset_scripts.generate_custom_anchors import EYOLO_Kmeans


remove_empty_frames = True
cluster_number = 9


dataset_version = ''
path_annotations = ['./dataset_scripts/kitchen/annotations_kitchen_train{}.txt'.format(dataset_version),
					 './dataset_scripts/kitchen/annotations_kitchen_val{}.txt'.format(dataset_version)]
path_classes = './dataset_scripts/kitchen/kitchen_classes{}.txt'.format(dataset_version)
class_names = ktrain.get_classes(path_classes)
	
	
merge_classes = [
					(['spoon', 'knife', 'fork', 'cutlery', 'ladle', 'utensil', 'spatula', 'tongs'], 'cutlery'),
					(['tap'], 'tap'),
					(['plate', 'bowl'], 'plate/bowl'),
					(['fridge', 'freezer'], 'fridge/freezer'),
					(['pan', 'pot', 'cooker:slow'], 'pan/pot'),
					(['salt', 'oil', 'vinegar'], 'salt/oil/vinegar'),
					(['cup', 'glass'], 'cup/glass'),
					(['bottle', 'jar', 'jug', 'liquid', 'liquid:washing', 'drink', 'beer', 'coke', 'can', 'soap'], 'bottle/soup'),
					(['bag', 'package', 'box', 'container'], 'bag/container'),
					(['hob', 'oven', 'heater'], 'oven'),
					(['bin', 'basket'], 'bin/basket'),
					(['cloth', 'clothes', 'shirt', 'sock', 'trousers', 'towel', 'napkin', 'tablecloth', 'boxer'], 'cloth'),
					(['kettle'], 'kettle'),
					(['microwave'], 'microwave'),
					(['colander'], 'colander'),
					(['maker:coffee', 'coffee'], 'coffee'),
					(['dishwasher', 'machine:washing'], 'washer'),
					(['food', 'onion', 'pepper', 'bread', 'tomato', 'cheese', 'salad', 'garlic', 'chicken', 'egg', 'sausage', 'sugar', 'potato', 'pasta', 'carrot', 'rice', 'cereal', 'lettuce', 'meat', 'lemon', 'corn', 'tortilla', 'olive', 'vegetable', 'cucumber', 'fish', 'chilli', 'bean:green', 'tea', 'noodle', 'onion:spring', 'courgette', 'pizza', 'banana', 'bacon', 'ingredient', 'paste:garlic', 'butter', 'tofu', 'cream', 'burger:tuna', 'salami', 'biscuit', 'wrap', 'squash', 'aubergine', 'mushroom', 'grape', 'tuna', 'avocado', 'blueberry', 'lime', 'nut:pine', 'raisin', 'celery', 'spinach', 'fruit', 'sandwich', 'pancake', 'leftover', 'pea', 'paprika', 'apple', 'beef', 'cumin', 'peach', 'leaf:mint', 'mat:sushi', 'leek', 'pie', 'broccoli', 'herb', 'kiwi', 'hummus', 'coconut', 'cabbage', 'nutella', 'omelette', 'paella', 'chip', 'risotto', 'soup', 'pear', 'berry', 'stick:crab', 'ring:onion', 'cherry', 'cake', 'sprout', 'almond', 'mint', 'flake:chilli', 'nesquik', 'melon', 'smoothie', 'grass:lemon', 'dumpling', 'crisp', 'caper', 'chocolate', 'popcorn', 'bar:cereal', 'pineapple', 'turkey', 'curry', 'parsley', 'basil', 'coriander', 'leaf'], 'food'),
		]	

remove_classes = [ c for c in class_names if c not in [ v for k,v in merge_classes ] ]

version = 'cv2'


def parse_annotation(annotation):
	ann = annotation.split()
	img = ann[0]
	boxes = [ [int(b) for b in bb.split(',')] for bb in ann[1:]]
	return img, boxes

def merge_classes_in_annotations(annotations, merge_classes, class_names):
	for mc, new_class in merge_classes:
		
		mc_inds = [ class_names.index(c) for c in mc ]
	
		base_ind = min(mc_inds)
		inds_to_remove = sorted([ i for i in mc_inds if i != base_ind], reverse=True)
		class_names[base_ind] = new_class
		for itr in inds_to_remove: del class_names[itr]
		
		for ann_ind in range(len(annotations)):
			img, boxes = parse_annotation(annotations[ann_ind])		
			
			new_boxes = []
			for x_min,y_min,x_max,y_max,class_id in boxes:
				if class_id <= base_ind: pass
				elif class_id in inds_to_remove: class_id = base_ind
				else:
					for itr in inds_to_remove: 
						if class_id > itr: class_id -= 1
				if class_id < 0: print(class_id, base_ind, inds_to_remove)
				new_boxes.append('{},{},{},{},{}'.format(x_min,y_min,x_max,y_max,class_id)) 
	
			annotations[ann_ind] = img + ' ' + ' '.join(new_boxes)
	
	return annotations, class_names

def remove_classes_in_annotations(annotations, remove_classes, class_names):
	for rc in remove_classes:
		
		if rc not in class_names:
			print(rc, 'not in class_names')
			continue
		
		rc_ind = class_names.index(rc)
		
		del class_names[rc_ind]
	
		for ann_ind in range(len(annotations)):
			img, boxes = parse_annotation(annotations[ann_ind])		
	
			new_boxes = []
			for x_min,y_min,x_max,y_max,class_id in boxes:
				if class_id < rc_ind: class_id = class_id
				elif class_id == rc_ind: continue
				else: class_id -= 1
				new_boxes.append('{},{},{},{},{}'.format(x_min,y_min,x_max,y_max,class_id)) 
	
			annotations[ann_ind] = img + ' ' + ' '.join(new_boxes)
			
	return annotations, class_names



for annotations_file in path_annotations:
	
	class_names = ktrain.get_classes(path_classes)
	with open(annotations_file) as f: annotations = [ l for l in f.read().splitlines() ]
	
	annotations, class_names = merge_classes_in_annotations(annotations, merge_classes, class_names)
	annotations, class_names = remove_classes_in_annotations(annotations, remove_classes, class_names)
	
	if remove_empty_frames:
		annotations = [ ann for ann in annotations if len(ann.split())>1 ]

	# Store annotations
	new_annotations_file = annotations_file[:-4] + '_{}_{}.txt'.format(version, len(class_names))
	print('Saving:', new_annotations_file)
	with open(new_annotations_file, 'w') as f:
		for ann in annotations:
			f.write(ann + '\n')
			
	new_classes_file = path_classes[:-4] + '_{}_{}.txt'.format(version, len(class_names))
	print('Saving:', new_classes_file)
	with open(new_classes_file, 'w') as f:
		for c in class_names:
			f.write(c + '\n')
			
	annotations_to_coco(new_annotations_file, new_classes_file)

#	anchors_filename = '/'.join(annotations_file.split('/')[:-1]) + \
#								'/anchors_adl{}_{}.txt'.format(version, len(class_names))
#	kmeans = EYOLO_Kmeans(cluster_number, new_annotations_file)
#	anchors = kmeans.get_best_anchors()
#	kmeans.result2txt(anchors, anchors_filename)


#%%
	
if False:
	
	path_classes = './dataset_scripts/adl/adl_classes.txt'
	path_annotations = ['./dataset_scripts/adl/annotations_adl_train{}.txt',
							'./dataset_scripts/adl/annotations_adl_val{}.txt'
							]
	#version = 'v2'
	#merge_classes = [
	#					(['trash_can', 'basket', 'container', 'large_container'], 'generic_container'),
	#					(['monitor', 'tv'], 'monitor/tv'),
	#					(['shoe', 'shoes'], 'shoes'),
	#					(['perfume', 'bottle', 'milk/juice'], 'bottle'),
	#					(['cell', 'cell_phone'], 'cell_phone')
	#				]
	#remove_classes = ['keyboard', 'dent_floss', 'blanket']
	#remove_classes += ['comb', 'thermostat', 'tea_bag', 'pills', 'mop', 'vacuum', 
	#				   'bed', 'detergent', 'electric_keys']
	
	version = 'v3'
	merge_classes = [(['monitor', 'tv'], 'monitor/tv')]
	class_names = ktrain.get_classes(path_classes)
	remove_classes = [ c for c in class_names if c not in 
							  ['mug/cup', 'tap', 'laptop', 'monitor/tv', 
							   'knife/spoon/fork', 'washer/dryer','pan', 'microwave'] ]

	for suffix in ['', '_320', '_416', '_608']:
		for annotations_file in path_annotations:
			annotations_file = annotations_file.format(suffix)
			
			class_names = ktrain.get_classes(path_classes)
			with open(annotations_file) as f: annotations = [ l for l in f.read().splitlines() ]
			
			annotations, class_names = merge_classes_in_annotations(annotations, merge_classes, class_names)
			annotations, class_names = remove_classes_in_annotations(annotations, remove_classes, class_names)
			
			if remove_empty_frames:
				annotations = [ ann for ann in annotations if len(ann.split())>1 ]
		
		
			# Store annotations
			new_annotations_file = annotations_file[:-4] + '_{}_{}.txt'.format(version, len(class_names))
			print('Saving:', new_annotations_file)
			with open(new_annotations_file, 'w') as f:
				for ann in annotations:
					f.write(ann + '\n')
		
		
			new_classes_file = path_classes[:-4] + '_{}_{}.txt'.format(version, len(class_names))
			print('Saving:', new_classes_file)
			with open(new_classes_file, 'w') as f:
				for c in class_names:
					f.write(c + '\n')
					
			annotations_to_coco(new_annotations_file, new_classes_file)
	
			anchors_filename = '/'.join(annotations_file.split('/')[:-1]) + \
										'/anchors_adl{}_{}_{}.txt'.format(suffix, version, len(class_names))
			kmeans = EYOLO_Kmeans(cluster_number, new_annotations_file)
			anchors = kmeans.get_best_anchors()
			kmeans.result2txt(anchors, anchors_filename)


# %%

#clss = []
#for ann_ind in range(len(annotations)):
#	img, boxes = parse_annotation(annotations[ann_ind])
#	for x_min,y_min,x_max,y_max,class_id in boxes:
#		clss.append(class_id)
#clss = list(set(clss))
		





