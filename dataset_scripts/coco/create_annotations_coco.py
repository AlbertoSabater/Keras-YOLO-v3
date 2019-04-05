#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:40:07 2019

@author: asabater
"""

import os
import json
from tqdm import tqdm


instances = [
                ('annotations_coco_val', 'val2017/', '/mnt/hdd/datasets/coco/annotations_trainval2017/annotations/instances_val2017.json'),
                ('annotations_coco_train', 'train2017/', '/mnt/hdd/datasets/coco/annotations_trainval2017/annotations/instances_train2017.json'),
            ]


# %%

inst = json.load(open(instances[1][2], 'r'))


# %%

#categories_super = [ d['supercategory'] for d in inst['categories'] ]
categories = [d['name'] for d in inst['categories'] ]
categories_dict = { d['id']:i for d, i in zip(inst['categories'], range(len(categories))) }


categories_super = []
categories_super_dict = {}
count = -1
for d in inst['categories']:
    if d['supercategory'] not in categories_super:
        categories_super.append(d['supercategory'])
        count += 1
    categories_super_dict[d['id']] = count

        

with open('coco_classes_super.txt', 'w') as f:
    for c in categories_super:
        f.write(c + '\n')
        
with open('coco_classes.txt', 'w') as f:
    for c in categories:
        f.write(c + '\n')
        

# %%


for save_filename, prefix, path_instances in instances:
    print(save_filename)
    
    inst = json.load(open(path_instances, 'r'))
    images = { d['id']:d['file_name'] for d in inst['images'] }


    img_boxes = {}
    img_boxes_super = {}
    for ann in inst['annotations']:
        image_id = ann['image_id']
        
        if image_id not in img_boxes: 
            img_boxes[image_id] = []
            img_boxes_super[image_id] = []
        
        x, y, width, height = [ int(b) for b in ann['bbox'] ]
        img_boxes[image_id].append('{},{},{},{},{}'.format(x, y, x+width, y+height, 
                 categories_dict[ann['category_id']]))
        img_boxes_super[image_id].append('{},{},{},{},{}'.format(x, y, x+width, y+height, 
                 categories_super_dict[ann['category_id']]))


    with open(save_filename + '.txt', 'w') as f:
        for k,v in tqdm(img_boxes.items(), total=len(img_boxes)):
            f.write(prefix + images[k] + ' ' + ' '.join(v) + '\n')
            
    with open(save_filename + '_super.txt', 'w') as f:
        for k,v in tqdm(img_boxes_super.items(), total=len(img_boxes_super)):
            f.write(prefix + images[k] + ' ' + ' '.join(v) + '\n')
            

# %%
            









