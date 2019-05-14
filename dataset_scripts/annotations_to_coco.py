#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import random


# %%

#a = json.load(open('/home/asabater/projects/annotations/instances_val2017.json', 'r'))


# %%

#dataset_annotations_filename = './adl/annotations_adl_val_416_v2_27.txt'

#dataset_classes_filename = './adl/adl_classes_v2_27.txt'
#with open(dataset_classes_filename, 'r') as f: dataset_classes = f.read().splitlines()


# %%

def annotations_to_coco(dataset_annotations_filename, dataset_classes_filename):
    
    with open(dataset_annotations_filename, 'r') as f: dataset_annotations = f.read().splitlines()
    with open(dataset_classes_filename, 'r') as f: dataset_classes = f.read().splitlines()


    annotations_coco = {
                        'info': {
                                    'description': dataset_annotations_filename,
                                },
                        'annotations': [],
                        'categories': [],
                        'images': [],
                        'licenses': []
            }


    # Add categories
    for i, c in enumerate(dataset_classes):
        annotations_coco['categories'].append({
                                        'supercategorie': '',
                                        'id': i,
                                        'name': c
                                    })


    # Add images and annotations
    count = 0
    for l in dataset_annotations:
        l = l.split()
        img = l[0]
        bboxes = l[1:]
        annotations_coco['images'].append({
                                'license': None,
                                'filename': img,
                                'coco_url': None,
                                'height': None,
                                'width': None,
                                'date_captured': None,
                                'flickr_url': None,
                                'id': img[:-4]
                    })
                            
        for bb in bboxes:
            bb = bb.split(',')
            cat = bb[-1]
            x_min, y_min, x_max, y_max = [ int(b) for b in bb[:-1] ]
            width = x_max-x_min
            height = y_max-y_min
            annotations_coco['annotations'].append({
                            'segmentations': [],
                            'area': width * height,
                            'iscrowd': 0,
                            'image_id': img[:-4],
    #                        'bbox': [ int(b) for b in bb ],
                            'bbox': [ x_min, y_min, width, height ],
                            'category_id': int(cat),
                            'id': count
                    })
            count += 1
        
    # Store annotations_coco
    annotations_coco_filename = dataset_annotations_filename[:-4] + '_coco.json'
    json.dump(annotations_coco, open(annotations_coco_filename, 'w+'))
    print('Saving:', annotations_coco_filename)






if __name__ == '__main__':
    # your code
    
    for ann in ['train', 'val']:
        for version in['v3_8', 'v2_27']:
#            dataset_annotations_filename = './adl/annotations_adl_{}_v2_27.txt'.format(ann)
            dataset_annotations_filename = './adl/annotations_adl_{}_{}_r_fd|10,20|.txt'.format(ann, version)
            dataset_classes_filename = './adl/adl_classes.txt'
    #        dataset_annotations_filename = './coco/annotations_coco_val.txt'
    #        dataset_classes_filename = './coco/coco_classes.txt'
            annotations_to_coco(dataset_annotations_filename, dataset_classes_filename)



