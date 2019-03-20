#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 10:18:27 2019

@author: asabater
"""

# for((i=10;i<=20;i+=1)); do ffmpeg -i ADL_videos/P_$i.MP4 ADL_frames/P_$i/%8d.jpg; done


from tqdm import tqdm
from adl_classes import classes
import os


base_data = '/home/asabater/projects/ADL_dataset/'
folder_annotations = base_data + 'ADL_annotations/object_annotation/'
#file_annotations = 'annotations_adl.txt'
frames_file = base_data + 'ADL_frames/'

img_default_size = (1280, 960)
output_size = 320
num_val_files = 3

x_scale = output_size / img_default_size[0]
y_scale = output_size / img_default_size[1]


# %%


def store_annotations(first_file, last_file, annotations_filename):

    annotations = {}
    
    
    for af_num in tqdm(range(first_file,last_file+1,1), total=20):
    
        with open(base_data + 'ADL_annotations/object_annotation/object_annot_P_{:0>2}_annotated_frames.txt'.format(af_num), 'r') as f:
            for l in f.read().splitlines():
                annotations['ADL_frames_{}/P_{:0>2}/{}'.format(output_size, af_num, l)] = []
            
        
        af = folder_annotations + 'object_annot_P_{:0>2}.txt'.format(af_num)
        with open(af, 'r') as f:
    
            for fa in f.read().splitlines():
                
                object_track_id, x1, y1, x2, y2, frame_number, active, label = [ i for i in fa.split(' ') if i != '' ]
                
                img_name = 'ADL_frames_{}/P_{:0>2}/{}'.format(output_size, af_num, frame_number)
                if img_name not in annotations:
                    annotations[img_name] = []
                    print(img_name)
                
                annotations[img_name].append({
                            'xmax': int(float(x2) * x_scale)*2,
                            'xmin': int(float(x1) * x_scale)*2,
                            'ymax': int(float(y2) * y_scale)*2,
                            'ymin': int(float(y1) * y_scale)*2,         
                            'class': classes.index(label)
                        })
                
    
    to_remove = []
    for k in annotations.keys():
        if not os.path.isfile(base_data + k + '.jpg'): 
            to_remove.append(k)
            print('Remove:', k)
    
    for k in to_remove: del annotations[k]
        
    
    with open(annotations_filename, 'w+') as f:
    
        for k, v in annotations.items():
            
            f.write('{}.jpg {}\n'.format(k, 
                        ' '.join([ '{},{},{},{},{}'.format(bb['xmin'],bb['ymin'],bb['xmax'],bb['ymax'],bb['class']) for bb in v ])
                    ))
    
    
    print(annotations_filename, 'writted')


store_annotations(1, 17, 'annotations_adl_train_{}.txt'.format(output_size))
store_annotations(18, 20, 'annotations_adl_val_{}.txt'.format(output_size))


# %%

if False:
    with open('annotations_adl.txt', 'r') as f: adl = f.readlines()
    adl = [ '/'.join(l.split(' ')[0].split('/')[-2:])[:-4] for l in adl ]
    
    
    orig_frames = []
    for i in range(1,21,1):
        with open('ADL_annotations/object_annotation/object_annot_P_{:0>2}_annotated_frames.txt'.format(i), 'r') as f:
            orig_frames += [ 'P_{:0>2}/{}'.format(i, l) for l in f.read().splitlines() ]
    
    
    dist = list(set(orig_frames) - set(adl))


# %%

#from collections import Counter
#
#stats = []
#
#for af_num in tqdm(range(1,21,1), total=20):
#
#    af = folder_annotations + 'object_annot_P_{:0>2}.txt'.format(af_num)
#    with open(af, 'r') as f:
#
#        elements = [ l.split()[7] for l in f.read().splitlines() ]
#        num_elements = len(elements)
#        elements = dict(Counter(elements).items())
#        elements = { k:v/num_elements for k,v in elements.items() }
#        stats.append(elements)
#
#df = pd.DataFrame(stats).fillna(0)
#
#means = df.mean()
#
#diffs = [ (i, sum(abs(r-means))) for i,r  in df.iterrows() ]
#diffs = sorted(diffs, key=lambda tup: tup[1])

