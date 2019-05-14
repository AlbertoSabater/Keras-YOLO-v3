#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:04:30 2019

@author: asabater
"""
import sys
sys.path.append('..')

import pandas as pd
import ast
from tqdm import tqdm
import numpy as np
from annotations_to_coco import annotations_to_coco
from PIL import Image, ImageFont, ImageDraw


# TODO: escribir fichero de classes

base_path = '/home/asabater/projects/epic_dataset/' 
#mode = 'train'
#file_annotations = base_path + 'annotations_epic_{}.txt'.format(mode)


num_classes = [-1, 15, 25, 35]
#num_classes = [25]



df = pd.read_csv(base_path + 'annotations/EPIC_train_object_labels.csv')
classes = pd.read_csv(base_path + 'annotations/EPIC_noun_classes.csv')
#df['noun_id'] = df.noun_class.apply(lambda x: classes[classes.noun_id == x].class_key.values[0])

p_ids = sorted(df.participant_id.drop_duplicates())
train_ids = p_ids[:int(len(p_ids)*0.8)]
val_ids = [ pi for pi in p_ids if pi not in train_ids ]



# %%

df['img'] = df.apply(lambda r: '/home/asabater/projects/epic_dataset/EPIC_KITCHENS_2018/object_detection_images/{}/{}/{}/{:0>10}.jpg'.format(
            'train', r['participant_id'], r['video_id'], r['frame']
        ), axis=1)

df['bounding_boxes'] = df['bounding_boxes'].apply(lambda x: ast.literal_eval(x))
df['bounding_boxes'] = df.apply(lambda x: [ bb + (x['noun_class'],) for bb in x['bounding_boxes']], axis=1)

print('len == 0:', sum(df.bounding_boxes.apply(lambda x: len(x)==0)))
print('num_boxes:', sum(df.bounding_boxes.apply(lambda x: len(x))))
df = df[df.bounding_boxes.apply(lambda x: len(x)>0)]

df['image_size'] = df.img.apply(lambda x: Image.open(x).size)


# Filter out small boxes -> not recognized in a 320 image size
df['bounding_boxes'] = df.apply(lambda x: [ bb for bb in x['bounding_boxes'] 
									if bb[3] > (x['image_size'][1]/320) +1
										and bb[2] > (x['image_size'][1]/320) +1 ], axis=1)
print('len == 0:', sum(df.bounding_boxes.apply(lambda x: len(x)==0)))
print('num_boxes:', sum(df.bounding_boxes.apply(lambda x: len(x))))
df = df[df.bounding_boxes.apply(lambda x: len(x)>0)]


# %%

cm = []
indexes = []

for participant_id, g in df.groupby('participant_id'):
	cm.append(g.groupby('noun_class').size().to_dict())
	indexes.append(participant_id)

	
cm = pd.DataFrame(cm)
cm.index = indexes
cm.columns = [ classes[classes.noun_id == cl].class_key.values[0] for cl in cm.columns ]

res = [ len([ r for r in cm[col] if np.isnan(r) ]) for col in cm.columns ]
counts = list(zip(cm.columns, res))
counts = sorted(counts, key=lambda x: x[1])

cm = cm.loc[:, [ c[0] for c in counts ]]
cm.columns = [ '{}-{}'.format(i,c) for i,c in enumerate(cm.columns) ]
	

# %%

for version, n_classes in enumerate(num_classes):
	
	v_class_names = [ k for k,v in counts[:n_classes]]
	v_class_ids = [ classes[classes.class_key == cn].noun_id.values[0] for cn in v_class_names ]
	v_suffix = '_v{}_{}'.format(version, n_classes) if n_classes != -1 else ''
	
	annotations_train_filename = 'annotations_kitchen_train{}.txt'.format(v_suffix)
	annotations_val_filename = 'annotations_kitchen_val{}.txt'.format(v_suffix)
	classes_filename = 'kitchen_classes{}.txt'.format(v_suffix)
	
	with open(annotations_train_filename, 'w') as f_train, \
			open(annotations_val_filename, 'w') as f_val:
	
		for img, g in tqdm(df[df.noun_class.isin(v_class_ids)].groupby('img'), total=len(df.img.drop_duplicates()), file=sys.stdout):


			bbs = sum(g['bounding_boxes'], [])
#			bbs = [ bb for bb in bbs if bb[2]>=min_pixels and bb[3]>=min_pixels ]
			
			if len(bbs) > 0:	        
				image = Image.open(img)
				image_size = image.size
#
				annotation = '{} {}\n'.format(img, 
							   ' '.join([ '{},{},{},{},{}'.format(bb[1],bb[0],
											min(bb[1]+bb[3], image_size[0]),
											min(bb[0]+bb[2], image_size[1]),
											v_class_ids.index(bb[4])) 
									   for bb in bbs ])
							         )
				if g.participant_id.iloc[0] in train_ids:
					f_train.write(annotation)
				else:
					f_val.write(annotation)
					
	with open(classes_filename, 'w') as f:
#		for c_id in sorted(v_class_ids):
#			f.write('{}\n'.format(classes[classes.noun_id == c_id].class_key.values[0]))
		for c in v_class_names:
			f.write('{}\n'.format(c))

	print('Annotations and classes {} writted'.format(v_suffix))
	annotations_to_coco(annotations_train_filename, classes_filename)
	annotations_to_coco(annotations_val_filename, classes_filename)


