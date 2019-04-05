#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:53:31 2019

@author: asabater
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";  

import sys
sys.path.append('keras_yolo3/')
sys.path.append('keras_yolo3/yolo3/')

import json
import os
import numpy as np
import pandas as pd
import keras_yolo3.train as ktrain
#from keras.models import model_from_json
import train_utils

from tensorboard.backend.event_processing import event_accumulator
import datetime
import time

import cv2
from PIL import Image
from tqdm import tqdm

from eyolo import EYOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pyperclip
import matplotlib.pyplot as plt
import time


MIN_SCORE = 0.00005


# TODO: sacar métricas con diferentes IoU thresholds

def get_train_resume(model_folder):
    tb_files = [ model_folder + f for f in os.listdir(model_folder) if f.startswith('events.out.tfevents') ]
    
    train_losses, val_losses = [], []
    times = []
    for tbf in tb_files:
        ea = event_accumulator.EventAccumulator(tbf).Reload()
        train_losses += [ e.value for e in ea.Scalars('loss') ]
        val_losses += [ e.value for e in ea.Scalars('val_loss') ]
        times += [ e.wall_time for e in ea.Scalars('val_loss') ]
    
    
#    num_epochs = len(train_losses)
    val_loss = min(val_losses)
    train_loss = train_losses[val_losses.index(val_loss) ]
    
    train_init, train_end = min(times), max(times)
    
    train_init = datetime.datetime.fromtimestamp(train_init)
    train_end = datetime.datetime.fromtimestamp(train_end)
    
    train_diff = (train_end - train_init)
    train_diff = '{}d {:05.2f}h'.format(train_diff.days, train_diff.seconds/3600)
    
    return train_diff, train_loss, val_loss


def predict_and_store_from_annotations(model_folder, train_params, annotations_file, 
                                       preds_filename, score, iou):

    if os.path.isfile(preds_filename): return None, -1
    
    model = EYOLO(
                    model_image_size = tuple(train_params['input_shape']),
                    model_path = train_utils.get_best_weights(model_folder),
                    classes_path = train_params['path_classes'],
                    anchors_path = train_params['path_anchors'],
                    score = score,    # 0.3
                    iou = iou,      # 0.5
                )
    
    
    # Create pred file
    with open(annotations_file, 'r') as f: annotations = f.read().splitlines()
    preds = []
    
    t = time.time()
    total = len(annotations)
    for ann in tqdm(annotations[:total], total=total, file=sys.stdout):
        img = ann.split()[0]
        
        image = cv2.imread(train_params['path_dataset'] + img)
        image = Image.fromarray(image)
        boxes, scores, classes = model.get_prediction(image)
        
        for i in range(len(boxes)):
#            x_min, y_min, x_max, y_max = [ int(b) for b in boxes[i].tolist() ]
#            width = x_max-x_min
#            height = y_max-y_min
            left, bottom, right, top = [ int(b) for b in boxes[i].tolist() ]
            width = top-bottom
            height = right-left
            preds.append({
                            'image_id': img[:-4],
                            'category_id': int(classes[i]),
#                            'bbox': [ x_min, y_min, width, height ],
                            'bbox': [ bottom, left, width, height ],
                            'score': float(scores[i]),
                        })
    	
    fps = len(annotations)/(time.time()-t)
    json.dump(preds, open(preds_filename, 'w'))
    
#    preds_filename_new = '{}preds_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], MIN_SCORE, iou)
#    preds_train_new = [ pr for pr in preds_train if pr['score'] > MIN_SCORE ]
#    json.dump(preds_train_new, open(preds_filename_new, 'w'))
    
    return model, fps


def get_full_evaluation(annotations_file, preds_filename, eval_filename, class_names, full):
    
    eval_filename_full = eval_filename.replace('.json', '_full.json')
    ann = annotations_file[:-4] + '_coco.json'
    
    if full and os.path.isfile(eval_filename_full):
        # If full exists return full
        eval_stats =  json.load(open(eval_filename_full, 'r'))
        videos = [ k for k in eval_stats.keys() if not k.startswith('cat_') and not k.startswith('total') ]
        return eval_stats, videos
    elif os.path.isfile(eval_filename):
        # If regular eval exists load id
        eval_stats =  json.load(open(eval_filename, 'r'))
        
        if not full:
            videos = [ k for k in eval_stats.keys() if not k.startswith('cat_') and not k.startswith('total') ]
            return eval_stats, videos      
    else:
        # if no eval has been performed, initialize the dictionary and COCO
        eval_stats = {}

    cocoGt = COCO(ann)
    cocoDt = cocoGt.loadRes(preds_filename)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')        
    
#    if os.path.isfile(eval_filename): 
#        eval_stats =  json.load(open(eval_filename, 'r'))
##        videos = [ k for k in eval_stats.keys() if k.startswith('P_') ]
##        videos = [ k for k in eval_stats.keys() if not k.startswith('cat_') and not k.startswith('total') ]
##        return eval_stats, videos
#    else:
#        eval_stats = {}
        
    

#    eval_stats = {}
    

    
    if 'total' not in eval_stats:
        print(' Evaluating total')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        eval_stats['total'] = cocoEval.stats.tolist()
        json.dump(eval_stats, open(eval_filename, 'w'))
        
    if not full:
        # if not full return regular eval if not, continue the evaluation
        videos = [ k for k in eval_stats.keys() if not k.startswith('cat_') and not k.startswith('total') ]
        return eval_stats, videos  
    
    else:
    
        params = cocoEval.params
        imgIds = params.imgIds
        catIds = params.catIds
        
        # Evaluate each category
        for c in range(len(class_names)):
            print(' * Evaluationg the category |{}|'.format(class_names[c]))
            params.catIds = [c]
            cocoEval.evaluate()
            cocoEval.accumulate(params)
            cocoEval.summarize()
            eval_stats['cat_{}'.format(c)] = cocoEval.stats.tolist()
            params.catIds = catIds
        
        
        # Evaluate each video and each category per video
        videos = sorted(list(set([ i.split('/')[-2] for i in imgIds ])))
        for v in videos:
            print(' * Evaluating the video |{}|'.format(v))
            eval_stats[v] = {}
            v_imgIds = [ i for i in imgIds if v in i ]
            params.imgIds = v_imgIds
            cocoEval.evaluate()
            cocoEval.accumulate(params)
            cocoEval.summarize()
            eval_stats[v]['total'] = cocoEval.stats.tolist()
            
            for c in range(len(class_names)):
                print(' * Evaluationg the category |{}| in the video |{}|'.format(class_names[c], v))
                params.catIds = [c]
                cocoEval.evaluate()
                cocoEval.accumulate(params)
                cocoEval.summarize()
                eval_stats[v]['cat_{}'.format(c)] = cocoEval.stats.tolist()
                params.catIds = catIds
            
            params.imgIds = imgIds
            params.catIds = catIds
            
        json.dump(eval_stats, open(eval_filename_full, 'w'))

    return eval_stats, videos


def get_excel_resume(model_folder, train_params, train_loss, val_loss, eval_stats, train_diff, fps, score, iou):
    result = '{model_folder}\t{input_shape}\t{annotations}\t{anchors}\t{pretraining}\t{frozen_training:d}\t{training_time}'.format(
                model_folder = '/'.join(model_folder.split('/')[-2:]), 
                input_shape = train_params['input_shape'],
                annotations = train_params['path_annotations'],
                anchors = train_params['path_anchors'],
                pretraining = train_params['path_weights'],
                frozen_training = train_params['freeze_body'],
                training_time = train_diff
            )
    
    result += '\t{train_loss:.5f}\t{val_loss:.5f}'.format(train_loss=train_loss, val_loss=val_loss).replace('.', ',')
    result += '\t{score:.5f}\t{iou:.2f}'.format(score=score, iou=iou).replace('.', ',')
    
    result += '\t{mAP}\t{mAP50}\t{mAP75}\t{mAPS}\t{mAPM}\t{mAPL}'.format(
                mAP=eval_stats['total'][0]*100, mAP50=eval_stats['total'][1]*100, 
                mAP75=eval_stats['total'][2]*100, mAPS=eval_stats['total'][3]*100, 
                mAPM=eval_stats['total'][4]*100, mAPL=eval_stats['total'][5]*100, 
            ).replace('.', ',')
    
    result += '\t{fps:.2f}'.format(fps=fps)

    return result


def get_excel_resume_full(model_folder, train_params, train_loss, val_loss, eval_stats_train, eval_stats_val, train_diff, fps, score, iou):
    result = '{model_folder}\t{version}\t{input_shape}\t{annotations}\t{anchors}\t{pretraining}\t{frozen_training:d}\t{training_time}'.format(
                model_folder = '/'.join(model_folder.split('/')[-2:]), 
                version = train_params.get('version', ''),
                input_shape = train_params['input_shape'],
                annotations = train_params['path_annotations'],
                anchors = train_params['path_anchors'],
                pretraining = train_params['path_weights'],
                frozen_training = train_params['freeze_body'],
                training_time = train_diff
            )
    
    result += '\t{train_loss}\t{val_loss}'.format(
                train_loss=train_loss, val_loss=val_loss).replace('.', ',')

    result += '\t{mAPtrain:.5f}\t{mAP50train:.5f}'.format(
                mAPtrain = eval_stats_train['total'][0]*100,
                mAP50train = eval_stats_train['total'][1]*100,
            ).replace('.', ',')
    result += '\t{mAPval:.5f}\t{mAP50val:.5f}\t{mAP75val:.5f}'.format(
                mAPval = eval_stats_val['total'][0]*100,
                mAP50val = eval_stats_val['total'][1]*100,
                mAP75val = eval_stats_val['total'][2]*100,
            ).replace('.', ',')

    result += '\t{mAPS}\t{mAPM}\t{mAPL}'.format(
                mAPS=eval_stats_val['total'][3]*100, 
                mAPM=eval_stats_val['total'][4]*100, mAPL=eval_stats_val['total'][5]*100, 
            ).replace('.', ',')
    
    return result


def plot_prediction_resume(eval_stats, videos, class_names, by, plot):
    occurrences = {}
    for v in videos:
        v_res = {}
        for c in range(len(class_names)):
            v_res[class_names[c]] = eval_stats[v]['cat_{}'.format(c)][1]
        occurrences[v] = v_res
        
    
    occurrences = pd.DataFrame.from_dict(occurrences, orient='index')
    if by == 'video': occurrences = occurrences = occurrences.transpose()
    occurrences = occurrences.replace({-1: np.nan})
    
    meds = occurrences.mean()
    meds = meds.sort_values(ascending=False)
    
    occurrences = occurrences[meds.index]
    if plot:
        if by == 'video':
            keys = meds.index
            bars = [ eval_stats[k]['total'][1] for k in keys ]
        else:
            keys = [ 'cat_{}'.format(class_names.index(k)) for k in meds.index ]
            bars = [ eval_stats[k][1] for k in keys ]
        occurrences.boxplot(figsize=(20,7));
        plt.bar(range(1, len(bars)+1), bars, alpha=0.4)
        plt.xticks(rotation=50);
        plt.suptitle('Boxcox by: ' + by, fontsize=20);
        plt.show();
    
    return occurrences

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
    plt.legend(["{} mAP@50".format(video), "Total mAP@50", 'video-cat mAP', 'video mAP'])
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
#    plt.xticks(range(len(videos)), videos)
    plt.suptitle('Performance of the category: ' + cat, fontsize=20)
    plt.legend(["{} mAP@50".format(cat), "Total mAP@50", 'video-cat mAP', 'cat mAP'])
    plt.show()
    
    


def main(path_results, dataset_name, model_num, score, iou, num_annotation_file=1, plot=True, full=True):
    model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
    train_params = json.load(open(model_folder + 'train_params.json', 'r'))
    class_names = ktrain.get_classes(train_params['path_classes'])

    annotations_file = train_params['path_annotations'][num_annotation_file]
    preds_filename = '{}preds_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
    eval_filename = '{}stats_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
    print(preds_filename)
    print('='*80)
    
    train_diff, train_loss, val_loss = get_train_resume(model_folder)
    model, fps = predict_and_store_from_annotations(model_folder, train_params, annotations_file, preds_filename, score, iou)
    occurrences = None
    eval_stats, videos = get_full_evaluation(annotations_file, preds_filename, eval_filename, class_names, full)
    resume = get_excel_resume(model_folder, train_params, train_loss, val_loss, eval_stats, train_diff, fps, score, iou)
    plot_prediction_resume(eval_stats, videos, class_names, 'video', plot);
    occurrences = plot_prediction_resume(eval_stats, videos, class_names, 'class', plot)
    print(resume)
    return model, class_names, videos, occurrences, resume, eval_stats, train_params




path_results = '/mnt/hdd/egocentric_results/'
dataset_name = 'adl'
model_num = 6
#score = 
iou = 0.5
plot = True
full = True

annotation_files = [(0, 1), (MIN_SCORE, 0)]
#annotation_files = [(MIN_SCORE, 0)]
#annotation_files = [(0, 1)]
times = [ None for i in range(max([ af for _,af in annotation_files ])+1) ]
eval_stats_arr = [ None for i in range(max([ af for _,af in annotation_files ])+1) ]

for score, num_annotation_file in annotation_files:         # ,0

    print('='*80)
    print('dataset = {}, num_ann = {}, model = {}'.format(
            dataset_name, num_annotation_file, model_num))    
    print('='*80)
    
    times[num_annotation_file] = time.time()

    model, class_names, videos, occurrences, resume, eval_stats, train_params = main(
            path_results, dataset_name, model_num, score, iou, num_annotation_file, 
            plot, full)
    
    eval_stats_arr[num_annotation_file] = eval_stats
    
    if num_annotation_file == 1:
        pyperclip.copy(resume)

    times[num_annotation_file] = (time.time() - times[num_annotation_file])/60

eval_stats_train, eval_stats_val = eval_stats_arr
full_resume = get_excel_resume_full(resume.split('\t')[0], train_params, resume.split('\t')[7], resume.split('\t')[8], 
                             eval_stats_train, eval_stats_val, resume.split('\t')[6], 
                             resume.split('\t')[-1], score, iou)

print('='*80)
print(full_resume)
pyperclip.copy(full_resume)


#%%
    
plot_category_performance('laptop', class_names, eval_stats_train, videos)
plot_video_performance('P_12', class_names, eval_stats_train)

        
# %%
  
# Check the metric according to the score threshold
if False:
    
    #if __name__ == '__main__':
    path_results = '/mnt/hdd/egocentric_results/'
    dataset_name = 'voc'
    model_num = 0
    iou = 0.5
    plot = False
   
    
    metrics, resumes = [], []
    scores = [0.4, 0.3, 0.15, 0.1, 0.05, 0.005, 0.0005, 0.00005, 0]
    for num_annotation_file in [1]:         # ,0
        for score in scores:      # , 0.05, 0.005, 0.0005, 0.00005
            print('='*80)
            print('dataset = {}, score = {}, num_ann = {}, model = {}'.format(
                    dataset_name, score, num_annotation_file, model_num))
            print('='*80)
            model, occurrences, resume, eval_stats, train_params = main(path_results, dataset_name, 
                                   model_num, score, iou, num_annotation_file, plot)
    
            if num_annotation_file == 1:
                metrics.append(eval_stats['total'][1])
                resumes.append(resume)
    
    print('='*80)
    print('Best metric:', min(metrics))
    print('Best score:', scores[metrics.index(max(metrics))])
    pyperclip.copy(resumes[metrics.index(max(metrics))])
    plt.plot(range(len(scores)), metrics)


# %%
    
# Check the metric according to the iou threshold
if False:
    path_results = '/mnt/hdd/egocentric_results/'
    dataset_name = 'voc'
    model_num = 0
    score = 0
    plot = False
    
    metrics, resumes = [], []
    ious = [0.6, 0.5, 0.4, 0.2]
    for num_annotation_file in [1]:         # ,0
        for iou in ious:      # , 0.05, 0.005, 0.0005, 0.00005
            print('='*80)
            print('datset = {}, score = {}, iou = {}, num_ann = {}, model = {}'.format(
                    dataset_name, score, iou, num_annotation_file, model_num))
            print('='*80)
            model, occurrences, resume, eval_stats, train_params = main(
                            path_results, dataset_name, model_num, score, iou, 
                            num_annotation_file, plot)
    
            if num_annotation_file == 1:
                metrics.append(eval_stats['total'][1])
                resumes.append(resume)
    
    print('='*80)
    print('Best metric:', min(metrics))
    print('Best iou:', ious[metrics.index(max(metrics))])
    pyperclip.copy(resumes[metrics.index(max(metrics))])
    plt.plot(range(len(ious)), metrics)
    plt.xticks(range(len(ious)), ious);



# %%
    

if False:
    # %%

    path_results = '/mnt/hdd/egocentric_results/'
    dataset_name = 'default'
    model_num = 0
    score = 0.4
    iou = 0.5
    plot = True
 
    
#    path_classes = './dataset_scripts/voc/voc_classes.txt'
#    path_dataset = '/mnt/hdd/datasets/VOC/'
#    annotations_file = './dataset_scripts/voc/annotations_voc_val.txt'
    path_classes = 'base_models/coco_classes.txt'
    path_dataset = '/mnt/hdd/datasets/coco/'
    path_anchors = 'base_models/yolo_anchors.txt'
    annotations_file = './dataset_scripts/coco/annotations_coco_val.txt'
    
    
    input_shape = [416,416]
    model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
#    train_params = json.load(open(model_folder + 'train_params.json', 'r'))
    class_names = ktrain.get_classes(path_classes)

    preds_filename = '{}preds_{}_score{}_iou.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
    eval_filename = '{}stats_{}_score{}_iou.json'.format(model_folder, annotations_file.split('/')[-1][:-4], score, iou)
    train_params = {
                    'input_shape': input_shape,
                    'path_classes': path_classes,
                    'path_dataset': path_dataset,
                    'path_anchors': path_anchors 
                    }
    
    
    if os.path.isfile(preds_filename): os.remove(preds_filename)
    model, fps = predict_and_store_from_annotations(model_folder, train_params, annotations_file, preds_filename, score, iou)


# %%
 
    preds = json.load(open(preds_filename, 'r'))    
    
#    new_preds_filename = preds_filename + '_v2'
#    new_preds = preds[:]
#    for i in range(len(preds)): new_preds[i]['score'] = preds[i]['score']*1000
#    json.dump(new_preds, open(new_preds_filename, 'w'))
    
    
    scores_dt = [ d['score'] for d in preds ]
    binwidth = 0.05
    plt.hist(scores_dt, bins=np.arange(0, 1 + binwidth, binwidth), alpha = 0.8);   
  
    
    ann = annotations_file[:-4] + '_coco.json'
    
    cocoGt = COCO(ann)
    cocoDt = cocoGt.loadRes(preds_filename)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    eval_stats = cocoEval.stats.tolist()    
    
    print('mAP50:', eval_stats[1])
    
    
    
    # %%
    
#    train_diff, train_loss, val_loss = get_train_resume(model_folder)
    occurrences = None
    eval_stats, videos = get_full_evaluation(annotations_file, preds_filename, eval_filename, class_names)
#    resume = get_excel_resume(model_folder, train_params, train_loss, val_loss, eval_stats, train_diff, fps, score, iou)
    plot_prediction_resume(eval_stats, videos, class_names, 'video', plot);
    occurrences = plot_prediction_resume(eval_stats, videos, class_names, 'class', plot)


# %%
    
import json
import train_utils
import os

MIN_SCORE = 0.00005
iou = 0.5
path_results = '/mnt/hdd/egocentric_results/'
dataset_name = 'voc'
#model_num = 0

for model_num in range(0, 3, 1):
    model_folder = train_utils.get_model_path(path_results, dataset_name, model_num)
    train_params = json.load(open(model_folder + 'train_params.json', 'r'))
    annotations_file = [ tp for tp in train_params['path_annotations'] if 'train' in tp]
    
    if len(annotations_file) > 0:
        annotations_file = annotations_file[0]
    
        preds_filename = '{}preds_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], 0, iou)
        if not os.path.isfile(preds_filename): continue
        preds_train = json.load(open(preds_filename, 'r'))
    
        preds_filename_new = '{}preds_{}_score{}_iou{}.json'.format(model_folder, annotations_file.split('/')[-1][:-4], MIN_SCORE, iou)
        print(preds_filename_new)
        preds_train_new = [ pr for pr in preds_train if pr['score'] > MIN_SCORE ]
        json.dump(preds_train_new, open(preds_filename_new, 'w'))


#%%

#preds_val_file = '/mnt/hdd/egocentric_results/adl/0313_1557_model_0/preds_annotations_adl_val_416_score{}_iou0.5.json'
#preds_train_file = '/mnt/hdd/egocentric_results/adl/0313_1557_model_0/preds_annotations_adl_train_416_score{}_iou0.5.json'
#preds_val = json.load(open(preds_val_file.format(0), 'r'))
#preds_train = json.load(open(preds_train_file.format(0), 'r'))
#
## %%
#
#print('preds_train -> {} images'.format(len(list(set([ p['image_id'] for p in preds_train ])))))
#print('preds_val -> {} images'.format(len(list(set([ p['image_id'] for p in preds_val ])))))
#
#
## %%
#
#min_score = 0.000005
#preds_train_new = [ pr for pr in preds_train if pr['score'] > min_score]
#
#print('{} images'.format(len(list(set([ p['image_id'] for p in preds_train])))))
#print('{} images'.format(len(list(set([ p['image_id'] for p in preds_train_new ])))))
#
#
## %%
#
#json.dump(preds_train_new, open(preds_train_file.format(min_score), 'w'))
