#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 11:57:25 2019

@author: asabater
"""

import sys
sys.path.append('../keras_yolo3/')

from kmeans import YOLO_Kmeans
import numpy as np
from tqdm import tqdm



class EYOLO_Kmeans(YOLO_Kmeans):

    def __init__(self, cluster_number, filename):
        self.cluster_number = cluster_number
        self.filename = filename
        
    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split()
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result
    
    def get_best_anchors(self):
        all_boxes = self.txt2boxes()
        num_steps = 20
        best_iou, best_result = 0, None
        for i in tqdm(range(num_steps), total=num_steps, file=sys.stdout):
            result = self.kmeans(all_boxes, k=self.cluster_number)
            result = result[np.lexsort(result.T[0, None])]
            iou = self.avg_iou(all_boxes, result) * 100
            if iou > best_iou:
                best_iou = iou
                best_result = result
                
#        print("K anchors:\n {}".format(best_result))
        print("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, best_result) * 100))
        return best_result        
        
    def result2txt(self, data, output_filename):
        f = open(output_filename, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()
        
        

if __name__ == "__main__":

    cluster_number = 9
    suffix = '_v3_8_pr416'
    filename = "./adl/annotations_adl_train{}.txt".format(suffix)
    output_filename = './adl/anchors_adl{}.txt'.format(suffix)
    kmeans = EYOLO_Kmeans(cluster_number, filename)
    anchors = kmeans.get_best_anchors()
    kmeans.result2txt(anchors, output_filename)
    print(output_filename, 'generated')



