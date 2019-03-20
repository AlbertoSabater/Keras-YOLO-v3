#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:20:48 2019

@author: asabater
"""

# https://github.com/zxwxz/keras-yolo3/blob/51d55bbe4c680a7f8ae2bf995dd4929809bb77dc/train.py

#from multiprocessing.dummy import Pool as ThreadPool
import threading
import numpy as np
import itertools
import time
from multiprocessing.dummy import Pool as ThreadPool

import sys
sys.path.append('keras_yolo3/')

from yolo3.model import preprocess_true_boxes
from keras_yolo3.yolo3.utils import get_random_data


threadpool = ThreadPool(32)

class DataFactory(threading.Thread):
    def __init__(self, annotation_lines, batch_size, input_shape, anchors, num_classes):
        super().__init__()
        self.data_cond = threading.Condition()
        self.proc_cond = threading.Condition()
        self.output_datas = []
        self.annotation_lines = annotation_lines
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.anchors = anchors
        self.num_classes = num_classes
        assert len(self.annotation_lines) > 0
        assert self.batch_size > 0

    def run(self):
        while True:
            cur_data_size = len(self.output_datas)
            if cur_data_size < 5:
                for i in range(cur_data_size, 10):
                    output_data = data_generator_factory(self.annotation_lines, self.batch_size, self.input_shape, self.anchors, self.num_classes)
                    time.sleep(0.01)
                    self.data_cond.acquire()
                    self.output_datas.append(output_data)
                    self.data_cond.release()
            else:
                self.proc_cond.acquire()
                self.proc_cond.wait()
                self.proc_cond.release()

    def get_data(self):
        while True:
            Breakout=False
            self.data_cond.acquire()
            if len(self.output_datas) > 0:
                Breakout = True
                output_data = self.output_datas.pop()
                if len(self.output_datas) < 3:
                    self.proc_cond.acquire()
                    self.proc_cond.notify()
                    self.proc_cond.release()
            else:
                self.data_cond.wait()
            self.data_cond.release()
            if Breakout:
                return output_data


def data_generator_factory(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        if (i+batch_size > n):
            np.random.shuffle(annotation_lines)
            i = 0
        output = threadpool.starmap(get_random_data, zip(annotation_lines[i:i+batch_size], itertools.repeat(input_shape, batch_size)))
        image_data = list(zip(*output))[0]
        box_data = list(zip(*output))[1]
        i = i+batch_size
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper_factory(data_factory):
    return data_factory.get_data()






def data_generator_parallel(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        if (i+batch_size > n):
            np.random.shuffle(annotation_lines)
            i = 0
        output = threadpool.starmap(get_random_data, zip(annotation_lines[i:i+batch_size], itertools.repeat(input_shape, batch_size)))
        image_data = list(zip(*output))[0]
        box_data = list(zip(*output))[1]
        i = i+batch_size
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper_parallel(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator_parallel(annotation_lines, batch_size, input_shape, anchors, num_classes)





def data_generator_default(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=random)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper_default(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator_default(annotation_lines, batch_size, input_shape, anchors, num_classes, random)

