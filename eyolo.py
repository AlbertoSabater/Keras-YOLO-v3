#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:59:56 2019

@author: asabater
"""

import sys
sys.path.append('keras_yolo3/')

from timeit import default_timer as timer
from PIL import Image, ImageFont, ImageDraw
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.utils import multi_gpu_model

from yolo3.utils import letterbox_image
from yolo import YOLO

from emodel import yolo_body, r_yolo_body
from yolo3.model import tiny_yolo_body, yolo_eval

import cv2
import os
import random
import colorsys


class EYOLO(YOLO):
    
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            if self.td_len is not None and self.mode is not None:
                self.yolo_model = r_yolo_body(Input(shape=(self.td_len, None, None, 3)), 
                                              num_anchors//3, num_classes, self.td_len, self.mode)
            elif is_tiny_version:
                self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
            else:
                self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
#            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
#                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes
    
    
    def get_prediction(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            if type(image) is list:
                if len(image) > 1:
                    img_size = image[0].size
                    boxed_image = np.stack([ letterbox_image(image, tuple(reversed(self.model_image_size))) for image in image  ])
                else:
                    img_size = image[0].size
                    boxed_image = letterbox_image(image[0], tuple(reversed(self.model_image_size)))
            else:
                img_size = image.size
                boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

#        print('nn_input', image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [img_size[1], img_size[0]],
                K.learning_phase(): 0
            })
	
        return out_boxes, out_scores, out_classes
    
    
#    def detect_image(self, image):
#        start = timer()
#
#        if self.model_image_size != (None, None):
#            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
#            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
#            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
#        else:
#            new_image_size = (image.width - (image.width % 32),
#                              image.height - (image.height % 32))
#            boxed_image = letterbox_image(image, new_image_size)
#        image_data = np.array(boxed_image, dtype='float32')
#
##        print('nn_input', image_data.shape)
#        image_data /= 255.
#        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
#
#        out_boxes, out_scores, out_classes = self.sess.run(
#            [self.boxes, self.scores, self.classes],
#            feed_dict={
#                self.yolo_model.input: image_data,
#                self.input_image_shape: [image.size[1], image.size[0]],
#                K.learning_phase(): 0
#            })
#
##        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
#
#        
##        for i, c in reversed(list(enumerate(out_classes))):
##            predicted_class = self.class_names[c]
##            box = out_boxes[i]
##            score = out_scores[i]
##
##            label = '{} {:.2f}'.format(predicted_class, score)
##            
##            image = self.print_box(image, box, label, self.colors[c])
#
#        end = timer()
##        print(end - start)
##        return image
#        
#        return image, out_boxes, out_scores, out_classes
    
    def print_boxes(self, image, boxes, classes, scores=None, color=None):
        for i, c in reversed(list(enumerate(classes))):
            predicted_class = self.class_names[c]
            box = boxes[i]
#            score = '' if scores is None else scores[i]
            color_c = self.colors[c] if color is None else color
            
            if scores is None:
                label = '{}'.format(predicted_class)
            else:
                label = '{} {:.2f}'.format(predicted_class, scores[i])
            
            image = self.print_box(image, box, label, color_c)        
            
        return image
    
    
    def print_box(self, image, box, label, color):
        
        font = ImageFont.truetype(font='keras_yolo3/font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
#            print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range((image.size[0] + image.size[1]) // 300):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=color)
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=color)
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
        
        return image


def detect_video(yolo, video_path, output_path="", close_session=True):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        
        if frame is None: 
#            cv2.waitKey(1)
#            cv2.destroyAllWindows()
#            cv2.waitKey(1)
            break
    
#        print(1, type(frame), frame.shape)
        image = Image.fromarray(frame)
#        print(2, type(image), image.size)
        image = yolo.detect_image(image)
#        print(3, type(image), image.size)
        result = np.asarray(image)
#        print(4, type(result),result.shape)
        
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    if close_session: yolo.close_session()
    
    

def detect_video_folder(yolo, video_folder, wk=1):
    frames = os.listdir(video_folder)
    prev_time = timer()

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    
    for fr in frames:
        
#        image = cv2.imread(video_folder + fr)
#        image = Image.fromarray(image)
        image = Image.open(video_folder + fr)
        image, boxes, scores, classes = yolo.detect_image(image)
        image = yolo.print_boxes(image, boxes, classes, scores)
        result = np.asarray(image)
        
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(wk) & 0xFF == ord('q'):
            break

    return result
    

def predict_annotations(model, annotations, path_base, wk):
    with open(annotations, 'r') as f: annotations = f.read().splitlines()
    annotations = annotations[random.randint(0, len(annotations)):]
    
    prev_time = timer()
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    
    print(annotations[1])
    
    for l in annotations:
        
        l = l.split()
        img = l[0]
        boxes = [ [int(b) for b in bb.split(',') ] for bb in l[1:] ]
        classes = [ bb[-1] for bb in boxes ]
        boxes = [ bb[:-1] for bb in boxes ]
        boxes = [ [bb[1],bb[0],bb[3],bb[2]] for bb in boxes ]
        
#        image = cv2.imread(path_base + img)
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        image = Image.fromarray(image)    
        images = [ Image.open(path_base + img) for img in img.split(',') ]
        image = model.print_boxes(images[len(images)//2], boxes, classes, color=(0,255,0))

        boxes, scores, classes = model.get_prediction(images)
        image = model.print_boxes(image, boxes, classes, scores, color=(0,0,255))
    
        result = np.asarray(image)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        
        cv2.putText(result, text=fps + " | {}".format(img.split('/')[-2]), org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(wk) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    
    