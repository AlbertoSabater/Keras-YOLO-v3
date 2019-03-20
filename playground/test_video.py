import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";  

from eyolo import EYOLO, detect_video_folder, predict_annotations
import train_utils

import random
import os
import json


DATASET_DIR = '/media/asabater/hdd/datasets/imagenet_vid/ILSVRC2015/Data/VID/snippets/train/ILSVRC2015_VID_train_0000/'

# %%

img_size = 416          # 320, 416, 608


model = 'adl'

if model == 'tiny':
    anchors_path = 'base_models/tiny_yolo_anchors.txt'
    model_path = 'base_models/yolo_tiny.h5'
    classes_path = 'base_models/coco_classes.txt'
elif model == 'yolo':
    model_path = 'base_models/yolo.h5'
    anchors_path = 'base_models/yolo_anchors.txt'
    classes_path = 'base_models/coco_classes.txt'
elif model == 'spp':
    model_path = 'base_models/yolov3-spp.h5'
    anchors_path = 'base_models/yolo_anchors.txt'
    classes_path = 'base_models/coco_classes.txt'
#elif model == '9000':
#    model_path = 'model_data/yolo9000.h5'
#    anchors_path = 'model_data/yolo9000_anchors.txt'
elif model == 'openimages':
    anchors_path = 'base_models/yolo_anchors.txt'
    model_path = 'base_models/yolov3-openimages.h5'
    classes_path = 'base_models/openimages_classes.txt'
elif model == 'adl':
    version = '_v2_27'
#    anchors_path = 'base_models/yolo_anchors.txt'
    model_num = 13
    model_folder = train_utils.get_model_path('/mnt/hdd/egocentric_results/', 'adl', model_num)
    train_params = json.load(open(model_folder + 'train_params.json', 'r'))
    model_path = train_utils.get_best_weights(model_folder)
    classes_path = train_params['path_classes']
    anchors_path = train_params['path_anchors']
	
else:
    raise ValueError('Model not recognized')


print('Loading:', model_path)
print('='*80)
model = EYOLO(
                model_image_size = (img_size, img_size),
                model_path = model_path,
                anchors_path = anchors_path,
                classes_path = classes_path,
                score = 0.3,
                iou = 0.45
#                gpu_num = 2
            )


# %%

#video_folder = '/media/asabater/hdd/datasets/epic_kitchen/object_detection_images/EPIC_KITCHENS_2018/object_detection_images/train/P01/P01_01/'

#dataset_folder = '/mnt/hdd/datasets/epic_kitchen/object_detection_images/EPIC_KITCHENS_2018/object_detection_images/train/'
dataset_folder = '/home/asabater/projects/ADL_dataset/ADL_frames_416/'
video_folder = random.choice(dataset_folder)

for i in range(1):
    print(dataset_folder)
    options = [ f for f in os.listdir(dataset_folder) if f[-4:] != '.tar' ]
    dataset_folder += random.choice(options) + '/'

print('\n', dataset_folder)


# %%
 
wk = 250  
#annotations = './dataset_scripts/adl/annotations_adl_val_416.txt'
annotations = train_params['path_annotations'][1]
   
#result = detect_video_folder(model, daqtaset_folder, wk=500)
predict_annotations(model, annotations, wk)


# %%






