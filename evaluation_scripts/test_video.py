import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "";  

from eyolo import EYOLO, detect_video_folder, predict_annotations
import train_utils

import random
import os
import json



img_size = 416          # 320, 416, 608
model_image_size = (img_size, img_size)

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
    
elif model == 'voc':
#    path_dataset = '/mnt/hdd/datasets/adl_dataset/ADL_frames/'
    input_shape = (416,416)
#    path_dataset = '/mnt/hdd/datasets/VOC/'
#    path_annotations = ['./dataset_scripts/voc/annotations_voc_train.txt',
#                        './dataset_scripts/voc/annotations_voc_val.txt']
##    path_annotations = ['/home/asabater/projects/ADL_dataset/annotations_adl_train.txt',
##                        '/home/asabater/projects/ADL_dataset/annotations_adl_val.txt']
#    path_classes = './dataset_scripts/voc/voc_classes.txt'
    
#    anchors_path = 'base_models/yolo_anchors.txt'
    model_num = 0
    model_folder = train_utils.get_model_path('/mnt/hdd/egocentric_results/', 'voc', model_num)
    train_params = json.load(open(model_folder + 'train_params.json', 'r'))
#    model_path = model_folder + 'weights/trained_weights_stage_1.h5'
    classes_path = train_params['path_classes']
    anchors_path = train_params['path_anchors']
    model_image_size = train_params['input_shape']
    path_base = '/mnt/hdd/datasets/VOC/'
    
elif model == 'adl':
#    anchors_path = 'base_models/yolo_anchors.txt'
    model_num = 16
    model_folder = train_utils.get_model_path('/mnt/hdd/egocentric_results/', 'adl', model_num)
    train_params = json.load(open(model_folder + 'train_params.json', 'r'))
#    model_path = model_folder + 'weights/trained_weights_stage_1.h5'
    classes_path = train_params['path_classes']
    anchors_path = train_params['path_anchors']
    model_image_size = train_params['input_shape']
    path_base = '/home/asabater/projects/ADL_dataset/'
    
#elif model == 'coco':
#    model_folder = train_utils.get_model_path('/mnt/hdd/egocentric_results/', 'default', 0)
#    model_path = model_folder + 'weights/ep070-loss15.91974-val_loss14.88787.h5'
#    classes_path = 'base_models/coco_classes.txt'
#    path_base = '/mnt/hdd/datasets/coco/'
#    anchors_path = 'base_models/yolo_anchors.txt'
#    model_image_size = [416, 416]
	    
elif model == 'kitchen':
#    version = '_v3_35'
    model_num = 1
    model_folder = train_utils.get_model_path('/mnt/hdd/egocentric_results/', 'kitchen', model_num)
    train_params = json.load(open(model_folder + 'train_params.json', 'r'))
#    classes_path = './dataset_scripts/kitchen/kitchen_classes{}.txt'.format(version)
    classes_path = train_params['path_classes']
    path_base = ''
#    anchors_path = 'base_models/yolo_anchors.txt'
    anchors_path = train_params['path_anchors']
    model_image_size = [416, 416]
    
else:
    raise ValueError('Model not recognized')



if model_path is None:
	model_path = train_utils.get_best_weights(model_folder)


print('Loading:', model_path)
print('='*80)
model = EYOLO(
                model_image_size = (img_size, img_size),
                model_path = model_path,
                anchors_path = anchors_path,
                classes_path = classes_path,
                score = 0.1,
                iou = 0.5,
#                gpu_num = 2
                td_len = train_params['td_len'],
                mode = train_params['mode'],
				spp = train_params.get('spp', False)
            )


# %%
 
wk = 1000 // 4
#annotations = './dataset_scripts/adl/annotations_adl_val_416.txt'
annotations = train_params['path_annotations'][1]
#annotations = './dataset_scripts/coco/annotations_coco_val.txt'

#result = detect_video_folder(model, daqtaset_folder, wk=500)
predict_annotations(model, annotations, path_base, wk)


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






