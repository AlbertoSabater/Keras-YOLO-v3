from eyolo import EYOLO, detect_video_folder

import random
import os


DATASET_DIR = '/media/asabater/hdd/datasets/imagenet_vid/ILSVRC2015/Data/VID/snippets/train/ILSVRC2015_VID_train_0000/'


# %%

img_size = 608          # 320, 416, 608


model = 'yolo'

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
else:
    raise ValueError('Model not recognized')


model = EYOLO(
                model_image_size = (img_size, img_size),
                model_path = model_path,
                anchors_path = anchors_path,
                classes_path = classes_path,
            )


# %%
    
#TEST_FILE = 'ILSVRC2015_train_00040001.mp4'
#
#while True:
#    TEST_FILE = random.choice(os.listdir(DATASET_DIR))
#    print('FILE:', DATASET_DIR+TEST_FILE)
#    result = detect_video(model, DATASET_DIR+TEST_FILE, close_session=False)


#exit()


# %%

#video_folder = '/media/asabater/hdd/datasets/epic_kitchen/object_detection_images/EPIC_KITCHENS_2018/object_detection_images/train/P01/P01_01/'

dataset_folder = '/media/asabater/hdd/datasets/epic_kitchen/object_detection_images/EPIC_KITCHENS_2018/object_detection_images/train/'
video_folder = random.choice(dataset_folder)

for i in range(2):
    print(dataset_folder)
    options = [ f for f in os.listdir(dataset_folder) if f[-4:] != '.tar' ]
    dataset_folder += random.choice(options) + '/'

print('\n', dataset_folder)


# %%
    
result = detect_video_folder(model, dataset_folder)





