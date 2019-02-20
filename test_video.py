import sys
sys.path.append('keras_yolo3/')

#from yolo import YOLO, detect_video
from eyolo import EYOLO, detect_video


DATASET_DIR = '/media/asabater/hdd/datasets/imagenet_vid/ILSVRC2015/Data/VID/snippets/train/ILSVRC2015_VID_train_0000/'


# %%

img_size = 416          # 320, 416, 608


model = 'tiny'

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

TEST_FILE = 'ILSVRC2015_train_00040001.mp4'

print('FILE:', DATASET_DIR+TEST_FILE)
detect_video(model, DATASET_DIR+TEST_FILE, close_session=False)


#exit()


# %%

#from yolo import YOLO
#
#model = YOLO(
#                model_image_size=(img_size, img_size),
#                model_path = 'model_data/yolo9000.h5',
#                anchors_path = 'model_data/yolo9000_anchors.txt',
#                classes_path = 'model_data/9k_names.txt',
#            )

