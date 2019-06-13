# Keras YOLO v3

- [x] Spatial Pyramid Pooling (SPP)
- [x] Multi-scale training
- [x] OpenCv data augmentation (increases learning speed)
- [x] mAP Evaluation (complete, per cateogry and per subdataset (if exists))
- [x] Loss components (xy, wh, class, confidence_obj, confidence_noobj) weighting
- [x] Loss components logging on TensorBoard
- [ ] mAP TensorBoard logging metric
- [ ] Bounding box post-processing
- [ ] Recurrent YOLO model

## Introduction

A Keras implementation of YOLOv3 (Tensorflow backend) inspired by [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).

**Important.** To use this repo you must make sure that [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3) project has been properly downloaded.
```
git clone https://github.com/AlbertoSabater/Egocentric-object-detection
cd Egocentric-object-detection
git submodule update --init --recursive
```
---

## Download pretrained models and convert to Keras

1. Download weights and .cfg file from [YOLO website](http://pjreddie.com/darknet/yolo/) and store them in `./weights`.
2. Convert the weights to a Keras model and store it in `./base_models`.
```
# python keras_yolo3/convert.py cfg_file weights_file output_file

# Darknet-53
wget -O weights/darknet53.weights https://pjreddie.com/media/files/darknet53.conv.74
wget -O weights/darknet53.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/darknet53.cfg
python keras_yolo3/convert.py weights/darknet53.cfg weights/darknet53.weights base_models/darknet53.h5

# YOLOv3
wget -O weights/yolo.weights https://pjreddie.com/media/files/yolov3.weights
wget -O weights/yolo.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
python keras_yolo3/convert.py weights/yolo.cfg weights/yolo.weights base_models/yolo.h5

# YOLO-SPP
wget -O weights/yolo-spp.weights https://pjreddie.com/media/files/yolov3-spp.weights
wget -O weights/yolo-spp.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-spp.cfg
python keras_yolo3/convert.py weights/yolo-spp.cfg weights/yolo-spp.weights base_models/yolo-spp.h5

# YOLO-tiny
wget -O weights/yolo-tiny.weights https://pjreddie.com/media/files/yolov3-tiny.weights
wget -O weights/yolo-tiny.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
python keras_yolo3/convert.py weights/yolo-tiny.cfg weights/yolo-tiny.weights base_models/yolo-tiny.h5
```


## Testing

Use `eyolo_prediction.py` to predict the bounding boxes of a image or a video using a trained model.

```
usage: eyolo_prediction.py [-h] --model MODEL --anchors ANCHORS --classes
                           CLASSES --input INPUT [--image] [--spp]

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file
  --anchors ANCHORS  path to anchor definitions
  --classes CLASSES  path to class definitions
  --input INPUT      Video/image input path
  --image            Image detection mode
  --spp              use this option if the model uses SPP
```

Image prediction sample:
```
python eyolo_prediction.py --model base_models/yolo.h5 --anchors base_models/yolo_anchors.txt --classes base_models/coco_classes.txt --image --input test_images/image_1.jpg
```
Video prediction sample:
```
python eyolo_prediction.py --model base_models/yolo.h5 --anchors base_models/yolo_anchors.txt --classes base_models/coco_classes.txt --input test_images/video_1.mp4
```



## Training

1. Generate valid annotations files for training and validation:  
    One row for each image  
    Row format: `image_file_path box1 box2 ... boxN`  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space)  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```
2. Generate valid classe filenames. One class name in each line of a .txt file.
3. Convert annotations to coco style. It will be needed for the evaluation. Use `dataset_scripts/annotations_to_coco.py`:
```
usage: annotations_to_coco.py [-h] path_annotations path_classes

positional arguments:
  path_annotations  path to the annotations file to convert
  path_classes      path to the dataset classes filename

optional arguments:
  -h, --help        show this help message and exit
```
4. Run `train.py`. Check the script usage with `python train.py --help`.
```
usage: train.py [-h] [--path_dataset PATH_DATASET]
                [--frozen_epochs FROZEN_EPOCHS] [--input_shape INPUT_SHAPE]
                [--spp] [--multi_scale]
                path_results dataset_name path_classes path_anchors
                path_annotations_train path_annotations_val path_weights
                freeze_body

Script to train a YOLO v3 model. Note that the classes especified in
path_classes must be the same used in path_annotations_train and
path_annotations_val. If using --spp make sure to load the weights that share
the same NN architecture.

positional arguments:
  path_results          path where the store the training results
  dataset_name          subfolder where to store the training results
  path_classes          dataset classes file
  path_anchors          anchors file
  path_annotations_train
                        train annotations file
  path_annotations_val  validation annotations file
  path_weights          path to pretrained weights
  freeze_body           0 to not freezing 1 to freeze backbone 2 to freeze all
                        the model

optional arguments:
  -h, --help            show this help message and exit
  --path_dataset PATH_DATASET
                        path to each training image if not specified in
                        annotations file
  --frozen_epochs FROZEN_EPOCHS
                        number of frozen training epochs. Default 15
  --input_shape INPUT_SHAPE
                        training/validation input image shape. Must be a
                        multiple of 32. Default 416
  --spp                 to use Spatial Pyramid Pooling
  --multi_scale         to use multi-scale training
```

By default, the Neural Network uses pretrained weights, trains 15 epochs with frozen layers and unfreeze the model and trains until convergence.

If training with SPP weights, use `--spp` flag.

Weight each loss component by modifying manually `loss_perc` in `train.py`.

Under the folder and subfolder specified, weights (best and last_weights), model architecture and train_params will be stored.

During training, the loss, loss components (xy, wh, class, confidence_obj, confidence_noobj) and lr are logged with TensorBoard. Launch TensorBoard with: `tensorboard --logdir [path_results]/[dataset_name]/`

And the end of training mAP evaluation is performed with the best weighs learned and the last ones. Check next section.

To train a tiny YOLO model, use `base_models/tiny_yolo_anchors.txt` as anchors.


## Evaluation

mAP is calculated for all the training and test dataset, for each class and for each subdataset (if exists). The reference metric is mAP@50. As a result a json file is stored in model results folder with all the stats.

If `best_weights == True` evaluation is performed with the weights that minimizes the val loss during training.  
If `best_weigths == False` evaluation is performed with the last weights learned.  
If `best_weights is None`, training evaluation is performed with the kind of weights that minimizes the validation dataset mAP.

---

## Testing environment

* Python 3..6.8
* Keras 2.2.4
* tensorflow-gpu 1.13.1
