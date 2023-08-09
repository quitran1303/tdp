import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import keras
import math

import utils_origin as utils
import model_origin as modellib
import visualize
from model_origin import log
import json
from pycocotools import mask as maskUtils
import time
from config import Config

# %matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Path to Shapes trained weights
SHAPES_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes.h5")

CITY_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_cityscape_data_0069-copy.h5")

class CityscapeConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscape_data"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes
    NUM_CLASSES = 35

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
#     IMAGE_MIN_DIM = 1024
#     IMAGE_MAX_DIM = 2048
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200  # default value

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100  # small number on small dataset, better smaller than 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5  # bigger number improves validation stats accuracy but slows down
    


class InferenceConfig(CityscapeConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
config = InferenceConfig()
    
# config = CityscapeConfig()
config.display()

CITY_DIR = "../cityscapes-dataset/"

class CityscapesDataset(utils.Dataset):    
    def load_cityscapes(self, dataset_dir, subset):
        """Load a subset of the cityscapes dataset.
        dataset_dir: The root directory of the cityscapes dataset.
        subset: What to load (train, val, test)
        """
        self.class_labels = {
        'unlabeled':0,
        'ego vehicle':1,        
        'rectification border':2,
        'out of roi':3,           
        'static':4,               
        'dynamic':5,              
        'ground':6,               
        'road':7,                 
        'sidewalk':8,             
        'parking':9,              
        'rail track':10,           
        'building':11,            
        'wall':12,                 
        'fence':13,                
        'guard rail':14,           
        'bridge':15,               
        'tunnel':16,               
        'pole':17,                 
        'polegroup':18,            
        'traffic light':19,        
        'traffic sign':20,         
        'vegetation':21,           
        'terrain':22,              
        'sky':23,                  
        'person':24,               
        'rider':25,                
        'car':26,                  
        'truck':27,                
        'bus':28,                  
        'caravan':29,              
        'trailer':30,              
        'train':31,                
        'motorcycle':32,           
        'bicycle':33,              
        'license plate':34}
        
        annotation_dir = dataset_dir + 'gtFine_trainvaltest/' + subset + '_all.json'
        self.image_info = json.load(open(annotation_dir, 'r'))
        
        # Add classes
        for i in range(len(self.class_labels)):
            self.add_class("cityscape", i, list(self.class_labels.keys())[i])
    
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        pass
    
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # TODO: build dict **self.image_info** in this form
        # self.image_info.keys() = ['objects', 'imgWidth', 'imgHeight']
        # objects is a list which contains label and polygon (same as annotations form below)
        # imgHeight and imgWidth are numbers (usually 1024, 2048)
        annotations = self.image_info[image_id]["objects"]
        # annotations form: [{'label': label, 'polygon': [[x1,y1], [x2,y2] ...]}, ...]
        height = self.image_info[image_id]['imgHeight']
        width = self.image_info[image_id]['imgWidth']
        instance_masks = []
        class_ids = []
        for ann in annotations:
            m = self.annToMask(ann, height, width)
            
            label_tmp = ann['label']
            if ( not label_tmp in list(self.class_labels.keys()) ) and label_tmp.endswith('group'):
                label_tmp = label_tmp[:-len('group')]
            
            class_id = self.class_labels[label_tmp]
            instance_masks.append(m)
            class_ids.append(class_id)
            
        mask = np.stack(instance_masks, axis=2)
        class_ids = np.array(class_ids)
        
        return mask, class_ids
        
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentaion']
        # convert segm from [[x1, y1], [x2, y2]...] to [[x1, y1, x2, y2, ...]] 
        segm = [np.ravel(segm)]
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentaion']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

class CityscapeConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    RPNNMS_THRESHHOLD = 0.2
    # Give the configuration a recognizable name
    NAME = "cityscape_data"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes
    NUM_CLASSES = 35

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 2048
#     IMAGE_MIN_DIM = 512
#     IMAGE_MAX_DIM = 1024

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200  # default value

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50  # small number on small dataset, better smaller than 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5  # bigger number improves validation stats accuracy but slows down
    
config = CityscapeConfig()
config.display()

model_paths = ["logs/mask_rcnn_cityscape_data_all_add_layer.h5","logs/mask_rcnn_cityscape_data_all.h5"]
for model_path in model_paths:
# Create model in inference mode
    CITY_MODEL_PATH = os.path.join(ROOT_DIR, model_path)

    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    # Set weights file path
    if config.NAME == "shapes":
        weights_path = SHAPES_MODEL_PATH
    elif config.NAME == "coco":
        weights_path = COCO_MODEL_PATH
    elif config.NAME == 'cityscape_data':
        weights_path = CITY_MODEL_PATH
    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    dataset_val = CityscapesDataset()
    dataset_val.load_cityscapes(CITY_DIR, 'val')
    dataset_val.prepare()


    dataset_num = 50# set this variable to change the validation dataset
    image_ids = np.random.choice(dataset_val.image_ids,50)
    APs = []
    for image_id in image_ids:
        print(image_id)
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        #print(r)
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id,
                             r["rois"], r["class_ids"], r["scores"])
        if math.isnan(AP):
            continue
        else:
            APs.append(AP)

    print("mAP: ", np.mean(APs),"valid length: ",len(APs))

    model_type = model_path[28:-3]
    np.savetxt('new'+model_type+'.csv', APs, delimiter = ',')  
