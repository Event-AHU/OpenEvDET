"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()
import os.path as osp
import os 
import numpy as np
import json

from torchvision import datapoints
from pycocotools import mask as coco_mask
from src.core import register
from scipy.io import loadmat


__all__ = ['CocoDetection', 'EvDetection', 'SketchDetection']

@register
class SketchDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder, graph_folder, ann_file, transforms, return_masks, remap_mscoco_category=False, is_video_voxel=False):
        super(SketchDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.graph_folder = graph_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.is_video_voxel = is_video_voxel

    def __getitem__(self, idx):
        img, target = super(SketchDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=img.size[::-1]) # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        size = torch.tensor(img.shape[1:])
        target['size'] = size

        graph = self.getgraph(image_id)

        return img, graph, target
    
    def getgraph(self, idx):
        graph_path = self.coco.loadImgs(idx)[0]['graph']
        graph_path = osp.join(self.graph_folder, graph_path)
        f = open(graph_path, 'r')
        graph_data = json.load(f)
        return graph_data


    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s 


@register
class EvDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder, voxel_folder, ann_file, transforms, return_masks, remap_mscoco_category=False, is_video_voxel=False):
        super(EvDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.voxel_folder = voxel_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category
        self.is_video_voxel = is_video_voxel

    def __getitem__(self, idx):
        img, target = super(EvDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=img.size[::-1]) # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        size = torch.tensor(img.shape[1:])
        target['size'] = size

        voxel = self.getvoxel(idx)

        return img, voxel, target
    
    def getvoxel(self, idx):
        voxel_path = self.coco.loadImgs(idx)[0]['voxel_name']
        if self.is_video_voxel:
            voxel_coor = []
            voxel_features = []
            voxel = {}
            voxel_path = osp.dirname(voxel_path)
            voxel_name = ['frame0000.mat', 'frame0001.mat', 'frame0002.mat', 'frame0003.mat', 'frame0004.mat']
            for item in voxel_name:
                voxel_single = loadmat(osp.join(self.voxel_folder, voxel_path, item))
                voxel_coor.append(voxel_single['coor'])
                voxel_features.append(voxel_single['features'])
            voxel['coor'] = torch.tensor(np.concatenate(voxel_coor, axis=0), dtype=torch.float32)
            voxel['features'] = torch.tensor(np.concatenate(voxel_features, axis=0), dtype=torch.float32)
            
        else:
            voxel = {}
            voxel_single = loadmat(osp.join(self.voxel_folder, voxel_path))
            voxel['coor'] = torch.tensor(voxel_single['coor'][::4], dtype=torch.float32)
            voxel['features'] = torch.tensor(voxel_single['features'][::4], dtype=torch.float32)
        return voxel

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s 




@register
class CocoDetection(torchvision.datasets.CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=img.size[::-1]) # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        size = torch.tensor(img.shape[1:])
        target['size'] = size
        return img, target

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'

        return s 


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, remap_mscoco_category=False):
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [mscoco_category2label[obj["category_id"]] for obj in anno]
        else:
            classes = [obj["category_id"] for obj in anno]
            
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(w), int(h)])
        target["size"] = torch.as_tensor([int(w), int(h)])
    
        return image, target

#coco
# mscoco_category2name = {
#     1: 'person',
#     2: 'bicycle',
#     3: 'car',
#     4: 'motorcycle',
#     5: 'airplane',
#     6: 'bus',
#     7: 'train',
#     8: 'truck',
#     9: 'boat',
#     10: 'traffic light',
#     11: 'fire hydrant',
#     13: 'stop sign',
#     14: 'parking meter',
#     15: 'bench',
#     16: 'bird',
#     17: 'cat',
#     18: 'dog',
#     19: 'horse',
#     20: 'sheep',
#     21: 'cow',
#     22: 'elephant',
#     23: 'bear',
#     24: 'zebra',
#     25: 'giraffe',
#     27: 'backpack',
#     28: 'umbrella',
#     31: 'handbag',
#     32: 'tie',
#     33: 'suitcase',
#     34: 'frisbee',
#     35: 'skis',
#     36: 'snowboard',
#     37: 'sports ball',
#     38: 'kite',
#     39: 'baseball bat',
#     40: 'baseball glove',
#     41: 'skateboard',
#     42: 'surfboard',
#     43: 'tennis racket',
#     44: 'bottle',
#     46: 'wine glass',
#     47: 'cup',
#     48: 'fork',
#     49: 'knife',
#     50: 'spoon',
#     51: 'bowl',
#     52: 'banana',
#     53: 'apple',
#     54: 'sandwich',
#     55: 'orange',
#     56: 'broccoli',
#     57: 'carrot',
#     58: 'hot dog',
#     59: 'pizza',
#     60: 'donut',
#     61: 'cake',
#     62: 'chair',
#     63: 'couch',
#     64: 'potted plant',
#     65: 'bed',
#     67: 'dining table',
#     70: 'toilet',
#     72: 'tv',
#     73: 'laptop',
#     74: 'mouse',
#     75: 'remote',
#     76: 'keyboard',
#     77: 'cell phone',
#     78: 'microwave',
#     79: 'oven',
#     80: 'toaster',
#     81: 'sink',
#     82: 'refrigerator',
#     84: 'book',
#     85: 'clock',
#     86: 'vase',
#     87: 'scissors',
#     88: 'teddy bear',
#     89: 'hair drier',
#     90: 'toothbrush'
# }

# gen1
# mscoco_category2name = {
#     0: 'person',
#     1: 'car',
# }

#EvDET200k
mscoco_category2name = {
    1: 'people',
    2: 'car',
    3: 'bicycle',
    4: 'electric bicycle',
    5: 'basketball',
    6: 'ping_pong',
    7: 'goose',
    8: 'cat',
    9: 'bird',
    10: 'UAV'
}

#DSEC
# mscoco_category2name = {
#     0: 'pedestrian',
#     1: 'rider',
#     2: 'car',
#     3: 'bus',
#     4: 'truck',
#     5: 'bicycle',
#     6: 'motorcycle',
#     7: 'train'
# }

# ncaltech
# mscoco_category2name = {
#     0: 'BACKGROUND_Google',
#     1: 'Faces_easy',
#     2: 'Leopards',
#     3: 'Motorbikes',
#     4: 'accordion',
#     5: 'airplanes',
#     6: 'anchor',
#     7: 'ant',
#     8: 'barrel',
#     9: 'bass',
#     10: 'beaver',
#     11: 'binocular',
#     12: 'bonsai',
#     13: 'brain',
#     14: 'brontosaurus',
#     15: 'buddha',
#     16: 'butterfly',
#     17: 'camera',
#     18: 'cannon',
#     19: 'car_side',
#     20: 'ceiling_fan',
#     21: 'cellphone',
#     22: 'chair',
#     23: 'chandelier',
#     24: 'cougar_body',
#     25: 'cougar_face',
#     26: 'crab',
#     27: 'crayfish',
#     28: 'crocodile',
#     29: 'crocodile_head',
#     30: 'cup',
#     31: 'dalmatian',
#     32: 'dollar_bill',
#     33: 'dolphin',
#     34: 'dragonfly',
#     35: 'electric_guitar',
#     36: 'elephant',
#     37: 'emu',
#     38: 'euphonium',
#     39: 'ewer',
#     40: 'ferry',
#     41: 'flamingo',
#     42: 'flamingo_head',
#     43: 'garfield',
#     44: 'gerenuk',
#     45: 'gramophone',
#     46: 'grand_piano',
#     47: 'hawksbill',
#     48: 'headphone',
#     49: 'hedgehog',
#     50: 'helicopter',
#     51: 'ibis',
#     52: 'inline_skate',
#     53: 'joshua_tree',
#     54: 'kangaroo',
#     55: 'ketch',
#     56: 'lamp',
#     57: 'laptop',
#     58: 'llama',
#     59: 'lobster',
#     60: 'lotus',
#     61: 'mandolin',
#     62: 'mayfly',
#     63: 'menorah',
#     64: 'metronome',
#     65: 'minaret',
#     66: 'nautilus',
#     67: 'octopus',
#     68: 'okapi',
#     69: 'pagoda',
#     70: 'panda',
#     71: 'pigeon',
#     72: 'pizza',
#     73: 'platypus',
#     74: 'pyramid',
#     75: 'revolver',
#     76: 'rhino',
#     77: 'rooster',
#     78: 'saxophone',
#     79: 'schooner',
#     80: 'scissors',
#     81: 'scorpion',
#     82: 'sea_horse',
#     83: 'snoopy',
#     84: 'soccer_ball',
#     85: 'stapler',
#     86: 'starfish',
#     87: 'stegosaurus',
#     88: 'stop_sign',
#     89: 'strawberry',
#     90: 'sunflower',
#     91: 'tick',
#     92: 'trilobite',
#     93: 'umbrella',
#     94: 'watch',
#     95: 'water_lilly',
#     96: 'wheelchair',
#     97: 'wild_cat',
#     98: 'windsor_chair',
#     99: 'wrench',
#     100: 'yin_yang'
# }

mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}