"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data

import torchvision

torchvision.disable_beta_transforms_warning()

from torch.utils.data import Dataset, DataLoader
from torchvision import datapoints
from pycocotools import mask as coco_mask

from src.core import register
from PIL import Image
import os
import json
import numpy as np

__all__ = ["EsoDetection"]


@register
class EsoDetection(Dataset):
    __inject__ = ["transforms"]
    # __share__ = ["remap_mscoco_category"]

    def __init__(self, img_folder, density_file, transforms):
        super(Dataset, self).__init__()
        self._transforms = transforms
        # self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.density_file = density_file
        # self.return_masks = return_masks
        # self.remap_mscoco_category = remap_mscoco_category
        self.imgs = os.listdir(img_folder)
        self.img_size = np.array([1280, 720])
        self.density_dict = self.get_density_dict()

    def get_density_dict(self):
        with open(self.density_file, "r") as f:
            density_dict = json.load(f)
        f.close()
        return density_dict

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        imgs_name = ["0000.png", "0001.png", "0002.png", "0003.png", "0004.png"]
        dirname = self.imgs[idx]
        img_path = os.path.join(self.img_folder, dirname, "event_frame")
        label_path = os.path.join(self.img_folder, dirname, "labels.json")
        f = open(label_path, "r")
        label_dict = json.load(f)
        f.close()
        imgs = []
        density = torch.Tensor(self.density_dict[dirname])
        targets = []
        for i in range(len(imgs_name)):
            item = imgs_name[i]
            item_path = os.path.join(img_path, item)
            img = Image.open(item_path).convert("RGB")
            target = {}
            if len(label_dict[item]) == 0:#如果当前帧没有物体
                boxes_cls = torch.Tensor([[0,0,0.001,0.001,-1]])
            else:
                boxes_cls = torch.Tensor(label_dict[item])
            boxes = boxes_cls[:, :4]
            labels = boxes_cls[:, -1].long()
            target['image_id'] = torch.Tensor([idx + i*0.1])
            target["boxes"] = boxes
            target["labels"] = labels
            target["size"] = torch.Tensor(self.img_size).long()
            target["orig_size"] = torch.Tensor(self.img_size).long()
            if "boxes" in target:
                target["boxes"] = datapoints.BoundingBox(
                    target["boxes"], 
                    format=datapoints.BoundingBoxFormat.XYXY, 
                    spatial_size=img.size[::-1])  # h w
            if self._transforms is not None:
                img, target = self._transforms(img, target)
                shape_h,shape_w=img.shape[1:]
                target["size"] = torch.Tensor([shape_h,shape_w]).long()
            imgs.append(img)
            targets.append(target)
            
            images=torch.stack(imgs, dim=0)

        return images, density, targets

    def extra_repr(self) -> str:
        s = f" img_folder: {self.img_folder}\n "
        s += f" return_masks: {self.return_masks}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"

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

        # image_id = target["image_id"]
        # image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.remap_mscoco_category:
            classes = [eso_category2label[obj["category_id"]] for obj in anno]
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
        # target["image_id"] = image_id
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


eso_category2name = {
    0: "people",
    1: "car",
    2: "bicycle",
    3: "electric bicycle",
    4: "basketball",
    5: "ping pong",
    6: "goose",
    7: "cat",
    8: "bird",
    9: "UAV",
}

eso_category2label = {k: i for i, k in enumerate(eso_category2name.keys())}
eso_label2category = {v: k for k, v in eso_category2label.items()}
