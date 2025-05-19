"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

by lyuwenyu
"""

import math
import os
import sys
import pathlib
import map
import numpy as np
from typing import Iterable

import torch
import torch.amp 

from src.data import CocoEvaluator
from src.misc import (MetricLogger, SmoothedValue, reduce_dict)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.datapoints import BoundingBox

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = kwargs.get('print_freq', 10)
    
    ema = kwargs.get('ema', None)
    scaler = kwargs.get('scaler', None)

    for samples, rohs, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)

        # inputs=samples
        # targets_todevice = [{k: v.to(device) for k, v in t.items()} for t in targets]

        rohs = rohs.to(device)
        inputs=[samples, rohs]
        targets_todevice = []
        for tar in targets:
            # targets_sub = [{k: v.to(device) for k, v in t.items()} for t in tar]
            # targets_todevice.append(targets_sub)
            targets_todevice.extend([{k: v.to(device) for k, v in t.items()} for t in tar])

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(inputs, targets_todevice)
            
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets_todevice)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(inputs, targets_todevice)
            loss_dict = criterion(outputs, targets_todevice)
            loss = sum(loss_dict.values())
            # loss = 0
            # for item in loss_dict:
            #     loss += sum(item.values())
            # loss = loss / len(loss_dict)
            optimizer.zero_grad()
            loss.backward()
            
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
        
        # ema 
        if ema is not None:
            ema.update(model)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())
        
        # loss_value = 0
        # for item in loss_dict:
        #     loss_dict_reduced = reduce_dict(item)
        #     loss_value += sum(loss_dict_reduced.values())
        # loss_value = loss_value / len(loss_dict)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = postprocessors.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    map_evaluator = MeanAveragePrecision(iou_type='bbox', extended_summary=True)

    for samples, rohs, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        rohs = rohs.to(device)
        inputs=[samples, rohs]
        targets_todevice = []
        # for tar in targets:
        #     targets_sub = [{k: v.to(device) for k, v in t.items()} for t in tar]
        #     targets_todevice.append(targets_sub)
        #         targets_todevice = []
        for tar in targets:
            # targets_sub = [{k: v.to(device) for k, v in t.items()} for t in tar]
            # targets_todevice.append(targets_sub)
            targets_todevice.extend([{k: v.to(device) for k, v in t.items()} for t in tar])

        outputs = model(inputs, targets_todevice)

        # orig_target_sizes_list=[]
        # for tar in targets:
        #     orig_target_sizes_list.append( torch.stack([t["orig_size"] for t in tar], dim=0))
        orig_target_sizes = torch.stack([t["size"] for t in targets_todevice], dim=0) 
                
        results = postprocessors(outputs, orig_target_sizes)

        if map_evaluator is not None:
            # for res, tar in zip(results, targets):
            #     tar = [{k: v.to(device) for k, v in d.items()} for d in tar]
            #     map_evaluator.update(res, tar)
            map_evaluator.update(results, targets_todevice)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    metric=map_evaluator.compute()
    metric2print=list(metric.values())[:12]
    metric2print=[x.item() for x in metric2print]

    stats = {}
    stats['all'] = metric
    if 'bbox' in iou_types:
        stats['coco_eval_bbox'] = metric2print

    return stats, map_evaluator



