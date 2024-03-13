
import sys
from pathlib import Path
import torch
import cv2

yolov5_root = "/home/zhangqin/wangjl_data/yolov5-7.0/"
if yolov5_root not in sys.path:
    sys.path.append(yolov5_root)

from models.common import DetectMultiBackend
from utils.general import (
    check_img_size, non_max_suppression, 
    scale_boxes, scale_segments,
)
from utils.segment.general import masks2segments, process_mask_native
from utils.dataloaders import letterbox
from utils.torch_utils import smart_inference_mode
import numpy as np
from typing import Union
import json

from myutils.dataStruct import DetOut, SegOut



def load_yolov5_model(weight, device):
    model = DetectMultiBackend(weight, device)
    return model


def image2input(img0, imgsz, stride, device):
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]
    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img /= 255  # 0 - 255 to 0.0 - 1.0
    if len(img.shape) == 3:
        img = img.unsqueeze(0)  # expand for batch dim
    return img


@smart_inference_mode()
def yolov5_seg_infer(model, img, conf_thres, iou_thres, device, imgsz=(640,640), **kwargs):

    raw_shape = img.shape[:2]
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride) 
    input = image2input(img, imgsz, stride, device)
    pred, proto = model(input, **kwargs)[:2]
    pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

    res = []

    for i, det in enumerate(pred):  # per image
        if len(det):
            det[:, :4] = scale_boxes(input.shape[2:], det[:, :4], raw_shape).round()  # rescale boxes to im0 size
            masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], raw_shape[:2])  # HWC
            
            segments = [
                scale_segments(raw_shape, x, raw_shape, normalize=True)
                for x in reversed(masks2segments(masks))
            ]

            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6].cpu().numpy())):
                res.append(SegOut(
                    np.array(xyxy), np.array(xyxy), np.array(xyxy), np.array(xyxy), int(cls), float(conf), str(cls),  # did not compute xyxyn, xywh, xywhn, use xyxy instead
                    masks[j].cpu().numpy(), segments[j]*np.array(raw_shape)
                ))
   
    return res

@smart_inference_mode()
def yolov5_det_infer(model, img, conf_thres, iou_thres, device, imgsz=(640,640), **kwargs):
    
    raw_shape = img.shape[:2]
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride) 
    input = image2input(img, imgsz, stride, device)
    
    pred = model(input, **kwargs)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    res = []

    for i, det in enumerate(pred):  # per image
        if len(det):
            det[:, :4] = scale_boxes(input.shape[2:], det[:, :4], raw_shape).round()
            
            for *xyxy, conf, cls in reversed(det.cpu().numpy()):
                res.append(DetOut(
                    np.array(xyxy), np.array(xyxy), np.array(xyxy), np.array(xyxy), int(cls), float(conf), str(cls)       # did not compute xyxyn, xywh, xywhn, use xyxy instead
                ))

    return res

