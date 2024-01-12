from pathlib import Path
import cv2
from shutil import copy2
from myutils.labelme_tools import ModelLabel
from yolov8.model_infer import load_yolov8_model, yolov8_pose_infer, yolov8_det_infer
from yolov5.model_infer import load_yolov5_model, yolov5_det_infer, yolov5_seg_infer
import torch
from myutils.labelme_tools import txt2json

device = torch.device(0)
model = load_yolov5_model("/home/zhangqin/wangjl_data/yolov5-7.0/runs/train/dashboard_center/weights/last.pt", device)

# img = cv2.imread("/home/zhangqin/wangjl_data/meter_reading_recognition/test_dataset/abnormal/0_00009389_seg_biaoji.jpg")
# results = yolov8_pose_infer(model, img, 0.25, 0.5, 0)
# print(results)

root = Path("tmp")

model_label = ModelLabel(root, "det", model, "yolov5", {"iou_thres": 0.25, "conf_thres": 0.45, "device": device}, {})
model_label.run()
