from pathlib import Path
import cv2
from shutil import copy2
from myutils.labelme_tools import ModelLabel, txt2json
from yolov8.model_infer import load_yolov8_model, yolov8_pose_infer, yolov8_det_infer
from yolov5.model_infer import load_yolov5_model, yolov5_det_infer, yolov5_seg_infer
import torch
from myutils.labelme_tools import txt2json

device = torch.device(0)


# # use yolov8-seg to seg images.
# model = load_yolov8_model("/home/zhangqin/wangjl_data/ultralytics/runs/segment/seg_road_damege3/weights/last.pt")
# root = Path("road_damege/Norway")
# yolo2shape_params = {"save_points_num":30}   # save the polygon points_num
# model_params = {
#     "iou_thres": 0.25, 
#     "conf_thres": 0.45, 
#     "device": device
# }
# task = "seg"
# model_arch="yolov8"
# model_label = ModelLabel(
#     img_root=root, 
#     task=task, 
#     model=model, 
#     model_arch=model_arch, 
#     model_params=model_params, 
#     yolo2shape_params=yolo2shape_params
# )
# model_label.run()


# use yolov8-det to det images.
# model = load_yolov8_model("/home/zhangqin/wangjl_data/ultralytics/runs/detect/det_fallen_tree/weights/last.pt")
# root = Path("/home/zhangqin/wangjl_data/dataset/smart_city/fallen_tree/Fallen_Trees_Improve_1-1185.v1i.yolov8/valid/images")
# yolo2shape_params = {}
# model_params = {
#     "iou_thres": 0.25, 
#     "conf_thres": 0.45, 
#     "device": device
# }
# task = "det"
# model_arch="yolov8"
# model_label = ModelLabel(
#     img_root=root, 
#     task=task, 
#     model=model, 
#     model_arch=model_arch, 
#     model_params=model_params, 
#     yolo2shape_params=yolo2shape_params
# )
# model_label.run()




# # use yolov5-seg to seg images.
# model = load_yolov5_model("xxx.pt")
# root = Path("road_damege/Norway")
# yolo2shape_params = {"save_points_num":30}
# model_params = {
#     "iou_thres": 0.25, 
#     "conf_thres": 0.45, 
#     "device": device
# }
# task = "seg"
# model_arch="yolov5"
# model_label = ModelLabel(
#     img_root=root, 
#     task=task, 
#     model=model, 
#     model_arch=model_arch, 
#     model_params=model_params, 
#     yolo2shape_params=yolo2shape_params
# )
# model_label.run()



# # use yolov5-det to det images.
# model = load_yolov5_model("xxx.pt")
# root = Path("road_damege/Norway")
# yolo2shape_params = {}
# model_params = {
#     "iou_thres": 0.25, 
#     "conf_thres": 0.45, 
#     "device": device
# }
# task = "det"
# model_arch="yolov5"
# model_label = ModelLabel(
#     img_root=root, 
#     task=task, 
#     model=model, 
#     model_arch=model_arch, 
#     model_params=model_params, 
#     yolo2shape_params=yolo2shape_params
# )
# model_label.run()