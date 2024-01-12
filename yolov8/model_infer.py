
import sys
from pathlib import Path
import torch
import cv2

yolov8_root = Path("/home/zhangqin/wangjl_data/ultralytics")

if str(yolov8_root) not in sys.path:
    sys.path.append(str(yolov8_root))

from ultralytics import YOLO

from myutils.dataStruct import PoseOut, DetOut, SegOut




def load_yolov8_model(weight):
    model = YOLO(weight)
    return model


def yolov8_pose_infer(model, img, iou_thres, conf_thres, device, **kwargs):
    results = model(img, device=device, iou=iou_thres, conf=conf_thres, **kwargs)[0]
    
    res = []
    if results.keypoints.data.shape[1]:
        kpts = results.keypoints.xy.cpu().numpy()
        boxes = results.boxes.data.cpu().numpy()
        xywhs = results.boxes.xywh.cpu().numpy()
        xywhns = results.boxes.xywhn.cpu().numpy()
        xyxyns = results.boxes.xyxyn.cpu().numpy()  
        for box, kpt, xywh, xywhn, xyxyn in zip(boxes, kpts, xywhs, xywhns, xyxyns):
            xyxy, conf, class_id = box[:4], box[4], int(box[-1])
            res.append(PoseOut(
                xyxy, xyxyn, xywh, xywhn, class_id, conf, results.names[class_id], kpt
            ))
    return res


def yolov8_seg_infer(model, img, conf_thres, iou_thres, device, **kwargs):
    results = model(img, save_conf=True, conf=conf_thres, iou=iou_thres, device=device, **kwargs)[0]

    res = []
    if results.masks is not None:
        boxes = results.boxes.data.cpu().numpy()
        xywhs = results.boxes.xywh.cpu().numpy()
        xywhns = results.boxes.xywhn.cpu().numpy()
        xyxyns = results.boxes.xyxyn.cpu().numpy()  
        segments_list = results.masks.xy
        mask_list = results.masks.data.cpu().numpy() 
        for box, segments, mask, xywh, xywhn, xyxyn in zip(boxes, segments_list, mask_list, xywhs, xywhns, xyxyns):
            xyxy, conf, class_id = box[:4], box[4], int(box[-1])
            res.append(SegOut(
                xyxy, xyxyn, xywh, xywhn, class_id, conf, 
                results.names[class_id], mask, segments
            ))
    return res


def yolov8_det_infer(model, img, iou_thres, conf_thres, device, **kwargs):
    results = model(img, save_conf=True, conf=conf_thres, iou=iou_thres, device=device, **kwargs)[0]

    res = []
    if len(results.boxes.data) > 0:
        boxes = results.boxes.data.cpu().numpy()
        xywhs = results.boxes.xywh.cpu().numpy()
        xywhns = results.boxes.xywhn.cpu().numpy()
        xyxyns = results.boxes.xyxyn.cpu().numpy()  
        for box, xywh, xywhn, xyxyn in zip(boxes, xywhs, xywhns, xyxyns):
            xyxy, conf, class_id = box[:4], box[4], int(box[-1])
            res.append(DetOut(
                xyxy, xyxyn, xywh, xywhn, class_id, conf, results.names[class_id]
            ))
    return res





# if __name__ == '__main__':
#     weights = "/home/zhangqin/wangjl_data/digital_meter_reading_recognition/weights/last.pt"
#     device = torch.device(0)
#     model = load_yolov8_model(weights)

#     img = cv2.imread("/home/zhangqin/wangjl_data/digital_meter_reading_recognition/digital_meter_testset/test_dataset/2023_09_14_digital_113_jpeg.rf.e31dae714cf7674b9b4fb508590db400_seg7_000.jpg")
#     res = yolov8_det_infer(model, img, 0.1, 0.6, device)
#     print(res)
#     # img_root = Path("/home/zhangqin/wangjl_data/parseq-main/test_data")
#     # for img_f in img_root.iterdir():
#     #     img = cv2.imread(str(img_f))

#     #     res = ocr_model_infer(model, img, device)
#     #     print(res)