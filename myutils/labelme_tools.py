import numpy as np
import cv2
from pathlib import Path
import torch.nn as nn

import xml.dom.minidom

from typing import List, Union, Dict
from collections import OrderedDict
from yolov8.model_infer import yolov8_det_infer, yolov8_seg_infer, yolov8_pose_infer
from yolov5.model_infer import yolov5_det_infer, yolov5_seg_infer
from myutils.dataStruct import DetOut, SegOut, PoseOut
from myutils.utils import (
    read_json, write_json, 
    kmeans_cluster, xywh2xyxy
)

IMAGE_SUFFIX = [".jpg", ".png", ".jpeg", ".bmp"]



def labelme_info_generator(shapes: List, imagePath: str, imageHeight: int, imageWidth: int):
    labelme_info = {
        "flags": {},
        "shapes": shapes,
        "imagePath": imagePath,
        "imageData": None,
        "imageHeight": imageHeight,
        "imageWidth": imageWidth,
    }
    return labelme_info
    

def labelme_shape_generator(label: str, points: List, shape_type: str):
    shape = {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": shape_type,
        "flags": {},
    }
    return shape



def xml2json(xml_root: Union[Path,str], save_root: Union[Path,str]) -> None:
    """Convert PASCAL format to labelme format. Support detection (rectangle shape) only."""

    xml_root = Path(xml_root)
    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True, parents=True)

    for xml_f in xml_root.iterdir():
        if xml_f.suffix == ".xml":
            dom = xml.dom.minidom.parse(f"{xml_f}")

            shapes = []
            objects = dom.getElementsByTagName('object')
            for object in objects:
                name = object.getElementsByTagName('name')[0].firstChild.data
                bbox = object.getElementsByTagName('bndbox')[0]
                bbox_xmin = float(bbox.getElementsByTagName('xmin')[0].firstChild.data)
                bbox_ymin = float(bbox.getElementsByTagName('ymin')[0].firstChild.data)
                bbox_xmax = float(bbox.getElementsByTagName('xmax')[0].firstChild.data)
                bbox_ymax = float(bbox.getElementsByTagName('ymax')[0].firstChild.data)
                points = [[bbox_xmin, bbox_ymin], [bbox_xmax, bbox_ymax]]
                shape = labelme_shape_generator(name, points, "rectangle")
                shapes.append(shape)

            img_filename = dom.getElementsByTagName('filename')[0].firstChild.data
        
            file_size = dom.getElementsByTagName('size')
            height = int(file_size[0].getElementsByTagName("height")[0].firstChild.data)
            width = int(file_size[0].getElementsByTagName("width")[0].firstChild.data)

            labelme_info = labelme_info_generator(img_filename, height, width, shapes)

            json_f = save_root / (xml_f.stem+".json")
            write_json(labelme_info, json_f)


def json2txt(json_root: Union[Path,str], label2id: Dict[str, int], task: str ="det") -> None:
    """Convert labelme to yolo format. Support detection and segmentation only."""
    json_root = Path(json_root)
    for json_f in json_root.iterdir():
        if json_f.suffix == ".json":
            A = read_json(json_f)
            H, W = A["imageHeight"], A["imageWidth"]
            txt_f = json_f.with_suffix(".txt")

            if task == "det":
                for item in A["shapes"]:
                    label = item['label']
                    if label in label2id:
                        classid = label2id[label]
                        shape_type = item["shape_type"]
                        if shape_type == "rectangle" :
                            pt0, pt1 = item["points"]
                            x = (pt0[0]+pt1[0])/2/W
                            y = (pt0[1]+pt1[1])/2/H
                            w = abs(pt0[0]-pt1[0])/W
                            h = abs(pt0[1]-pt1[1])/H
                            with open(txt_f, "a") as f:
                                f.write(f"{classid} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
                
            elif task == "seg":
                for item in A["shapes"]:
                    label = item['label']
                    if label in label2id:
                        classid = label2id[label]
                        shape_type = item["shape_type"]
                        if shape_type == "polygon":
                            with open(txt_f, "a") as f:
                                f.write(f"{classid} ")
                                for point in item["points"]:
                                    x, y = point
                                    x, y = x/W, y/H
                                    f.write(f"{x:.5f} {y:.5f} ")
                                f.write("\n")

            else:
                raise NotImplementedError("Support det and seg only.")


def txt2json(root: Union[Path,str], id2label: Dict[int, str]={}, task: str="det") -> None:
    """Convert yolo to labelme format. Support detection and segmentation only.

    Args:
        root (Path): images and labels are in the root directory.
        id2label (Dict, optional): _description_. Defaults to {}.
        task (str, optional): _description_. Defaults to "det".
    """
    root = Path(root)
    images_dir = root / "images"
    labels_dir = root / "labels"

    for img_f in images_dir.iterdir():
        if img_f.suffix in IMAGE_SUFFIX:
            img = cv2.imread(str(img_f))
            H, W = img.shape[:2]
            
            txt_f = labels_dir / img_f.with_suffix(".txt").name
            if txt_f.exists():
                with open(txt_f, "r") as f:
                    lines = f.readlines()

                shapes = []
                for line in lines:
                    line = line.split(" ")
                    line = [float(i) for i in line]

                    if task == "det":
                        class_id, xywh = int(line[0]), line[1:]
                        xyxy = xywh2xyxy(xywh, (H, W))
                        points = [xyxy[:2], xyxy[2:]]

                        label = id2label[class_id] if class_id in id2label else str(class_id)
                        shape = labelme_shape_generator(label, points, "rectangle")

                    elif task == "seg":
                        class_id, points = int(line[0]), line[1:]
                        points = np.array(points, dtype=np.float32).reshape((-1, 2))
                        points = points * np.array((W, H))
                        points = points.tolist()

                        label = id2label[class_id] if class_id in id2label else str(class_id)
                        shape = labelme_shape_generator(label, points, "polygon")
                    
                    else:
                        raise NotImplementedError("Support det and seg only.")
                    
                    shapes.append(shape)
                
                if len(shapes) > 0:
                    json_f = img_f.with_suffix(".json")
                    labelme_info = labelme_info_generator(shapes, img_f.name, H, W)
                    write_json(labelme_info, json_f)
                





def detout2rect(outputs: List[DetOut]):
    shapes = []
    for out in outputs:
        rect = labelme_shape_generator(
            out.label, out.xyxy.reshape((2,2)).tolist(), "rectangle"
        )
        shapes.append(rect)
    return shapes


def remove_redundant_points(points: np.ndarray, save_points_num: int):
    if len(points) <= save_points_num:
        return points
    labels = kmeans_cluster(points, save_points_num)
    
    select_idx = OrderedDict()
    for i in labels:
        if i not in select_idx:
            tmp_idx = int(np.where(labels==i)[0].mean())
            select_idx[i] = tmp_idx
    select_points = points[list(select_idx.values())]

    return select_points

def segout2polygon(outputs: List[SegOut], save_points_num: int):
    shapes = []
    for out in outputs:
        points = out.segments.reshape((-1,2))
        points = remove_redundant_points(points, save_points_num).tolist()
        polygon = labelme_shape_generator(
            out.label, points, "polygon"
        )
        shapes.append(polygon)
    return shapes


def poseout2kpt(outputs: List[PoseOut]):
    shapes = []
    for i, out in enumerate(outputs):
        rect = labelme_shape_generator(
            f"{out.label}{i}", out.xyxy.reshape((2,2)).tolist(), "rectangle"
        )

        kpts = out.kpt.reshape((-1, 2)).tolist()
        if len(kpts) == 1:
            shape_type = "point"
        elif len(kpts) == 2:
            shape_type = "line"
        else:
            shape_type = "polygon"
        
        kpt_shape = labelme_shape_generator(f"{out.label}{i}", kpts, shape_type)
        shapes += [rect, kpt_shape]

    return shapes



class ModelLabel:

    support_tasks = ["det", "seg", "pose"]
    support_models = ["yolov5", "yolov8"]

    infer_func = {
        "det": {"yolov5": yolov5_det_infer, "yolov8": yolov8_det_infer},
        "seg": {"yolov5": yolov5_seg_infer, "yolov8": yolov8_seg_infer},
        "pose": {"yolov8": yolov8_pose_infer}
    }
    yolo2shape_func = {
        "det": detout2rect,
        "seg": segout2polygon,
        "pose": poseout2kpt
    }

    def __init__(
            self, img_root: Union[Path, str], task: str, model: nn.Module, 
            model_arch: str, model_params: Dict, yolo2shape_params: Dict
        ) -> None:

        if task not in self.support_tasks:
            raise NotImplementedError(f"task must be in {self.support_tasks}")
        
        if model_arch not in self.support_models:
            raise NotImplementedError(f"model_arch must be in {self.support_models}")
    
        self.img_root = Path(img_root)
        self.task = task
        self.model = model
        self.model_arch = model_arch
        self.model_params = model_params
        self.yolo2shape_params = yolo2shape_params

        self.infer = self.infer_func[task][model_arch]
        self.yolo2shape = self.yolo2shape_func[task]


    def run(self):
        for img_f in self.img_root.iterdir():
            if img_f.suffix in IMAGE_SUFFIX:
                img = cv2.imread(str(img_f))
                outputs = self.infer(self.model, img, **self.model_params)
                shapes = self.yolo2shape(outputs, **self.yolo2shape_params)

                json_f = img_f.with_suffix(".json")
                if json_f.exists():
                    A = read_json(json_f)
                    A["shapes"] += shapes
                else:
                    H, W = img.shape[:2]
                    A = labelme_info_generator(shapes, img_f.name, H, W)
                write_json(A, json_f)
    

