import json
from pathlib import Path
from typing import List, Union, Dict, Tuple
import numpy as np
from sklearn.cluster import KMeans



def read_json(json_f: Union[Path, str]):
    json_f = Path(json_f)
    if json_f.exists():
        with open(json_f, "r", encoding="utf-8") as f:
            A = json.load(f)
        return A
    else:
        raise FileNotFoundError(f"{json_f} does not exist.")
    

def write_json(content: Dict, json_f: Union[Path, str]):
    json_f = Path(json_f)
    save_root = json_f.absolute().parent
    if not save_root.exists():
        save_root.mkdir(exist_ok=True, parents=True)
    
    with open(json_f, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def kmeans_cluster(data: np.ndarray, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels


def xywh2xyxy(xywh:Union[List, Tuple], imgsz: Union[List, Tuple]=None):

    ctr_x, ctr_y, w, h = [float(i) for i in xywh]
    pt0_x = ctr_x - 0.5*w
    pt0_y = ctr_y - 0.5*h
    pt1_x = ctr_x + 0.5*w
    pt1_y = ctr_y + 0.5*h
    if imgsz is not None:
        H, W = imgsz
        pt0_x = pt0_x*W
        pt0_y = pt0_y*H
        pt1_x = pt1_x*W 
        pt1_y = pt1_y*H
    
    return (pt0_x, pt0_y, pt1_x, pt1_y)


def organize_yolo_dataset_dir(root: Union[str,Path]):
    """organize yolov5/yolov8 dataset dir.

    Args:
        root (Union[str,Path]): _description_
    """
    root = Path(root)
    images_root = root / "images"
    labels_root = root / "labels"
    images_root.mkdir(exist_ok=True)
    labels_root.mkdir(exist_ok=True)

    for txt_f in root.iterdir():
        if txt_f.suffix == ".txt":
            json_f = txt_f.with_suffix(".json")
            with open(json_f, "r") as f:
                A = json.load(f)
            img_suffix = Path(A["imagePath"]).suffix
            img_f = txt_f.with_suffix(img_suffix)
            if img_f.exists():
                txt_f.rename(labels_root / txt_f.name)
                img_f.rename(images_root / img_f.name)
                json_f.rename(images_root / json_f.name)