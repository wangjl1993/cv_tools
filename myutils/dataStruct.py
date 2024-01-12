from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import numpy as np
from typing import List, Union


@dataclass_json
@dataclass
class DetOut:
    xyxy: np.ndarray
    xyxyn: np.ndarray
    xywh: np.ndarray
    xywhn: np.ndarray
    class_id: int
    conf: float
    label: str

    def __post_init__(self):
        self.xyxy = self.xyxy.astype(np.int16)
        self.xywh = self.xywh.astype(np.int16)
        self.center = self.xyxy[:2]


@dataclass_json
@dataclass
class SegOut(DetOut):
    mask: np.ndarray
    segments: np.ndarray


@dataclass_json
@dataclass
class PoseOut(DetOut):
    kpt: np.ndarray

