## Some features about annotating images

1. label image using YOLOV8, support detection/segmentation/pose_detection.
2. label image using YOLOV5, support detection/segmentation.
3. convert format, such as: pascal->labelme, labelme<->yolo.

### attention
You need append your YOLOV5/YOLOV8 package into yolov5_dir and yolov5_dir.

```python
import sys
yolov5_root = "/home/{*}/yolov5-7.0"
if yolov5_root not in sys.path:
    sys.path.append(yolov5_root)
```