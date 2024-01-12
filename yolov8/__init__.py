from pathlib import Path
import sys

root = Path("/home/zhangqin/wangjl_data/ultralytics")

if str(root) not in sys.path:
    sys.path.append(str(root))

from ultralytics import YOLO
# print(sys.path)