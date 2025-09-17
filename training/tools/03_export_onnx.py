"""
python training/tools/export_onnx.py --weights runs/train/tlogo_v11m/weights/best.pt --out weights/best.onnx

"""


import argparse, shutil
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True, help="path to best.pt")
ap.add_argument("--img", type=int, default=896)
ap.add_argument("--opset", type=int, default=12)
ap.add_argument("--out", default="weights/best.onnx")
args = ap.parse_args()

model = YOLO(args.weights)
onnx_path = model.export(format="onnx", imgsz=args.img, opset=args.opset, simplify=True)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
shutil.copy(onnx_path, args.out)
print("Saved:", args.out)
