"""
python tools/validate.py --weights runs/train/tlogo_v11m/weights/best.pt --split val
"""

import argparse, json
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True, help=".pt weights")
ap.add_argument("--data", default="training/data.yaml")
ap.add_argument("--imgsz", type=int, default=896)
ap.add_argument("--iou", type=float, default=0.5)
ap.add_argument("--conf", type=float, default=0.55)
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--out", default="out/metrics.json")
args = ap.parse_args()

model = YOLO(args.weights)
res = model.val(data=args.data, imgsz=args.imgsz, iou=args.iou, conf=args.conf, split=args.split, verbose=False)
metrics = res.results_dict

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(dict(split=args.split, conf=args.conf, iou=args.iou, imgsz=args.imgsz, **metrics), f, indent=2,
              ensure_ascii=False)

print(json.dumps(metrics, indent=2))
print("Saved:", args.out)
