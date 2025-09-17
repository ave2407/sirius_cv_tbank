#!/usr/bin/env python3
import os
import glob
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection

PROMPTS = [
    "T-Bank shield logo",
    "буква Т в щите",
    "логотип Т-Банка в щите",
    "shield with letter T"
]

def save_yolo_txt(txt_path: Path, boxes, img_w, img_h):
    with open(txt_path, "w") as f:
        for x1, y1, x2, y2 in boxes:
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="папка с изображениями")
    parser.add_argument("--out", required=True, help="куда сохранить результаты")
    parser.add_argument("--score-thr", type=float, default=0.25, help="порог детекции")
    args = parser.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.out)
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").eval()
    if torch.cuda.is_available():
        model = model.cuda()

    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        img_paths.extend(glob.glob(str(img_dir / "**" / ext), recursive=True))

    for img_path in tqdm(img_paths, desc="auto-annotating"):
        img = Image.open(img_path).convert("RGB")
        inputs = processor(text=[PROMPTS], images=[img], return_tensors="pt", truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])  # (h,w)
        if torch.cuda.is_available():
            target_sizes = target_sizes.cuda()
        results = processor.post_process_object_detection(
            outputs, threshold=args.score_thr, target_sizes=target_sizes
        )[0]
        boxes = results["boxes"].cpu().numpy()

        out_img_path = out_images / Path(img_path).name
        img.save(out_img_path)

        out_txt_path = out_labels / (Path(img_path).stem + ".txt")
        if len(boxes) > 0:
            save_yolo_txt(out_txt_path, boxes, img.width, img.height)
        else:
            open(out_txt_path, "w").close()

if __name__ == "__main__":
    main()
