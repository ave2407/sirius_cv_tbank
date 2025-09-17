#!/usr/bin/env python3
"""
Свип NMS IoU на валидации и сохранение результатов (CSV + интерактивный HTML график).

Примеры:
  python sweep_iou.py --model runs/detect/exp/weights/best.pt --data data_balanced.yaml
  python training/tools/05_find_iou.py --model weights/best.onnx --data data/merged_tlogo/data.yaml --imgsz 960 --split val --iou-start 0.3 --iou-end 0.9 --iou-step 0.05
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

def auto_device():
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def run_sweep(model_path: Path, data_yaml: Path, imgsz: int, device: str,
              conf: float, split: str, batch: int,
              iou_start: float, iou_end: float, iou_step: float,
              out_dir: Path) -> pd.DataFrame:
    model = YOLO(str(model_path))

    ious = np.round(np.arange(iou_start, iou_end + 1e-9, iou_step), 2)
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)

    val_kwargs = dict(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        conf=conf,
        verbose=False,
        plots=False,
        save_json=False,
        split=split,
        batch=batch,
    )

    for iou in tqdm(ious, desc="Sweeping IoU"):
        r = model.val(**val_kwargs, iou=float(iou))
        rows.append({
            "iou":        float(iou),
            "mAP50-95":   float(r.box.map),
            "mAP50":      float(r.box.map50),
            "Precision":  float(r.box.mp),
            "Recall":     float(r.box.mr),
            "img_ms":     float(r.speed.get("inference", float("nan"))),
        })

    df = pd.DataFrame(rows).sort_values("iou").reset_index(drop=True)
    csv_path = out_dir / "iou_sweep.csv"
    df.to_csv(csv_path, index=False)

    # Лучший по mAP50-95
    best_idx = int(df["mAP50-95"].idxmax())
    best = df.loc[best_idx].to_dict()
    best_iou = best["iou"]
    print(f"\n✔ Best IoU by mAP50-95 = {best_iou:.2f} | "
          f"mAP50-95={best['mAP50-95']:.4f} mAP50={best['mAP50']:.4f} "
          f"P={best['Precision']:.4f} R={best['Recall']:.4f}")
    print(f"CSV saved to: {csv_path}")

    # Интерактивные графики → HTML
    try:
        import plotly.express as px
        from plotly.offline import plot as plot_offline

        fig1 = px.line(df, x="iou", y=["mAP50-95", "mAP50"], markers=True, title="mAP vs NMS IoU")
        fig1.add_vline(x=best_iou, line_dash="dash")
        fig2 = px.line(df, x="iou", y=["Precision", "Recall"], markers=True, title="Precision/Recall vs NMS IoU")
        fig2.add_vline(x=best_iou, line_dash="dash")

        html_path = out_dir / "iou_sweep.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(plot_offline(fig1, include_plotlyjs="cdn", output_type="div"))
            f.write("<hr/>")
            f.write(plot_offline(fig2, include_plotlyjs=False, output_type="div"))
        print(f"HTML saved to: {html_path}")

        # (опционально) статичные PNG, если установлен kaleido
        try:
            fig1.write_image(out_dir / "iou_map.png", scale=2)
            fig2.write_image(out_dir / "iou_pr.png",  scale=2)
            print("PNG plots saved.")
        except Exception:
            pass

    except Exception as e:
        print(f"Plotly not available or failed to render HTML: {e}")

    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="Path to best.pt")
    ap.add_argument("--data", type=Path, required=True, help="Path to data.yaml (or data_balanced.yaml)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default=auto_device(), help="cuda:0 or cpu (auto)")
    ap.add_argument("--conf", type=float, default=0.001, help="confidence threshold used for AP calc")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="which split to evaluate")
    ap.add_argument("--batch", type=int, default=32)

    ap.add_argument("--iou-start", type=float, default=0.30)
    ap.add_argument("--iou-end",   type=float, default=0.90)
    ap.add_argument("--iou-step",  type=float, default=0.05)

    ap.add_argument("--out", type=Path, default=Path("./iou_sweep_out"))
    args = ap.parse_args()

    if not args.model.exists():
        ap.error(f"model not found: {args.model}")
    if not args.data.exists():
        ap.error(f"data.yaml not found: {args.data}")

    # Защитимся от редкой ошибки с SymPy:
    try:
        import sympy  # noqa
        assert hasattr(sympy, "printing")
    except Exception:
        print("⚠️  SymPy в окружении повреждён. Установите стабильную версию: pip install 'sympy==1.12'")

    run_sweep(args.model, args.data, args.imgsz, args.device,
              args.conf, args.split, args.batch,
              args.iou_start, args.iou_end, args.iou_step,
              args.out)

if __name__ == "__main__":
    raise SystemExit(main())
