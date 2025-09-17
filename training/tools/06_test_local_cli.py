#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, mimetypes
from pathlib import Path
import cv2, numpy as np, requests

MIME_FALLBACK = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp",
}

def guess_mime(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return MIME_FALLBACK.get(path.suffix.lower(), mt or "application/octet-stream")

def draw_boxes(img: np.ndarray, detections: list, thickness: int = 3) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    color = (255, 0, 0)
    for i, det in enumerate(detections):
        b = det.get("bbox", {})
        x1 = int(max(0, min(w - 1, b.get("x_min", 0))))
        y1 = int(max(0, min(h - 1, b.get("y_min", 0))))
        x2 = int(max(0, min(w - 1, b.get("x_max", 0))))
        y2 = int(max(0, min(h - 1, b.get("y_max", 0))))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            lbl = f"{i}: bbox"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, lbl, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out

def main():
    ap = argparse.ArgumentParser(description="Тест /detect: отправить изображение(я), распечатать JSON и отрисовать bbox.")
    ap.add_argument("inputs", nargs="+", help="Пути к изображениям")
    ap.add_argument("--url", default="http://127.0.0.1:8000/detect", help="URL эндпоинта /detect")
    ap.add_argument("--outdir", default="out/local_test", help="Куда сохранять визуализации")
    ap.add_argument("--thickness", type=int, default=3)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save_json", action="store_true", help="Сохранить ответ как <name>.json рядом с визуализацией")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    ok_total = fail_total = 0

    for ip_str in args.inputs:
        ip = Path(ip_str)
        if not ip.exists():
            print(f"[WARN] Нет файла: {ip}"); fail_total += 1; continue

        img = cv2.imread(str(ip))
        if img is None:
            print(f"[WARN] Не удалось прочитать изображение: {ip}"); fail_total += 1; continue

        mime = guess_mime(ip)
        with open(ip, "rb") as f:
            files = {"file": (ip.name, f, mime)}  # <-- правильный MIME
            resp = s.post(args.url, files=files, timeout=60)

        text = resp.text
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            print(f"[ERR] {ip.name} -> HTTP {resp.status_code}, non-JSON response:\n{text[:500]}")
            fail_total += 1; continue

        if resp.status_code != 200 or "error" in payload:
            print(f"[ERR] {ip.name} -> HTTP {resp.status_code}, body:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
            fail_total += 1; continue

        dets = payload.get("detections", [])
        print(f"\n[{ip.name}] detections ({len(dets)}):")
        for d in dets:
            print(json.dumps({"bbox": d.get("bbox", {})}, ensure_ascii=False))

        vis = draw_boxes(img, dets, thickness=args.thickness)
        out_img = outdir / f"{ip.stem}_det{ip.suffix}"
        cv2.imwrite(str(out_img), vis)
        if args.save_json:
            (outdir / f"{ip.stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Сохранено: {out_img}")

        if args.show:
            cv2.imshow("detections", vis); cv2.waitKey(0); cv2.destroyAllWindows()

        ok_total += 1

    print(f"\nГотово: OK={ok_total}, FAIL={fail_total}, outdir={outdir.resolve()}")

if __name__ == "__main__":
    main()
