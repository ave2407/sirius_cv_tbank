# server/inference.py
from __future__ import annotations
import os
import cv2
import numpy as np
import onnxruntime as ort
from .utils import letterbox, xywh2xyxy, scale_boxes_back, nms

CONF_THRES = float(os.getenv("CONF_THRES", 0.55))
IOU_THRES = float(os.getenv("IOU_THRES", 0.50))
MAX_DET = int(os.getenv("MAX_DET", 300))
MIN_WH = int(os.getenv("MIN_WH", 3))

MODEL_PATH = os.getenv("MODEL_PATH") or os.getenv("HOST_WEIGHTS") or "weights/best.onnx"
PROVIDERS = [p for p in (os.getenv("ORT_PROVIDERS", "CPUExecutionProvider").split(",")) if p]

# ---- модель + согласование размера ----
_sess = ort.InferenceSession(MODEL_PATH, providers=PROVIDERS)
_inp = _sess.get_inputs()[0]
_net_h, _net_w = _inp.shape[2], _inp.shape[3]

if isinstance(_net_h, int) and isinstance(_net_w, int) and _net_h == _net_w:
    IMG_SIZE = int(os.getenv("IMG_SIZE", _net_h))
else:
    IMG_SIZE = int(os.getenv("IMG_SIZE", 960))  # динамика: дефолт/ENV


def infer_image_bboxes_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """BGR -> np.ndarray (N,4) в пикселях исходного изображения (x1,y1,x2,y2,int)."""
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((0, 4), dtype=np.int32)

    h0, w0 = img_bgr.shape[:2]
    img_lb, ratio, pad = letterbox(img_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW

    out = _sess.run(None, {_inp.name: x})[0]
    out = np.squeeze(out, axis=0) if out.ndim == 3 and out.shape[0] == 1 else out
    arr = out
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # (5, N) — channels-first
    # возможные варианты:
    # (5, N)          -> single-class, pre-NMS
    # (N, 5)          -> single-class, pre-NMS (channels-last)
    # (5+C, N) / (N,5+C) -> multi-class, pre-NMS (редко для 1 класса)
    # (..., 6)        -> post-NMS [x1,y1,x2,y2,score,(class)] — тут NMS не нужен

    boxes_xyxy = None
    scores = None
    need_nms = True  # для pre-NMS

    H_lb, W_lb = img_lb.shape[:2]

    if arr.ndim == 2 and arr.shape[0] == 5:
        # (5, N) -> транспонируем к (N, 5)
        preds = arr.T  # (N, 5) == [x, y, w, h, obj]
        xywh = preds[:, :4].astype(np.float32)
        scores = preds[:, 4].astype(np.float32)
        boxes_xyxy = xywh2xyxy(xywh)
    elif arr.ndim == 2 and arr.shape[1] == 5:
        # (N, 5) -> уже как надо
        preds = arr
        xywh = preds[:, :4].astype(np.float32)
        scores = preds[:, 4].astype(np.float32)
        boxes_xyxy = xywh2xyxy(xywh)
    elif arr.ndim == 2 and (arr.shape[0] > 5 or arr.shape[1] > 5):
        # multi-class (pre-NMS): obj * max(cls)
        if arr.shape[0] > 5:
            preds = arr.T  # (N, 5+C)
        else:
            preds = arr  # (N, 5+C)
        xywh = preds[:, :4].astype(np.float32)
        obj = preds[:, 4].astype(np.float32)
        cls = preds[:, 5:].astype(np.float32) if preds.shape[1] > 5 else None
        scores = obj * (cls.max(axis=1) if cls is not None and cls.size else 1.0)
        boxes_xyxy = xywh2xyxy(xywh)
    elif arr.ndim == 2 and arr.shape[1] >= 6:
        # post-NMS: [x1,y1,x2,y2,score,(class)] — NMS НЕ нужен
        boxes_xyxy = arr[:, :4].astype(np.float32)
        scores = arr[:, 4].astype(np.float32)
        need_nms = False
    else:
        # формат неизвестен
        return np.zeros((0, 4), dtype=np.int32)

    # если координаты оказались 0..1 — домножим на размер letterbox-кадра
    if float(np.nanmax(boxes_xyxy)) <= 2.0:
        boxes_xyxy[:, [0, 2]] *= W_lb
        boxes_xyxy[:, [1, 3]] *= H_lb

    # фильтр по conf
    m = scores >= CONF_THRES
    boxes_xyxy, scores = boxes_xyxy[m], scores[m]
    if boxes_xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.int32)

    # NMS для pre-NMS вариантов
    if need_nms:
        keep = nms(boxes_xyxy, scores, iou_thres=IOU_THRES, max_det=MAX_DET)
        boxes_xyxy = boxes_xyxy[keep]

    # перенос в исходные координаты и округление
    boxes_scaled = scale_boxes_back(boxes_xyxy, ratio, pad, (h0, w0))
    return np.round(boxes_scaled).astype(np.int32)
