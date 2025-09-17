# server/utils.py
from __future__ import annotations
import cv2
import numpy as np

def letterbox(im: np.ndarray, new_shape=960, color=(114, 114, 114)):
    """Resize+pad до квадрата (как в Ultralytics). Возвращает: img, ratio, (dw,dh)."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    dw /= 2; dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def scale_boxes_back(boxes_xyxy: np.ndarray, ratio: float, pad: tuple[float, float], orig_shape) -> np.ndarray:
    """Перевод координат из letterbox-пространства в исходный кадр + клип."""
    (dw, dh) = pad
    boxes = boxes_xyxy.astype(np.float32).copy()
    boxes[:, [0, 2]] -= dw
    boxes[:, [1, 3]] -= dh
    boxes[:, :4] /= max(ratio, 1e-9)

    h0, w0 = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0 - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0 - 1)
    return boxes

def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    area_b = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres=0.5, max_det=300) -> np.ndarray:
    """Простой NMS для xyxy."""
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0 and len(keep) < max_det:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = box_iou_xyxy(boxes[i:i+1], boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious < iou_thres]
    return np.array(keep, dtype=int)
