import os
import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import List, Tuple
from .utils import letterbox, xywh_to_xyxy, nms_numpy, clip_boxes_xyxy

# Конфиг из окружения
MODEL_PATH = os.getenv("MODEL_PATH", "weights/best.onnx")
CONF_THRES = float(os.getenv("CONF_THRES", "0.50"))  # возьмите из вашего best_conf
IOU_THRES = float(os.getenv("IOU_THRES", "0.50"))
IMG_SIZE = int(float(os.getenv("IMG_SIZE", "960")))  # под ваш экспорт
MAX_DET = int(os.getenv("MAX_DET", "300"))

# ...
# вместо:
# providers = [('CUDAExecutionProvider', {'cudnn_conv_use_max_workspace': '1'})] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
# session = ort.InferenceSession(MODEL_PATH, providers=providers)

# делаем так (CPU-оптимально и безопасно):
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# Тюним потоки из ENV (см. Dockerfile)
so.intra_op_num_threads = int(os.getenv("ORT_INTRA_THREADS", "0"))  # 0 = по умолчанию
so.inter_op_num_threads = int(os.getenv("ORT_INTER_THREADS", "0"))

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=['CPUExecutionProvider']
)
inp = session.get_inputs()[0]
inp_name = inp.name


def _preprocess(pil_img: Image.Image) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[int, int]]:
    """Возвращает тензор [1,3,H,W] float32 и (ratio, dw, dh), (orig_w, orig_h)."""
    im = pil_img.convert("RGB")
    orig_w, orig_h = im.size
    r, dw, dh = letterbox((orig_h, orig_w), IMG_SIZE)
    new_w, new_h = int(round(orig_w * r)), int(round(orig_h * r))
    im_resized = im.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (114, 114, 114))
    canvas.paste(im_resized, (int(round(dw)), int(round(dh))))
    arr = np.asarray(canvas, dtype=np.uint8)
    arr = arr.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW
    arr = np.expand_dims(arr, 0)
    return arr, (r, dw, dh), (orig_w, orig_h)


def _postprocess(pred: np.ndarray, ratio_dw_dh, orig_wh) -> Tuple[np.ndarray, np.ndarray]:
    """
    Универсальный постпроцесс для стандартного ONNX экспорта Ultralytics YOLO:
    pred: (1, N, D) или (1, D, N), где D = 4 + 1 + C. Берём класс 0 (tbank_logo).
    Возвращает (boxes_xyxy_abs, scores).
    """
    if pred.ndim == 3:
        if pred.shape[1] in (84, 85) and pred.shape[2] > pred.shape[1]:
            pred = np.transpose(pred, (0, 2, 1))  # (1, N, D)
    pred = pred[0]  # (N, D)

    D = pred.shape[1]
    if D < 6:
        # Неожиданный формат
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # разбор компонент
    xywh = pred[:, 0:4]
    obj = pred[:, 4:5]
    cls = pred[:, 5:]  # (N, C)

    # если классов больше 1 — берём класс 0 (наш единственный)
    if cls.shape[1] >= 1:
        cls0 = cls[:, 0:1]
    else:
        # на всякий случай, если экспорт без классов — используем obj
        cls0 = np.ones_like(obj)

    scores = (obj * cls0).squeeze(1)
    keep = scores >= CONF_THRES
    if not np.any(keep):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    xywh = xywh[keep]
    scores = scores[keep]

    # xywh (в координатах входа IMG_SIZE) -> xyxy входа
    boxes = xywh_to_xyxy(xywh)

    # снимаем letterbox: (x - dw)/r, (y - dh)/r
    r, dw, dh = ratio_dw_dh
    orig_w, orig_h = orig_wh
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / r
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / r

    # клип к оригинальному изображению
    boxes = clip_boxes_xyxy(boxes, orig_w, orig_h)

    # NMS
    if boxes.shape[0] > 0:
        keep_idx = nms_numpy(boxes, scores, IOU_THRES)
        if keep_idx.size > 0:
            boxes = boxes[keep_idx]
            scores = scores[keep_idx]
    # ограничение по числу детекций
    if boxes.shape[0] > MAX_DET:
        order = scores.argsort()[::-1][:MAX_DET]
        boxes, scores = boxes[order], scores[order]

    return boxes, scores


def infer_bytes(image_bytes: bytes) -> List[Tuple[int, int, int, int]]:
    pil_img = Image.open(io.BytesIO(image_bytes))
    arr, ratio_dw_dh, orig_wh = _preprocess(pil_img)
    pred = session.run(None, {inp_name: arr})[0]
    boxes, _scores = _postprocess(pred, ratio_dw_dh, orig_wh)
    boxes = boxes.astype(np.int32)
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]
