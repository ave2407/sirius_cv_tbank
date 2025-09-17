# server/app.py (фрагмент)
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from .inference import infer_image_bboxes_bgr
from .schemas import DetectionResponse, Detection, BoundingBox, ErrorResponse  # если используешь Pydantic-схемы

app = FastAPI()


@app.post("/detect", response_model=DetectionResponse,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    # 1) читаем байты
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Decode failed: not an image")

    # 2) инференс -> боксы в пикселях исходного изображения
    try:
        boxes = infer_image_bboxes_bgr(img)  # np.ndarray (N,4) int
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 3) формируем ответ строго по контракту
    dets = []
    for x1, y1, x2, y2 in boxes:
        if x2 <= x1 or y2 <= y1:
            continue
        dets.append(
            Detection(bbox=BoundingBox(x_min=int(x1), y_min=int(y1), x_max=int(x2), y_max=int(y2)))
        )
    return DetectionResponse(detections=dets)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
