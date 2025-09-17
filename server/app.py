import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .schemas import BoundingBox, Detection, DetectionResponse, ErrorResponse
from .inference import infer_bytes

SUPPORTED = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/jpg"}

app = FastAPI(
    title="T-Bank Logo Detector",
    version="1.0.0",
    description="REST API сервис детекции логотипа Т-Банка."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    if file.content_type not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Unsupported content-type: {file.content_type}. Supported: {', '.join(SUPPORTED)}")

    try:
        content = await file.read()
        boxes = infer_bytes(content)
        detections = [
            Detection(bbox=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2))
            for (x1, y1, x2, y2) in boxes
        ]
        return DetectionResponse(detections=detections)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="inference_failed", detail=str(e)).dict(),
        )
