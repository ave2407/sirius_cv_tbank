FROM python:3.11.9-slim-bookworm

# Больше стабильности при сборке и меньше сюрпризов от pip
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OMP_WAIT_POLICY=PASSIVE

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary --no-cache-dir -r requirements.txt

RUN mkdir -p /models
COPY weights/best.onnx /models/best.onnx

COPY server /app/server
EXPOSE 8000

# Обновим путь к весам по умолчанию
ENV MODEL_PATH=/models/best.onnx \
    CONF_THRES=0.55 \
    IOU_THRES=0.50 \
    IMG_SIZE=960 \
    MAX_DET=300 \
    ORT_INTRA_THREADS=4 \
    ORT_INTER_THREADS=1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]