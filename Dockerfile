FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# System deps: Python 3.11, OpenCV headless, video codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        libgl1 libglib2.0-0 libgomp1 \
        ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.11 /usr/bin/python3 \
    && ln -sf python3 /usr/bin/python

WORKDIR /app

# Install Python deps before copying source so this layer is cached
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-docker.txt

# Copy source (data/ is excluded via .dockerignore and mounted at runtime)
COPY configs/     configs/
COPY experiments/ experiments/
COPY scripts/     scripts/
COPY src/         src/
COPY tests/       tests/
COPY Makefile     Makefile
COPY dvc.yaml     dvc.yaml

# Smoke-test that imports work — fails the build if something is broken
RUN python -c "import torch, torchvision, ultralytics, optuna, mlflow, dvc; \
               from src.gait.detection.model import TCN; \
               from src.gait.image.extractor import ImageFeatureExtractor; \
               print('All imports OK')"

CMD ["python", "-m", "experiments.gait_detection.tcn_tuning", \
     "--config", "configs/experiments/tcn_tuning.yaml"]
