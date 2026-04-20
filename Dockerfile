FROM --platform=linux/amd64 nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-dev \
        python3.13-venv \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.13 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.13 /usr/bin/python

RUN wget -qO get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --ignore-installed --no-cache-dir -r requirements.txt

COPY configs/     configs/
COPY experiments/ experiments/
COPY scripts/     scripts/
COPY src/         src/
COPY tests/       tests/
COPY Makefile     Makefile
COPY dvc.yaml     dvc.yaml
