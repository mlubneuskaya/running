export PYTHONPATH := $(shell pwd)
PYTHON := python

CONFIG_YOLO    := ./configs/config_yolo.yaml
CONFIG_MP      := ./configs/config_mediapipe.yaml
CONFIG_OVERLAY_YOLO := ./configs/config_overlay_yolo.yaml
CONFIG_OVERLAY_MEDIAPIPE := ./configs/config_overlay_mediapipe.yaml

YOLO_OUTPUT_DIR := ./data/output/yolo/overlays
MEDIAPIPE_OUTPUT_DIR := ./data/output/mediapipe/overlays

.PHONY: help setup process-yolo process-mp overlay visualize all clean

help:
	@echo "Available commands:"
	@echo "  make setup          - Install required python packages"
	@echo "  make process-yolo   - Run YOLOv8 processing (Video -> JSON)"
	@echo "  make process-mp     - Run MediaPipe processing (Video -> JSON)"
	@echo "  make overlay        - Generate stick figure videos (JSON + Video -> Overlay)"
	@echo "  make visualize      - Generate biomechanics graphs (JSON -> PNG)"
	@echo "  make all            - Run YOLO processing, Overlay, and Visualization sequentially"
	@echo "  make clean          - Remove pycache and temporary files"

setup:
	@echo "Installing requirements..."
	pip install -r requirements.txt

process-yolo:
	@echo "Starting YOLO Processing..."
	$(PYTHON) ./scripts/run_pose_detection.py --config $(CONFIG_YOLO)

process-mp:
	@echo "Starting MediaPipe Processing..."
	$(PYTHON) ./scripts/run_pose_detection.py --config $(CONFIG_MP)

overlay-yolo:
	@echo "Generating Overlay Videos..."
	$(PYTHON) ./scripts/run_overlay.py --config $(CONFIG_OVERLAY_YOLO)

overlay-mediapipe:
	@echo "Generating Overlay Videos..."
	$(PYTHON) ./scripts/run_overlay.py --config $(CONFIG_OVERLAY_MEDIAPIPE)

all: process-yolo overlay-yolo
	@echo "Pipeline complete! Results saved to ./data/output"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete