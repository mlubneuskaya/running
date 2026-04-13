export PYTHONPATH := $(shell pwd)
PYTHON := python

# ── pose pipeline configs ─────────────────────────────────────────────────────
CONFIG_YOLO              := ./configs/config_yolo.yaml
CONFIG_MP                := ./configs/config_mediapipe.yaml
CONFIG_OVERLAY_YOLO      := ./configs/config_overlay_yolo.yaml
CONFIG_OVERLAY_MEDIAPIPE := ./configs/config_overlay_mediapipe.yaml

# ── gait detection experiment configs ────────────────────────────────────────
CONFIG_STAGE1   := ./configs/experiments/baselines.yaml
CONFIG_STAGE2   := ./configs/experiments/tcn_tuning.yaml
CONFIG_STAGE3   := ./configs/experiments/xgb_tuning.yaml
CONFIG_STAGE4   := ./configs/experiments/tcn_leave_one_out.yaml
CONFIG_STAGE5    := ./configs/experiments/tcn_training.yaml

YOLO_OUTPUT_DIR := ./data/output/yolo/overlays
MEDIAPIPE_OUTPUT_DIR := ./data/output/mediapipe/overlays

.PHONY: help setup process-yolo process-mp overlay visualize all clean \
        stage1 stage2 stage4 xgb-tune train gait-all

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Pose pipeline:"
	@echo "    make setup          - Install required python packages"
	@echo "    make process-yolo   - Run YOLOv8 processing (Video -> JSON)"
	@echo "    make process-mp     - Run MediaPipe processing (Video -> JSON)"
	@echo "    make overlay        - Generate stick figure videos (JSON + Video -> Overlay)"
	@echo "    make visualize      - Generate biomechanics graphs (JSON -> PNG)"
	@echo "    make all            - Run YOLO processing, Overlay, and Visualization sequentially"
	@echo ""
	@echo "  Gait detection experiments:"
	@echo "    make stage1         - Kinematic baseline + XGBoost ablation"
	@echo "    make stage2         - TCN hyperparameter search (Optuna)"
	@echo "    make stage4         - Full LOAO cross-validation"
	@echo "    make train          - Final training on selected Optuna trials"
	@echo "    make xgb-tune       - XGBoost hyperparameter tuning (Optuna)"
	@echo "    make gait-all       - Run stage1 → stage2 → stage4 sequentially"
	@echo ""
	@echo "    make clean          - Remove pycache and temporary files"

setup:
	@echo "Installing requirements..."
	pip install -r requirements.txt

process-yolo:
	@echo "Starting YOLO Processing..."
	$(PYTHON) ./scripts/run_pose_detection.py --config $(CONFIG_YOLO)

process-mediapipe:
	@echo "Starting MediaPipe Processing..."
	$(PYTHON) ./scripts/run_pose_detection.py --config $(CONFIG_MP)

overlay-yolo:
	@echo "Generating Overlay Videos..."
	$(PYTHON) ./scripts/run_overlay.py --config $(CONFIG_OVERLAY_YOLO)

overlay-mediapipe:
	@echo "Generating Overlay Videos..."
	$(PYTHON) ./scripts/run_overlay.py --config $(CONFIG_OVERLAY_MEDIAPIPE)

yolo: process-yolo overlay-yolo
	@echo "Pipeline complete! Results saved to ./data/output"

mediapipe: process-mediapipe overlay-mediapipe
	@echo "Pipeline complete! Results saved to ./data/output"

# ── gait detection experiments ────────────────────────────────────────────────

stage1:
	@echo "Running Stage 1 — kinematic baseline + XGBoost ablation..."
	$(PYTHON) -m experiments.gait_detection.baselines --config $(CONFIG_STAGE1)

stage2:
	@echo "Running Stage 2 — TCN hyperparameter search (Optuna)..."
	$(PYTHON) -m experiments.gait_detection.tcn_tuning --config $(CONFIG_STAGE2)

stage3:
	@echo "Running XGBoost hyperparameter tuning..."
	$(PYTHON) -m experiments.gait_detection.xgb_tuning --config $(CONFIG_STAGE3)

stage4:
	@echo "Running Stage 4 — LOAO cross-validation..."
	$(PYTHON) -m experiments.gait_detection.tcn_leave_one_out --config $(CONFIG_STAGE4)

stage5:
	@echo "Running final training on selected trials..."
	$(PYTHON) -m experiments.gait_detection.tcn_training --config $(CONFIG_STAGE5)

gait-all: stage1 stage2 stage3 stage4 stage5
	@echo "All gait detection experiments complete."

# ── housekeeping ──────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete