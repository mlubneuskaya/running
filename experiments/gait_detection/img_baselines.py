"""Image pipeline Stage 1 — per-frame MLP baseline.

Trains a frame-level MLP (no temporal context) on pre-extracted CNN features
to validate that the backbone produces class-discriminative representations
before investing in the full TCN pipeline.

A macro F1 above ~0.70 indicates the features are usable.  Below that, try a
different backbone, a deeper feature layer, or fine-tuning.

Usage
-----
    python -m experiments.gait_detection.img_baselines
    python -m experiments.gait_detection.img_baselines --config configs/experiments/img_baselines.yaml

Output
------
    <output_dir>/img_baselines.json
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mlflow

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.metrics import per_class_f1, confusion_matrix
from src.gait.detection.train import get_device
from src.gait.gait_data.dataset import (
    compute_class_weights,
    train_test_split,
    tuning_split,
)
from src.gait.image.dataset import load_image_dataset
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/experiments/img_baselines.yaml"


def _flatten(records) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all frame features and labels across records."""
    feats  = np.concatenate([r.features for r in records], axis=0)
    labels = np.concatenate([r.labels   for r in records], axis=0)
    return feats.astype(np.float32), labels.astype(np.int64)


def _build_mlp(input_dim: int, n_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes),
    )


def train_mlp(
    train_records,
    val_records,
    cfg: ExperimentConfig,
    device: torch.device,
    max_epochs: int,
    batch_size: int,
) -> tuple[nn.Module, dict]:
    X_train, y_train = _flatten(train_records)
    X_val,   y_val   = _flatten(val_records)

    n_features = X_train.shape[1]
    model = _build_mlp(n_features, cfg.n_classes).to(device)

    class_weights = compute_class_weights(train_records, cfg.n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_epoch    = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss(weight=class_weights)(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            xv = torch.from_numpy(X_val).to(device)
            yv = torch.from_numpy(y_val).to(device)
            val_loss = criterion(model(xv), yv).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch

        if epoch % 20 == 0:
            logger.info("Epoch %3d  val_loss=%.4f", epoch, val_loss)

    # Evaluate
    model.eval()
    with torch.no_grad():
        xv = torch.from_numpy(X_val).to(device)
        logits = model(xv)
        y_pred = logits.argmax(dim=-1).cpu().numpy()

    f1 = per_class_f1(y_val, y_pred, n_classes=cfg.n_classes, class_names=cfg.class_names)
    cm = confusion_matrix(y_val, y_pred, n_classes=cfg.n_classes)

    metrics = {
        "best_val_loss": best_val_loss,
        "best_epoch":    best_epoch,
        "f1":            f1,
        "confusion_matrix": cm.tolist(),
    }
    return model, metrics


def main(cfg: ExperimentConfig, features_dir: str, video_input_dir: str, max_epochs: int, batch_size: int) -> None:
    mlflow.set_experiment("gait_image_baselines")

    logger.info("Loading image dataset …")
    all_records = load_image_dataset(cfg.annotations_csv, features_dir, video_input_dir)
    logger.info("%d records loaded.", len(all_records))

    records, _, test_athletes = train_test_split(all_records)
    logger.info("Test athletes excluded: %s", test_athletes)
    train_records, val_records = tuning_split(records, cfg.n_val_athletes_tuning, cfg.tuning_seed)
    logger.info("Train: %d records  Val: %d records", len(train_records), len(val_records))

    device = get_device()

    with mlflow.start_run(run_name="img_mlp_baseline"):
        mlflow.log_params({
            "features_dir": features_dir,
            "max_epochs":   max_epochs,
            "batch_size":   batch_size,
            "lr":           cfg.lr,
            "n_val_athletes": cfg.n_val_athletes_tuning,
        })

        _, metrics = train_mlp(train_records, val_records, cfg, device, max_epochs, batch_size)

        mlflow.log_metric("macro_f1",   metrics["f1"]["macro"])
        mlflow.log_metric("val_loss",   metrics["best_val_loss"])
        for cls in cfg.class_names:
            mlflow.log_metric(f"f1_{cls}", metrics["f1"][cls])

    logger.info(
        "MLP baseline  macro_F1=%.3f  left=%.3f  right=%.3f  flight=%.3f",
        metrics["f1"]["macro"],
        metrics["f1"]["left_stance"],
        metrics["f1"]["right_stance"],
        metrics["f1"]["flight"],
    )

    out = {
        "n_train_records": len(train_records),
        "n_val_records":   len(val_records),
        "test_athletes":   test_athletes,
        "metrics":         metrics,
    }
    out_path = cfg.results_path("img_baselines")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg = ExperimentConfig(**raw.get("experiment", {}))
    features_dir    = raw["features_dir"]
    video_input_dir = raw.get("video_input_dir", "data/input/optojump")
    max_epochs      = raw.get("max_epochs", 50)
    batch_size      = raw.get("batch_size", 256)

    main(cfg, features_dir, video_input_dir, max_epochs, batch_size)
