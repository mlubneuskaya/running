"""Image pipeline — ResNet leave-one-athlete-out cross-validation.

Uses best hyperparameters from resnet_tune stage.  Trains one ResNet model
per fold with early stopping, records best epoch per fold, and saves
per-frame softmax probabilities for the held-out records.

Also computes ``avg_best_epoch`` across folds — used by resnet_train to
decide how many epochs to train the final model.

Usage
-----
    python -m experiments.gait_detection.image.resnet_loao \
        --config configs/experiments/image/resnet_loao.yaml

Output
------
    <output_dir>/manifest.json    — per-record probs paths (train records only)
    <output_dir>/probs/           — .npy arrays, shape (T, n_classes) float32
    <output_dir>/loao.json        — F1 + confusion matrix + avg_best_epoch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experiments.gait_detection.config import ExperimentConfig, get_split_config
from src.gait.detection.metrics import (
    aggregate_confusion_matrices,
    confusion_matrix,
    per_class_f1,
)
from src.gait.detection.train import get_device, seed_everything

from src.gait.gait_data.dataset import loao_splits, train_test_split
from src.gait.image.dataset import load_image_records_for_finetune
from src.gait.image.finetune import FrameCropDataset, ResNetFinetune
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)


def _safe_stem(video_path: str) -> str:
    stem = os.path.splitext(video_path)[0]
    return re.sub(r"[/\\]", "_", stem).lstrip("_")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def _val_loss(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    for X, y in val_loader:
        X, y = X.to(device), y.to(device)
        total += criterion(model(X), y).item()
    return total / max(len(val_loader), 1)


@torch.no_grad()
def _get_probs(
    model: nn.Module,
    rec_info: tuple,
    img_size: int,
    bbox_padding: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    ds     = FrameCropDataset([rec_info], img_size=img_size, bbox_padding=bbox_padding)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    parts  = []
    for X, _ in loader:
        parts.append(F.softmax(model(X.to(device)), dim=-1).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _build_model(params: dict, backbone_name: str, n_classes: int, device: torch.device) -> tuple:
    model = ResNetFinetune(
        backbone_name=backbone_name,
        n_classes=n_classes,
        dropout=params["dropout"],
    ).to(device)
    backbone_lr = params["lr"] * params["backbone_lr_factor"]
    head_params = [*model.drop.parameters(), *model.fc.parameters()]
    optimizer = torch.optim.Adam(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": head_params,                 "lr": params["lr"]},
        ],
        weight_decay=params["weight_decay"],
    )
    return model, optimizer


def run_fold(
    train_info: list,
    val_info: list,
    cfg: ExperimentConfig,
    params: dict,
    backbone_name: str,
    img_size: int,
    bbox_padding: float,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, list]:
    """Train one fold; return (train_result, y_true, y_pred, val_probs).

    Returns
    -------
    train_result : dict with best_epoch and best_val_f1
    y_true       : np.ndarray concatenated labels from val_info
    y_pred       : np.ndarray argmax predictions
    val_probs    : list[np.ndarray] — per-record probs (T, n_classes)
    """
    train_ds = FrameCropDataset(train_info, img_size=img_size, bbox_padding=bbox_padding)
    val_ds   = FrameCropDataset(val_info,   img_size=img_size, bbox_padding=bbox_padding)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    model, optimizer = _build_model(params, backbone_name, cfg.n_classes, device)

    best_val_loss    = float("inf")
    best_epoch       = 1
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        _train_epoch(model, train_loader, optimizer, criterion, device, cfg.max_grad_norm)
        loss = _val_loss(model, val_loader, device)

        if loss < best_val_loss:
            best_val_loss    = loss
            best_epoch       = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                break

        if epoch % 10 == 0:
            logger.info("    Epoch %3d  val_loss=%.4f  best=%d", epoch, loss, best_epoch)

    val_probs = [
        _get_probs(model, rec_info, img_size, bbox_padding, cfg.batch_size, device)
        for rec_info in val_info
    ]

    y_true = np.concatenate([rec_info[0].labels for rec_info in val_info])
    y_pred = np.concatenate([p.argmax(axis=1) for p in val_probs])

    train_result = {"best_epoch": best_epoch, "best_val_loss": float(best_val_loss)}
    return train_result, y_true, y_pred, val_probs


def main(
    cfg: ExperimentConfig,
    params: dict,
    backbone_name: str,
    img_size: int,
    bbox_padding: float,
    pose_dir: str,
    video_input_dir: str,
    output_dir: str,
    n_test: dict,
    seed: int,
) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_image_resnet_loao")

    probs_dir = os.path.join(output_dir, "probs")
    os.makedirs(probs_dir, exist_ok=True)

    logger.info("Loading image records for fine-tuning …")
    all_records_info = load_image_records_for_finetune(
        cfg.annotations_csv, pose_dir, video_input_dir
    )
    all_records = [r for r, _, _ in all_records_info]
    logger.info("%d records loaded.", len(all_records))

    _, _, test_athletes = train_test_split(all_records, n_test=n_test, seed=seed)
    test_set = set(test_athletes)
    train_info = [t for t in all_records_info if t[0].athlete not in test_set]
    train_records = [t[0] for t in train_info]
    logger.info("Test athletes excluded: %s  (%d train records)", test_athletes, len(train_info))

    info_by_video = {t[0].video_path: t for t in train_info}

    n_athletes = len({r.athlete for r in train_records})
    device = get_device()
    fold_results = []
    all_cms = []
    manifest_records = []
    best_epochs = []

    with mlflow.start_run(run_name="resnet_loao"):
        mlflow.log_params(params)
        mlflow.log_params({"backbone_name": backbone_name})

        for fold_i, (fold_train_recs, fold_val_recs, athlete) in enumerate(
            loao_splits(train_records), 1
        ):
            logger.info(
                "[Fold %d/%d] Held out: %s (%d videos)",
                fold_i, n_athletes, athlete, len(fold_val_recs),
            )

            fold_train_info = [info_by_video[r.video_path] for r in fold_train_recs]
            fold_val_info   = [info_by_video[r.video_path] for r in fold_val_recs]

            train_result, y_true, y_pred, val_probs = run_fold(
                fold_train_info, fold_val_info, cfg, params,
                backbone_name, img_size, bbox_padding, device,
            )
            best_epochs.append(train_result["best_epoch"])

            for rec_info, probs in zip(fold_val_info, val_probs):
                rec      = rec_info[0]
                out_name = _safe_stem(rec.video_path) + ".npy"
                out_path = os.path.join(probs_dir, out_name)
                np.save(out_path, probs)
                manifest_records.append({
                    "video_path": rec.video_path,
                    "athlete":    rec.athlete,
                    "split":      "train",
                    "n_frames":   int(probs.shape[0]),
                    "probs_path": out_path,
                })

            f1 = per_class_f1(y_true, y_pred, n_classes=cfg.n_classes, class_names=cfg.class_names)
            cm = confusion_matrix(y_true, y_pred, n_classes=cfg.n_classes)
            all_cms.append(cm)

            fold_result = {
                "athlete":          athlete,
                "n_val_videos":     len(fold_val_recs),
                "f1":               f1,
                "confusion_matrix": cm.tolist(),
                "training":         train_result,
            }
            fold_results.append(fold_result)

            with mlflow.start_run(run_name=f"fold_{athlete}", nested=True):
                mlflow.log_metric("macro_f1",     f1["macro"])
                mlflow.log_metric("best_epoch",   train_result["best_epoch"])
                mlflow.log_metric("best_val_loss", train_result["best_val_loss"])
                for cls in cfg.class_names:
                    mlflow.log_metric(f"f1_{cls}", f1[cls])

            logger.info(
                "  best_epoch=%d  val_loss=%.4f  F1 macro=%.3f  left=%.3f  right=%.3f  flight=%.3f",
                train_result["best_epoch"], train_result["best_val_loss"], f1["macro"],
                f1["left_stance"], f1["right_stance"], f1["flight"],
            )

        macro_f1s = [r["f1"]["macro"] for r in fold_results]
        total_cm  = aggregate_confusion_matrices(all_cms)
        avg_best_epoch = int(round(float(np.mean(best_epochs))))

        loao_summary = {
            "best_params":            params,
            "backbone_name":          backbone_name,
            "n_classes":              cfg.n_classes,
            "class_names":            cfg.class_names,
            "n_folds":                len(fold_results),
            "avg_best_epoch":         avg_best_epoch,
            "macro_f1_mean":          float(np.mean(macro_f1s)),
            "macro_f1_std":           float(np.std(macro_f1s)),
            "per_class_f1_mean": {
                cls: float(np.mean([r["f1"][cls] for r in fold_results]))
                for cls in cfg.class_names
            },
            "total_confusion_matrix": total_cm.tolist(),
            "folds":                  fold_results,
        }

        mlflow.log_metric("macro_f1_mean",  loao_summary["macro_f1_mean"])
        mlflow.log_metric("macro_f1_std",   loao_summary["macro_f1_std"])
        mlflow.log_metric("avg_best_epoch", avg_best_epoch)

    loao_path = os.path.join(output_dir, "loao.json")
    with open(loao_path, "w") as f:
        json.dump(loao_summary, f, indent=2)
    logger.info("LOAO results saved → %s", loao_path)

    manifest = {
        "best_params":   params,
        "backbone_name": backbone_name,
        "img_size":      img_size,
        "bbox_padding":  bbox_padding,
        "n_classes":     cfg.n_classes,
        "class_names":   cfg.class_names,
        "fps":           cfg.fps,
        "test_athletes": list(test_athletes),
        "records":       manifest_records,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved → %s", manifest_path)

    logger.info(
        "LOAO done  macro_F1=%.3f ± %.3f  avg_best_epoch=%d",
        loao_summary["macro_f1_mean"], loao_summary["macro_f1_std"], avg_best_epoch,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg              = ExperimentConfig(**raw.get("experiment", {}))
    best_params_json = raw["best_params_json"]
    pose_dir         = raw["pose_dir"]
    video_input_dir  = raw.get("video_input_dir", "data/input/optojump")
    output_dir       = raw.get("output_dir", "data/output/gait/image/resnet/loao")
    backbone_name    = raw.get("backbone_name", "resnet18")
    img_size         = raw.get("img_size", 224)
    bbox_padding     = raw.get("bbox_padding", 0.1)

    with open(best_params_json) as _f:
        _bp = json.load(_f)
    params        = _bp["best_params"]
    backbone_name = _bp.get("backbone_name", backbone_name)
    img_size      = _bp.get("img_size",      img_size)
    bbox_padding  = _bp.get("bbox_padding",  bbox_padding)

    logger.info("Loaded best params from %s: %s", best_params_json, params)
    split_cfg = get_split_config(cfg.dataset_config)
    n_test    = split_cfg["n_test"]
    seed      = split_cfg.get("seed", 42)

    main(cfg, params, backbone_name, img_size, bbox_padding, pose_dir, video_input_dir, output_dir,
         n_test=n_test, seed=seed)
