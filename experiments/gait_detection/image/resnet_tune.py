"""Image pipeline — full ResNet fine-tuning with Optuna hyperparameter search.

Trains the complete ResNet backbone + linear head (all weights unfrozen) on
cropped video frames.  Differential learning rates allow the backbone to be
updated more conservatively than the head.

Objective: minimise validation cross-entropy loss on the fixed tuning validation split.

Usage
-----
    python -m experiments.gait_detection.image.resnet_tune \
        --config configs/experiments/image/resnet_tune.yaml

Output
------
    <output_dir>/best_params.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import mlflow
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.gait_detection.config import ExperimentConfig, get_split_config
from src.gait.detection.train import get_device, seed_everything
from src.gait.gait_data.dataset import train_test_split, tuning_split
from src.gait.image.dataset import load_image_records_for_finetune
from src.gait.image.finetune import FrameCropDataset, ResNetFinetune
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    t = spec["type"]
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown search space type '{t}' for parameter '{name}'")


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


def objective(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
    backbone_name: str,
    search_space: dict,
) -> float:
    params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}

    device = get_device()
    model  = ResNetFinetune(
        backbone_name=backbone_name,
        n_classes=cfg.n_classes,
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
    criterion = nn.CrossEntropyLoss()

    best_val_loss    = float("inf")
    best_epoch       = 1
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        _train_epoch(model, train_loader, optimizer, criterion, device, cfg.max_grad_norm)
        loss = _val_loss(model, val_loader, device)

        trial.report(loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if loss < best_val_loss:
            best_val_loss    = loss
            best_epoch       = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                break

    trial.set_user_attr("best_epoch", best_epoch)

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch",    best_epoch)

    return best_val_loss


def main(
    cfg: ExperimentConfig,
    n_trials: int,
    search_space: dict,
    backbone_name: str,
    pose_dir: str,
    video_input_dir: str,
    img_size: int,
    bbox_padding: float,
    n_test: dict,
    seed: int,
) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_image_resnet_tuning")

    logger.info("Loading image records for fine-tuning …")
    all_records_info = load_image_records_for_finetune(
        cfg.annotations_csv, pose_dir, video_input_dir,
        n_trim_padding=cfg.n_trim_padding,
    )
    all_records = [r for r, _, _ in all_records_info]
    logger.info("%d records loaded.", len(all_records))

    records, _, test_athletes = train_test_split(all_records, n_test=n_test, seed=seed)
    test_set = set(test_athletes)
    train_info = [t for t in all_records_info if t[0].athlete not in test_set]
    logger.info("Test athletes excluded: %s", test_athletes)

    train_records = [r for r, _, _ in train_info]
    train_info_tuning, val_info = _split_records_info(train_info, cfg, records)

    logger.info(
        "Train: %d records  Val: %d records",
        len(train_info_tuning), len(val_info),
    )

    logger.info("Loading crops into RAM (train) …")
    train_ds = FrameCropDataset(train_info_tuning, img_size=img_size, bbox_padding=bbox_padding)
    logger.info("Loading crops into RAM (val) …")
    val_ds   = FrameCropDataset(val_info,          img_size=img_size, bbox_padding=bbox_padding)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    with mlflow.start_run(run_name="optuna_study"):
        mlflow.log_params({
            "n_trials":       n_trials,
            "n_val_athletes": cfg.n_val_athletes_tuning,
            "backbone_name":  backbone_name,
        })

        study.optimize(
            lambda trial: objective(
                trial, train_loader, val_loader, cfg, backbone_name, search_space
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_epoch  = study.best_trial.user_attrs.get("best_epoch", cfg.max_epochs)
        logger.info("Best val loss : %.4f  (epoch %d)", study.best_value, best_epoch)
        logger.info("Best params   : %s", best_params)

        mlflow.log_metric("best_val_loss", study.best_value)
        mlflow.log_metric("best_epoch",    best_epoch)
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

    os.makedirs(cfg.output_dir, exist_ok=True)
    out = {
        "best_params":   best_params,
        "best_val_loss": study.best_value,
        "best_epoch":    best_epoch,
        "backbone_name": backbone_name,
        "img_size":      img_size,
        "bbox_padding":  bbox_padding,
    }
    out_path = cfg.results_path("best_params")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Params saved → %s", out_path)


def _split_records_info(
    records_info: list,
    cfg: ExperimentConfig,
    all_train_records: list,
) -> tuple[list, list]:
    """Apply tuning_split athlete logic to records_info tuples."""
    _, val_records = tuning_split(all_train_records, cfg.n_val_athletes_tuning, cfg.random_seed)
    val_athletes   = {r.athlete for r in val_records}
    train_info = [t for t in records_info if t[0].athlete not in val_athletes]
    val_info   = [t for t in records_info if t[0].athlete in val_athletes]
    return train_info, val_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg             = ExperimentConfig(**raw.get("experiment", {}))
    n_trials        = raw.get("optuna", {}).get("n_trials", 50)
    search_space    = raw.get("search_space", {})
    backbone_name   = raw.get("backbone_name", "resnet18")
    pose_dir        = raw["pose_dir"]
    video_input_dir = raw.get("video_input_dir", "data/input/optojump")
    img_size        = raw.get("img_size", 224)
    bbox_padding    = raw.get("bbox_padding", 0.1)

    if not search_space:
        raise ValueError("'search_space' section is missing or empty in the config.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    split_cfg = get_split_config(cfg.dataset_config)
    n_test    = split_cfg["n_test"]
    seed      = split_cfg.get("seed", 42)

    main(cfg, n_trials, search_space, backbone_name, pose_dir, video_input_dir, img_size, bbox_padding,
         n_test=n_test, seed=seed)
