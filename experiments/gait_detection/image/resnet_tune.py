"""Image pipeline — finetune linear head on ResNet features (Optuna tuning).

Tunes a linear classifier (dropout + fc) on top of frozen ResNet features.
Objective: maximize macro F1 on the fixed tuning validation split.

Usage
-----
    python -m experiments.gait_detection.image.resnet_tune --config configs/experiments/image/resnet_tune.yaml

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
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.metrics import per_class_f1
from src.gait.detection.train import get_device, seed_everything
from src.gait.gait_data.dataset import train_test_split, tuning_split
from src.gait.image.dataset import load_image_dataset
from src.gait.image.finetune import GaitFrameDataset, LinearHead
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
def _val_f1(
    model: nn.Module,
    records: list,
    device: torch.device,
    n_classes: int,
    class_names: list[str],
) -> float:
    model.eval()
    y_true_all, y_pred_all = [], []
    for rec in records:
        X = torch.from_numpy(rec.features).float().to(device)
        y_pred_all.append(model(X).argmax(dim=-1).cpu().numpy())
        y_true_all.append(rec.labels)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return per_class_f1(y_true, y_pred, n_classes=n_classes, class_names=class_names)["macro"]


def objective(
    trial: optuna.Trial,
    train_records: list,
    val_records: list,
    cfg: ExperimentConfig,
    n_features: int,
    search_space: dict,
) -> float:
    params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}

    device = get_device()
    model  = LinearHead(n_features, n_classes=cfg.n_classes, dropout=params["dropout"]).to(device)
    opt    = torch.optim.Adam(
        model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
    )
    crit   = nn.NLLLoss()

    ds     = GaitFrameDataset(train_records)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    best_f1          = 0.0
    best_epoch       = 0
    patience_counter = 0

    for epoch in range(1, cfg.max_epochs + 1):
        _train_epoch(model, loader, opt, crit, device, cfg.max_grad_norm)
        f1 = _val_f1(model, val_records, device, cfg.n_classes, cfg.class_names)

        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if f1 > best_f1:
            best_f1    = f1
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                break

    trial.set_user_attr("best_epoch", best_epoch)

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("best_val_macro_f1", best_f1)
        mlflow.log_metric("best_epoch",        best_epoch)

    return best_f1


def main(
    cfg: ExperimentConfig,
    n_trials: int,
    search_space: dict,
    features_dir: str,
    video_input_dir: str,
) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_image_resnet_tuning")

    logger.info("Loading image dataset …")
    all_records = load_image_dataset(cfg.annotations_csv, features_dir, video_input_dir)
    logger.info("%d records loaded.", len(all_records))

    records, _, test_athletes = train_test_split(all_records)
    logger.info("Test athletes excluded: %s", test_athletes)
    train_records, val_records = tuning_split(records, cfg.n_val_athletes_tuning, cfg.random_seed)
    logger.info("Train: %d records  Val: %d records", len(train_records), len(val_records))

    n_features = train_records[0].features.shape[1]
    logger.info("n_features: %d", n_features)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    with mlflow.start_run(run_name="optuna_study"):
        mlflow.log_params({
            "n_trials":       n_trials,
            "n_val_athletes": cfg.n_val_athletes_tuning,
            "n_features":     n_features,
        })

        study.optimize(
            lambda trial: objective(trial, train_records, val_records, cfg, n_features, search_space),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_params = study.best_params
        best_epoch  = study.best_trial.user_attrs.get("best_epoch", cfg.max_epochs)
        logger.info("Best macro F1 : %.4f  (epoch %d)", study.best_value, best_epoch)
        logger.info("Best params   : %s", best_params)

        mlflow.log_metric("best_val_macro_f1", study.best_value)
        mlflow.log_metric("best_epoch",        best_epoch)
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

    os.makedirs(cfg.output_dir, exist_ok=True)
    out = {
        "best_params":       best_params,
        "best_val_macro_f1": study.best_value,
        "best_epoch":        best_epoch,
        "n_features":        n_features,
    }
    out_path = cfg.results_path("best_params")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Params saved → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg             = ExperimentConfig(**raw.get("experiment", {}))
    n_trials        = raw.get("optuna", {}).get("n_trials", 50)
    search_space    = raw.get("search_space", {})
    features_dir    = raw["features_dir"]
    video_input_dir = raw.get("video_input_dir", "data/input/optojump")

    if not search_space:
        raise ValueError("'search_space' section is missing or empty in the config.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main(cfg, n_trials, search_space, features_dir, video_input_dir)
