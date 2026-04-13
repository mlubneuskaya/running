"""Final training — train one model per selected Optuna trial on the full dataset.

Loads hyperparameters directly from the Optuna study for each specified trial ID,
trains on all available data, and saves a checkpoint per trial.  Intended to run
after leave-one-out CV has identified the best-performing trial configurations.

Usage
-----
    python -m experiments.gait_detection.tcn_training
    python -m experiments.gait_detection.tcn_training --config configs/experiments/tcn_training.yaml

Output
------
    <checkpoint_dir>/trial_<id>.pt   — one checkpoint per trial
    <output_dir>/tcn_training.json   — per-trial training summary
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.gait.detection.dilations as dilation_schedules
from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.model import TCN
from src.gait.detection.train import train_epoch, get_device
from src.gait.gait_data.dataset import load_dataset, compute_class_weights, GaitWindowDataset
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/experiments/tcn_training.yaml"

_REQUIRED_PARAMS = {"lr", "dropout", "n_blocks", "n_filters", "kernel_size", "dilation_schedule"}


def _load_study(study_cfg: dict, experiment_cfg: ExperimentConfig) -> optuna.Study:
    storage_type = study_cfg.get("storage_type", "journal")
    storage_path = os.path.expandvars(study_cfg.get("log", experiment_cfg.optuna_storage))
    study_name   = study_cfg.get("name", experiment_cfg.study_name)

    if storage_type == "journal":
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(storage_path)
        )
    else:
        storage = storage_path

    return optuna.load_study(study_name=study_name, storage=storage)


def _params_from_trial(trial: optuna.trial.FrozenTrial) -> dict:
    params = trial.params
    missing = _REQUIRED_PARAMS - params.keys()
    if missing:
        raise KeyError(
            f"Trial #{trial.number} is missing required parameters: {sorted(missing)}"
        )
    return params


def train_trial(
    trial_id: int,
    params: dict,
    records,
    cfg: ExperimentConfig,
    max_epochs: int,
    log_every: int,
) -> dict:
    n_features = len(cfg.feature_idx) if cfg.feature_idx is not None else 22
    dilations  = getattr(dilation_schedules, params["dilation_schedule"])(params["n_blocks"])

    model = TCN(
        n_features=n_features,
        n_blocks=params["n_blocks"],
        n_filters=params["n_filters"],
        kernel_size=params["kernel_size"],
        dropout=params["dropout"],
        dilations=dilations,
    )

    device        = get_device()
    model         = model.to(device)
    class_weights = compute_class_weights(records, n_classes=cfg.n_classes).to(device)
    criterion     = nn.NLLLoss(weight=class_weights)
    optimizer     = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_schedule_factor, patience=cfg.lr_schedule_patience,
    )

    dataset = GaitWindowDataset(records, window_size=cfg.window_size, feature_idx=cfg.feature_idx)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    train_losses: list[float] = []

    for epoch in range(1, max_epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion, device, cfg.max_grad_norm)
        train_losses.append(loss)
        scheduler.step(loss)
        if epoch % log_every == 0:
            logger.info("Trial %d  Epoch %3d/%d  train=%.4f", trial_id, epoch, max_epochs, loss)

    ckpt_path = cfg.checkpoint_path(f"trial_{trial_id}")
    torch.save(model.state_dict(), ckpt_path)
    logger.info("Trial %d  saved → %s", trial_id, ckpt_path)

    return {
        "trial_id":     trial_id,
        "params":       params,
        "epochs":       max_epochs,
        "final_loss":   train_losses[-1],
        "train_losses": train_losses,
        "checkpoint":   ckpt_path,
    }


def main(
    cfg: ExperimentConfig,
    study_cfg: dict,
    trial_ids: list[int],
    max_epochs: int,
    log_every: int = 10,
) -> None:
    logger.info("Loading dataset …")
    records = load_dataset(cfg.annotations_csv, fps=cfg.fps)
    logger.info("%d videos loaded.", len(records))

    logger.info("Loading Optuna study …")
    study = _load_study(study_cfg, cfg)
    trials_by_id = {t.number: t for t in study.trials}

    results = []
    for trial_id in trial_ids:
        if trial_id not in trials_by_id:
            raise KeyError(f"Trial #{trial_id} not found in study '{study.study_name}'.")
        trial  = trials_by_id[trial_id]
        params = _params_from_trial(trial)
        logger.info(
            "Training trial %d  (val_loss=%.4f  dilation=%s  n_blocks=%d  n_filters=%d  kernel=%d)",
            trial_id, trial.value, params["dilation_schedule"],
            params["n_blocks"], params["n_filters"], params["kernel_size"],
        )
        result = train_trial(trial_id, params, records, cfg, max_epochs, log_every)
        results.append(result)

    out = {
        "max_epochs": max_epochs,
        "n_models":   len(results),
        "models":     [{k: v for k, v in r.items() if k != "train_losses"} for r in results],
    }
    out_path = cfg.results_path("tcn_training")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Summary saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw = load_config(args.config)

    cfg        = ExperimentConfig(**raw.get("experiment", {}))
    study_cfg  = raw.get("study", {})
    trial_ids  = raw.get("trial_ids", [])
    max_epochs = raw.get("max_epochs", 300)
    log_every  = raw.get("log_every", 10)

    if not trial_ids:
        raise ValueError("'trial_ids' is empty or missing in the config.")
    if not study_cfg:
        raise ValueError("'study' section is missing in the config.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main(cfg, study_cfg, trial_ids, max_epochs, log_every)
