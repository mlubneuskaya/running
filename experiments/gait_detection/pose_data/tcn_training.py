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

import mlflow
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.gait.detection.dilations as dilation_schedules
from experiments.gait_detection.config import ExperimentConfig, get_split_config
from experiments.gait_detection.study_utils import load_study, params_from_trial
from src.gait.detection.model import TCN
from src.gait.detection.train import train_epoch, get_device, seed_everything
from src.gait.gait_data.dataset import load_dataset, compute_class_weights, GaitWindowDataset, train_test_split
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)



def epochs_from_loao(trial_id: int, loao_dir: str) -> int:
    """Return the mean best_epoch across LOAO folds for this trial."""
    path = os.path.join(loao_dir, "loao_trial.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LOAO JSON not found: {path}\n"
            f"Run stage 4 (tcn_leave_one_out) first."
        )

    with open(path) as f:
        data = json.load(f)

    best_epochs = [
        fold["training"]["best_epoch"]
        for fold in data["folds"]
        if fold["training"]["best_epoch"] is not None
    ]
    if not best_epochs:
        raise ValueError(
            f"No best_epoch data found in LOAO folds for trial {trial_id}."
        )

    n_epochs = int(round(sum(best_epochs) / len(best_epochs)))
    logger.info(
        "Trial %d  LOAO best_epoch: mean=%d  (folds: %s)",
        trial_id, n_epochs, best_epochs,
    )
    return n_epochs


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
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_schedule_factor, patience=cfg.lr_schedule_patience,
    )

    dataset = GaitWindowDataset(records, window_size=cfg.window_size, feature_idx=cfg.feature_idx)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    train_losses: list[float] = []

    with mlflow.start_run(run_name=f"pose_tcn_train_trial_{trial_id}"):
        mlflow.log_params(params)
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("n_features", n_features)

        for epoch in range(1, max_epochs + 1):
            loss = train_epoch(model, loader, optimizer, criterion, device, cfg.max_grad_norm)
            train_losses.append(loss)
            scheduler.step(loss)
            mlflow.log_metric("train_loss", loss, step=epoch)
            if epoch % log_every == 0:
                logger.info("Trial %d  Epoch %3d/%d  train=%.4f", trial_id, epoch, max_epochs, loss)

        ckpt_path = cfg.checkpoint_path(f"checkpoint")
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_metric("final_loss", train_losses[-1])
        mlflow.log_param("checkpoint_path", ckpt_path)
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
    loao_dir: str,
    log_every: int = 10,
    *,
    n_test: dict,
    seed: int,
) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_pose_tcn_training")
    logger.info("Loading dataset …")
    all_records = load_dataset(cfg.annotations_csv, fps=cfg.fps, n_trim_padding=cfg.n_trim_padding, dataset=cfg.dataset)
    logger.info("%d videos loaded.", len(all_records))

    records, test_records, test_athletes = train_test_split(all_records, n_test=n_test, seed=seed)
    logger.info(
        "Test set excluded from training: %s (%d videos). Training on %d videos.",
        test_athletes, len(test_records), len(records),
    )

    logger.info("Loading Optuna study …")
    study = load_study(study_cfg, cfg)
    trials_by_id = {t.number: t for t in study.trials}

    results = []
    for trial_id in trial_ids:
        if trial_id not in trials_by_id:
            raise KeyError(f"Trial #{trial_id} not found in study '{study.study_name}'.")
        trial  = trials_by_id[trial_id]
        params = params_from_trial(trial)
        logger.info(
            "Training trial %d  (val_loss=%.4f  dilation=%s  n_blocks=%d  n_filters=%d  kernel=%d)",
            trial_id, trial.value, params["dilation_schedule"],
            params["n_blocks"], params["n_filters"], params["kernel_size"],
        )
        max_epochs = epochs_from_loao(trial_id, loao_dir)
        result = train_trial(trial_id, params, records, cfg, max_epochs, log_every)
        results.append(result)

    out = {
        "max_epochs": max_epochs,
        "n_models":   len(results),
        "models":     [{k: v for k, v in r.items() if k != "train_losses"} for r in results],
    }
    out_path = cfg.results_path("training")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Summary saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw = load_config(args.config)

    cfg              = ExperimentConfig(**raw.get("experiment", {}))
    study_cfg        = raw.get("study", {})
    trial_ids        = raw.get("trial_ids", [])
    best_params_json = raw.get("best_params_json")
    loao_dir         = raw.get("loao_dir")
    log_every        = raw.get("log_every", 10)

    if not trial_ids:
        if best_params_json:
            with open(best_params_json) as _f:
                _bp = json.load(_f)
            trial_ids = [_bp["best_trial_id"]]
            logger.info("Auto-selected trial %d from %s", trial_ids[0], best_params_json)
        else:
            raise ValueError(
                "'trial_ids' is empty and 'best_params_json' is not set. "
                "Either list trial IDs or point 'best_params_json' at the "
                "stage2 best_params.json file."
            )
    if not study_cfg:
        raise ValueError("'study' section is missing in the config.")
    if not loao_dir:
        raise ValueError("'loao_dir' is missing in the config.")

    split_cfg = get_split_config(cfg.dataset_config)
    n_test    = split_cfg["n_test"]
    seed      = split_cfg.get("seed", 42)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main(cfg, study_cfg, trial_ids, loao_dir, log_every, n_test=n_test, seed=seed)
