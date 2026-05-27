"""Image pipeline Stage 5 — final training on all train athletes.

Mirrors tcn_training.py but uses pre-extracted CNN image features.
Epoch budget is read from the LOAO fold means.  Logged to MLflow.

Usage
-----
    python -m experiments.gait_detection.img_tcn_train
    python -m experiments.gait_detection.img_tcn_train --config configs/experiments/image/tcn_training.yaml

Output
------
    <checkpoint_dir>/checkpoint.pt
    <output_dir>/img_training.json
"""

from __future__ import annotations

import argparse
import json
import logging

import mlflow
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import src.gait.detection.dilations as dilation_schedules
from experiments.gait_detection.config import ExperimentConfig
from experiments.gait_detection.study_utils import load_study, params_from_trial
from experiments.gait_detection.pose_data.tcn_training import epochs_from_loao
from src.gait.detection.model import TCN
from src.gait.detection.train import get_device, train_epoch, seed_everything
from src.gait.gait_data.dataset import GaitWindowDataset, compute_class_weights, train_test_split
from src.gait.image.dataset import load_image_dataset
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)



def train_trial(
    trial_id: int,
    params: dict,
    records,
    cfg: ExperimentConfig,
    n_features: int,
    max_epochs: int,
    log_every: int,
) -> dict:
    dilations = getattr(dilation_schedules, params["dilation_schedule"])(params["n_blocks"])
    model     = TCN(
        n_features=n_features,
        n_blocks=params["n_blocks"],
        n_filters=params["n_filters"],
        kernel_size=params["kernel_size"],
        dropout=params["dropout"],
        dilations=dilations,
    )

    device        = get_device()
    model         = model.to(device)
    class_weights = compute_class_weights(records, cfg.n_classes).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_schedule_factor, patience=cfg.lr_schedule_patience,
    )

    dataset = GaitWindowDataset(records, window_size=cfg.window_size, feature_idx=None)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    train_losses: list[float] = []

    with mlflow.start_run(run_name=f"img_tcn_train_trial_{trial_id}"):
        mlflow.log_params(params)
        mlflow.log_param("max_epochs",  max_epochs)
        mlflow.log_param("n_features",  n_features)

        for epoch in range(1, max_epochs + 1):
            loss = train_epoch(model, loader, optimizer, criterion, device, cfg.max_grad_norm)
            train_losses.append(loss)
            scheduler.step(loss)
            mlflow.log_metric("train_loss", loss, step=epoch)
            if epoch % log_every == 0:
                logger.info("Trial %d  Epoch %3d/%d  train=%.4f", trial_id, epoch, max_epochs, loss)

        ckpt_path = cfg.checkpoint_path("checkpoint")
        torch.save(model.state_dict(), ckpt_path)
        mlflow.log_metric("final_loss",   train_losses[-1])
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")
        logger.info("Saved checkpoint → %s", ckpt_path)

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
    features_dir: str,
    video_input_dir: str,
    log_every: int,
) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_image_tcn_training")

    logger.info("Loading image dataset …")
    all_records = load_image_dataset(cfg.annotations_csv, features_dir, video_input_dir)
    logger.info("%d records loaded.", len(all_records))

    records, _, test_athletes = train_test_split(all_records)
    logger.info("Test set excluded: %s (%d train records)", test_athletes, len(records))

    n_features = records[0].features.shape[1]
    logger.info("n_features: %d", n_features)

    study        = load_study(study_cfg, cfg)
    trials_by_id = {t.number: t for t in study.trials}

    results = []
    for trial_id in trial_ids:
        if trial_id not in trials_by_id:
            raise KeyError(f"Trial #{trial_id} not found in study '{study.study_name}'.")
        params     = params_from_trial(trials_by_id[trial_id])
        max_epochs = epochs_from_loao(trial_id, loao_dir)
        logger.info("Training trial %d  max_epochs=%d", trial_id, max_epochs)
        result = train_trial(trial_id, params, records, cfg, n_features, max_epochs, log_every)
        results.append(result)

    out = {
        "n_models": len(results),
        "models":   [{k: v for k, v in r.items() if k != "train_losses"} for r in results],
    }
    out_path = cfg.results_path("img_training")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Summary saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg              = ExperimentConfig(**raw.get("experiment", {}))
    study_cfg        = raw.get("study", {})
    trial_ids        = raw.get("trial_ids", [])
    best_params_json = raw.get("best_params_json")
    loao_dir         = raw.get("loao_dir")
    features_dir     = raw["features_dir"]
    video_input_dir  = raw.get("video_input_dir", "data/input/optojump")
    log_every        = raw.get("log_every", 10)

    loao_best_json = raw.get("loao_best_json")

    if not trial_ids:
        if loao_best_json:
            with open(loao_best_json) as _f:
                _lb = json.load(_f)
            trial_ids = [_lb["best_trial_id"]]
            logger.info("Auto-selected trial %d from LOAO best: %s", trial_ids[0], loao_best_json)
        elif best_params_json:
            with open(best_params_json) as _f:
                _bp = json.load(_f)
            trial_ids = [_bp["best_trial_id"]]
            logger.info("Auto-selected trial %d from %s", trial_ids[0], best_params_json)
        else:
            raise ValueError("'trial_ids' is empty and neither 'loao_best_json' nor 'best_params_json' is set.")
    if not study_cfg:
        raise ValueError("'study' section is missing.")
    if not loao_dir:
        raise ValueError("'loao_dir' is missing.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main(cfg, study_cfg, trial_ids, loao_dir, features_dir, video_input_dir, log_every)
