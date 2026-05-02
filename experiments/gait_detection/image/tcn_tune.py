"""Image pipeline Stage 2 — TCN hyperparameter search with Optuna + MLflow.

Same search space and study design as the pose-based tcn_tuning.py, but using
pre-extracted CNN image features.  n_features is inferred from the loaded data
(512 for ResNet-18, 1280 for EfficientNet-B0).

Each trial is logged as a nested MLflow run.

Usage
-----
    python -m experiments.gait_detection.img_tcn_tune
    python -m experiments.gait_detection.img_tcn_tune --config configs/experiments/image/tcn_tuning.yaml

Output
------
    <output_dir>/best_params.json
    <study_log>
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import mlflow
import optuna
from torch.utils.data import DataLoader

import src.gait.detection.dilations as dilation_schedules
from experiments.gait_detection.config import ExperimentConfig
from experiments.gait_detection.tcn_tuning import _suggest
from src.gait.detection.model import TCN
from src.gait.detection.train import Trainer, TrainerConfig
from src.gait.gait_data.dataset import (
    GaitSequenceDataset,
    GaitWindowDataset,
    compute_class_weights,
    train_test_split,
    tuning_split,
)
from src.gait.image.dataset import load_image_dataset
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)



def objective(
    trial: optuna.Trial,
    train_records,
    val_records,
    cfg: ExperimentConfig,
    search_space: dict,
    n_features: int,
) -> float:
    params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}

    dilations = getattr(dilation_schedules, params["dilation_schedule"])(params["n_blocks"])
    model     = TCN(
        n_features=n_features,
        n_blocks=params["n_blocks"],
        n_filters=params["n_filters"],
        kernel_size=params["kernel_size"],
        dropout=params["dropout"],
        dilations=dilations,
    )

    class_weights = compute_class_weights(train_records, n_classes=cfg.n_classes)
    trainer_cfg = TrainerConfig(
        lr=params["lr"],
        dropout=params["dropout"],
        batch_size=cfg.batch_size,
        max_epochs=cfg.max_epochs,
        early_stopping_patience=cfg.early_stopping_patience,
        lr_schedule_factor=cfg.lr_schedule_factor,
        lr_schedule_patience=cfg.lr_schedule_patience,
        max_grad_norm=cfg.max_grad_norm,
        window_size=cfg.window_size,
        checkpoint_path=None,
    )
    trainer = Trainer(model, class_weights, trainer_cfg, trial=trial)

    train_ds = GaitWindowDataset(train_records, window_size=cfg.window_size, feature_idx=None)
    val_ds   = GaitSequenceDataset(val_records, feature_idx=None)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1,              shuffle=False,
                              collate_fn=GaitSequenceDataset.collate, num_workers=0)

    result = trainer.fit(train_loader, val_loader)

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_loss",   result["best_val_loss"])
        mlflow.log_metric("best_epoch", result["best_epoch"])

    trial.set_user_attr("best_epoch", result["best_epoch"])
    return result["best_val_loss"]


def main(cfg: ExperimentConfig, storage, n_trials: int, search_space: dict, features_dir: str, video_input_dir: str, top_n: int = 3) -> None:
    mlflow.set_experiment("gait_image_tcn_tuning")

    logger.info("Loading image dataset …")
    all_records = load_image_dataset(cfg.annotations_csv, features_dir, video_input_dir)
    logger.info("%d records loaded.", len(all_records))

    records, _, test_athletes = train_test_split(all_records)
    logger.info("Test athletes excluded: %s", test_athletes)
    train_records, val_records = tuning_split(records, cfg.n_val_athletes_tuning, cfg.tuning_seed)
    logger.info("Train: %d  Val: %d", len(train_records), len(val_records))

    n_features = train_records[0].features.shape[1]
    logger.info("n_features (backbone output dim): %d", n_features)

    study = optuna.create_study(
        study_name=cfg.study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    with mlflow.start_run(run_name="optuna_study"):
        study.optimize(
            lambda trial: objective(trial, train_records, val_records, cfg, search_space, n_features),
            n_trials=n_trials,
        )
        mlflow.log_metric("best_val_loss",    study.best_value)
        mlflow.log_metric("best_trial_id",    study.best_trial.number)
        mlflow.log_params(study.best_params)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_trials = sorted(completed, key=lambda t: t.value)[:top_n]

    out = {
        "best_trial_id": study.best_trial.number,
        "best_params":   study.best_params,
        "best_val_loss": study.best_value,
        "top_trials": [
            {"trial_id": t.number, "params": t.params, "val_loss": t.value}
            for t in top_trials
        ],
    }
    out_path = cfg.results_path("best_params")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Best params saved to %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg             = ExperimentConfig(**raw.get("experiment", {}))
    optuna_section  = raw.get("optuna", {})
    search_space    = raw.get("search_space", {})
    features_dir    = raw["features_dir"]
    video_input_dir = raw.get("video_input_dir", "data/input/optojump")

    if not search_space:
        raise ValueError("'search_space' section is missing or empty in the config.")

    n_trials     = optuna_section.get("n_trials", 20)
    top_n        = optuna_section.get("top_n", 3)
    storage_type = optuna_section.get("storage_type", "journal")
    storage_path = os.path.expandvars(optuna_section.get("storage", cfg.optuna_storage))

    if storage_type == "journal":
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(storage_path)
        )
    else:
        storage = storage_path

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main(cfg, storage, n_trials, search_space, features_dir, video_input_dir, top_n)
