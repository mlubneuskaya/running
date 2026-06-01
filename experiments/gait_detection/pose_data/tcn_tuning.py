"""Stage 2 — Joint hyperparameter search over all 5 parameters.

Tunes learning rate, dropout, n_blocks, n_filters, and kernel_size together
in a single Optuna study.  Multiple workers can share the same Optuna storage
to contribute trials concurrently.

All settings are read from a YAML config file.  No experiment parameters are
passed on the command line.

Usage
-----
    python -m experiments.gait_detection.stage2_optuna
    python -m experiments.gait_detection.stage2_optuna --config configs/experiments/stage2_optuna.yaml

Output
------
    experiments/gait_detection/results/stage2_best_params.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import mlflow
import optuna
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from experiments.gait_detection.config import ExperimentConfig, get_split_config
import src.gait.detection.dilations as dilation_schedules
from src.gait.detection.model import TCN
from src.gait.detection.train import TrainerConfig, Trainer, seed_everything
from src.gait.gait_data.dataset import compute_class_weights, GaitWindowDataset, GaitSequenceDataset, load_dataset, \
    tuning_split, train_test_split
from src.pose.utils.load_config import load_config



def _suggest(trial, name: str, spec: dict):
    t = spec["type"]
    if t == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    if t == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown search space type '{t}' for parameter '{name}'")


def objective(trial, train_records, val_records, cfg: ExperimentConfig, search_space: dict):
    params = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}

    lr          = params["lr"]
    dropout     = params["dropout"]
    n_blocks    = params["n_blocks"]
    n_filters   = params["n_filters"]
    kernel_size = params["kernel_size"]
    schedule    = params["dilation_schedule"]

    dilations  = getattr(dilation_schedules, schedule)(n_blocks)
    n_features = len(cfg.feature_idx) if cfg.feature_idx is not None else 22

    model = TCN(
        n_features=n_features,
        n_blocks=n_blocks,
        n_filters=n_filters,
        kernel_size=kernel_size,
        dropout=dropout,
        dilations=dilations,
    )
    class_weights = compute_class_weights(train_records, n_classes=cfg.n_classes)

    trainer_cfg = TrainerConfig(
        lr=lr,
        dropout=dropout,
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

    train_ds = GaitWindowDataset(train_records, window_size=cfg.window_size, feature_idx=cfg.feature_idx)
    val_ds   = GaitSequenceDataset(val_records, feature_idx=cfg.feature_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              collate_fn=GaitSequenceDataset.collate, num_workers=0)

    result = trainer.fit(train_loader, val_loader)

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_loss",   result["best_val_loss"])
        mlflow.log_metric("best_epoch", result["best_epoch"])

    trial.set_user_attr("train_losses", result["train_losses"])
    trial.set_user_attr("val_losses",   result["val_losses"])
    trial.set_user_attr("best_epoch",   result["best_epoch"])
    return result["best_val_loss"]


def main(cfg: ExperimentConfig, storage: str | None, n_trials: int, search_space: dict, n_test: dict, seed: int) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_pose_tcn_tuning")

    all_records = load_dataset(cfg.annotations_csv, fps=cfg.fps, n_trim_padding=cfg.n_trim_padding, dataset=cfg.dataset)
    records, _, test_athletes = train_test_split(all_records, n_test=n_test, seed=seed)
    logger.info("Test athletes excluded from tuning: %s", test_athletes)
    train_records, val_records = tuning_split(
        records, n_val_athletes=cfg.n_val_athletes_tuning, seed=cfg.random_seed
    )

    storage = storage or cfg.optuna_storage
    study = optuna.create_study(
        study_name=cfg.study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    with mlflow.start_run(run_name="optuna_study"):
        study.optimize(
            lambda trial: objective(trial, train_records, val_records, cfg, search_space),
            n_trials=n_trials,
        )
        mlflow.log_metric("best_val_loss",  study.best_value)
        mlflow.log_metric("best_trial_id",  study.best_trial.number)
        mlflow.log_params(study.best_params)

    out = {
        "best_trial_id": study.best_trial.number,
        "best_params":   study.best_params,
        "best_val_loss": study.best_value,
    }
    out_path = cfg.results_path("best_params")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw = load_config(args.config)

    cfg = ExperimentConfig(**raw.get("experiment", {}))

    optuna_section = raw.get("optuna", {})
    n_trials     = optuna_section.get("n_trials", 10)
    storage_type = optuna_section.get("storage_type", "sqlite")
    storage_path = os.path.expandvars(optuna_section.get("storage", cfg.optuna_storage))
    search_space = raw.get("search_space", {})

    if not search_space:
        raise ValueError("'search_space' section is missing or empty in the config.")

    if storage_type == "journal":
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(storage_path)
        )
    else:
        storage = storage_path  # SQLite URL passed as string

    split_cfg = get_split_config(cfg.dataset_config)
    n_test    = split_cfg["n_test"]
    seed      = split_cfg.get("seed", 42)

    main(cfg, storage, n_trials, search_space, n_test=n_test, seed=seed)
