"""Stage 2 — Joint hyperparameter search over all 5 parameters.

Tunes learning rate, dropout, n_blocks, n_filters, and kernel_size together
in a single Optuna study.  Designed for parallel execution: multiple SLURM
jobs point at the same SQLite database and contribute trials concurrently.

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
import os

import optuna
from torch.utils.data import DataLoader

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.model import TCN
from src.gait.detection.train import TrainerConfig, Trainer
from src.gait.gait_data.dataset import compute_class_weights, GaitWindowDataset, GaitSequenceDataset, load_dataset, \
    tuning_split
from src.pose.utils.load_config import load_config

DEFAULT_CONFIG = "configs/experiments/stage2_optuna.yaml"


def objective(trial, train_records, val_records, cfg: ExperimentConfig):
    lr          = trial.suggest_float("lr",       1e-4, 1e-2, log=True)
    dropout     = trial.suggest_float("dropout",  0.1,  0.4)
    n_blocks    = trial.suggest_int("n_blocks",   3,    5)
    n_filters   = trial.suggest_categorical("n_filters",   [32, 64, 128])
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])

    n_features = len(cfg.feature_idx) if cfg.feature_idx is not None else 22

    model = TCN(
        n_features=n_features,
        n_blocks=n_blocks,
        n_filters=n_filters,
        kernel_size=kernel_size,
        dropout=dropout,
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
    trial.set_user_attr("train_losses", result["train_losses"])
    trial.set_user_attr("val_losses",   result["val_losses"])
    trial.set_user_attr("best_epoch",   result["best_epoch"])
    return result["best_val_loss"]


def main(cfg: ExperimentConfig, storage: str | None, n_trials: int) -> None:
    records = load_dataset(cfg.annotations_csv, fps=cfg.fps)
    train_records, val_records = tuning_split(
        records, n_val_athletes=cfg.n_val_athletes_tuning, seed=cfg.tuning_seed
    )

    storage = storage or cfg.optuna_storage
    study = optuna.create_study(
        study_name=cfg.study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
    )

    study.optimize(
        lambda trial: objective(trial, train_records, val_records, cfg),
        n_trials=n_trials,
    )


    out = {"best_params": study.best_params, "best_val_loss": study.best_value}
    out_path = cfg.results_path("stage2_best_params")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="Path to YAML config file (default: configs/experiments/stage2_optuna.yaml)",
    )
    args = parser.parse_args()

    raw = load_config(args.config)

    cfg = ExperimentConfig(**raw.get("experiment", {}))

    optuna_section = raw.get("optuna", {})
    storage  = os.path.expandvars(optuna_section.get("storage", cfg.optuna_storage))
    n_trials = optuna_section.get("n_trials", 10)

    main(cfg, storage, n_trials)
