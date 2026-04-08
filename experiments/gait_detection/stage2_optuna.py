"""Stage 2 — Joint hyperparameter search over all 5 parameters.

Tunes learning rate, dropout, n_blocks, n_filters, and kernel_size together
in a single Optuna study.  Separating architecture from training dynamics and
fixing one before tuning the other risks evaluating good architectures with
the wrong lr/dropout — a joint search avoids this at negligible extra cost
since TPE handles 5 dimensions well.

Designed for parallel execution: multiple SLURM jobs (or local processes) point
at the same SQLite database and contribute trials concurrently.

Usage (local)
-------------
    python -m experiments.gait_detection.stage2_optuna
    python -m experiments.gait_detection.stage2_optuna --n_trials 10

Usage (PLGrid — see slurm/stage2_optuna.sbatch)
-----------------------------------------------
    python -m experiments.gait_detection.stage2_optuna \\
        --storage sqlite:////net/shared/path/study.db \\
        --n_trials 10
"""

from __future__ import annotations

import argparse
import json

import optuna
from torch.utils.data import DataLoader

from experiments.gait_detection.config import ExperimentConfig
from src.gait_detection.dataset import (
    GaitSequenceDataset,
    GaitWindowDataset,
    compute_class_weights,
    load_dataset,
    tuning_split,
)
from src.gait_detection.model import TCN
from src.gait_detection.train import Trainer, TrainerConfig


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
    parser.add_argument("--storage",  default=None, help="Optuna storage URL (default from config)")
    parser.add_argument("--n_trials", type=int, default=10, help="Trials to run in this process")
    parser.add_argument("--config_overrides", default="{}", help="JSON string of ExperimentConfig overrides")
    args = parser.parse_args()

    overrides = json.loads(args.config_overrides)
    cfg = ExperimentConfig(**overrides) if overrides else ExperimentConfig()
    main(cfg, args.storage, args.n_trials)
