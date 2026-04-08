"""Shared experiment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # Data
    annotations_csv: str = "data/output/annotations/optojump/ml_training_dataset.csv"
    fps: float = 120.0
    n_classes: int = 3
    class_names: list[str] = field(default_factory=lambda: ["left_stance", "right_stance", "flight"])

    # Training defaults
    batch_size: int = 16
    max_epochs: int = 200
    early_stopping_patience: int = 20
    lr: float = 1e-3
    dropout: float = 0.2
    window_size: int = 75
    max_grad_norm: float = 1.0
    lr_schedule_factor: float = 0.5
    lr_schedule_patience: int = 10

    # Architecture defaults
    n_blocks: int = 4
    n_filters: int = 64
    kernel_size: int = 3

    # Optuna — single joint study over all 5 hyperparameters
    optuna_storage: str = "sqlite:///experiments/gait_detection/study.db"
    study_name: str = "tcn_joint"
    n_trials_total: int = 50

    # Output
    output_dir: str = "experiments/gait_detection/results"
    checkpoint_dir: str = "experiments/gait_detection/checkpoints"

    # Tuning split
    n_val_athletes_tuning: int = 4
    tuning_seed: int = 42

    def checkpoint_path(self, name: str) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return os.path.join(self.checkpoint_dir, f"{name}.pt")

    def results_path(self, name: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, f"{name}.json")
