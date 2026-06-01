"""Shared experiment configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # ── Dataset ───────────────────────────────────────────────────────────────
    # Single place to change datasets/fps/split: edit configs/dataset.yaml and
    # set mode = "tempos" | "optojump" | "cross".
    dataset_config: str = "configs/dataset.yaml"

    # The fields below are auto-populated from dataset_config in __post_init__.
    # Override them here only in tests or one-off scripts.
    annotations_csv: str = ""
    dataset: str = ""
    fps: float = 120.0
    n_trim_padding: int = 0

    n_classes: int = 3
    class_names: list[str] = field(default_factory=lambda: ["left_stance", "right_stance", "flight"])

    # Training defaults
    random_seed: int = 42
    batch_size: int = 16
    max_epochs: int = 200
    early_stopping_patience: int = 20
    lr: float = 1e-3
    dropout: float = 0.2
    window_size: int = 75
    max_grad_norm: float = 1.0
    lr_schedule_factor: float = 0.5
    lr_schedule_patience: int = 10

    # Feature selection
    # Index layout (from features.py):
    #   0–5   norm. y-positions   (L/R heel, big_toe, ankle)
    #   6–11  y-velocities        (L/R heel, big_toe, ankle)
    #   12–15 x-velocities        (L/R heel, big_toe)
    #   16–19 joint angles        (L/R knee, L/R ankle)
    #   20–21 hip y + hip dy/dt
    feature_idx: list[int] | None = None  # None → all 22 features

    # Architecture defaults
    n_blocks: int = 4
    n_filters: int = 64
    kernel_size: int = 3

    # Optuna
    optuna_storage: str = "sqlite:///experiments/gait_detection/study.db"
    study_name: str = "tcn_joint"
    n_trials_total: int = 50

    # Output
    output_dir: str = "experiments/gait_detection/results"
    checkpoint_dir: str = "experiments/gait_detection/checkpoints"

    # Tuning split
    n_val_athletes_tuning: int = 4

    def __post_init__(self):
        if self.dataset_config and not self.annotations_csv:
            self._load_dataset_params()

    def _load_dataset_params(self) -> None:
        from src.pose.utils.load_config import load_config
        dc = load_config(self.dataset_config)
        mode = dc["mode"]
        # In cross mode the training dataset is optojump
        ds_key = "optojump" if mode == "cross" else mode
        params = dc["datasets"][ds_key]
        self.annotations_csv  = params["annotations_csv"]
        self.dataset          = params["dataset"]
        self.fps              = float(params["fps"])
        self.n_trim_padding   = int(params["n_trim_padding"])

    def checkpoint_path(self, name: str) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        return os.path.join(self.checkpoint_dir, f"{name}.pt")

    def results_path(self, name: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, f"{name}.json")


def get_split_config(dataset_config_path: str) -> dict:
    """Return the n_test / seed dict for the active mode."""
    from src.pose.utils.load_config import load_config
    dc = load_config(dataset_config_path)
    return dc["splits"][dc["mode"]]


def get_test_dataset_params(dataset_config_path: str) -> dict | None:
    """Return test-dataset params for cross mode; None otherwise.

    In cross mode the test set is the full tempos dataset (no hold-out split).
    In tempos/optojump modes the test set comes from train_test_split, so this
    returns None.
    """
    from src.pose.utils.load_config import load_config
    dc = load_config(dataset_config_path)
    if dc["mode"] != "cross":
        return None
    return dc["datasets"]["tempos"]
