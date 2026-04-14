"""Shared Optuna study helpers for TCN experiment scripts."""

from __future__ import annotations

import os

import optuna

from experiments.gait_detection.config import ExperimentConfig

REQUIRED_PARAMS: frozenset[str] = frozenset(
    {"lr", "dropout", "n_blocks", "n_filters", "kernel_size", "dilation_schedule"}
)


def load_study(study_cfg: dict, experiment_cfg: ExperimentConfig) -> optuna.Study:
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


def params_from_trial(trial: optuna.trial.FrozenTrial) -> dict:
    params  = trial.params
    missing = REQUIRED_PARAMS - params.keys()
    if missing:
        raise KeyError(
            f"Trial #{trial.number} is missing required parameters: {sorted(missing)}"
        )
    return params
