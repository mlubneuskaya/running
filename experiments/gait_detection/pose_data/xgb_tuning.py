"""XGBoost hyperparameter tuning with Optuna.

Tunes 7 XGBoost hyperparameters on the fixed tuning split.
Objective: maximize macro F1 across all frames in the validation set.
Saves the best model and its parameters.

Usage
-----
    python -m experiments.gait_detection.stage_xgb_tune
    python -m experiments.gait_detection.stage_xgb_tune --n_trials 50
    python -m experiments.gait_detection.stage_xgb_tune --config_overrides '{"feature_idx": [0,1,2,3,4,5]}'

Output
------
    experiments/gait_detection/results/best_params.json
    experiments/gait_detection/checkpoints/xgboost_best.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import joblib
import mlflow
import numpy as np
import optuna
import xgboost as xgb

logger = logging.getLogger(__name__)

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.metrics import per_class_f1
from src.gait.detection.train import seed_everything
from src.gait.gait_data.dataset import load_dataset, tuning_split, train_test_split
from src.pose.utils.load_config import load_config


def _flatten(records, feature_idx=None):
    X, y = [], []
    for r in records:
        feats = r.features if feature_idx is None else r.features[:, feature_idx]
        X.append(feats)
        y.append(r.labels)
    return np.vstack(X), np.concatenate(y)


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ExperimentConfig,
) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators",      100,  800),
        "max_depth":        trial.suggest_int("max_depth",           3,   10),
        "learning_rate":    trial.suggest_float("learning_rate",  1e-2,  0.3, log=True),
        "subsample":        trial.suggest_float("subsample",        0.5,  1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5,  1.0),
        "min_child_weight": trial.suggest_int("min_child_weight",    1,   10),
        "gamma":            trial.suggest_float("gamma",            0.0,  5.0),
    }
    clf = xgb.XGBClassifier(
        **params,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        verbosity=0,
        random_state=cfg.random_seed,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = clf.predict(X_val)
    f1 = per_class_f1(y_val, y_pred, n_classes=cfg.n_classes, class_names=cfg.class_names)
    return f1["macro"]


def main(cfg: ExperimentConfig | None = None, n_trials: int = 50) -> None:
    cfg = cfg or ExperimentConfig()
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_pose_xgb_tuning")

    all_records = load_dataset(cfg.annotations_csv, fps=cfg.fps)
    records, _, _ = train_test_split(all_records)
    train_records, val_records = tuning_split(
        records, n_val_athletes=cfg.n_val_athletes_tuning, seed=cfg.random_seed
    )

    X_train, y_train = _flatten(train_records, cfg.feature_idx)
    X_val,   y_val   = _flatten(val_records,   cfg.feature_idx)
    logger.info("Train frames: %d  Val frames: %d  Features: %d",
                X_train.shape[0], X_val.shape[0], X_train.shape[1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    with mlflow.start_run(run_name="pose_xgb_tune"):
        mlflow.log_params({
            "n_trials":       n_trials,
            "n_val_athletes": cfg.n_val_athletes_tuning,
            "n_features":     X_train.shape[1],
            "feature_idx":    str(cfg.feature_idx),
        })

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, cfg),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_params = study.best_params
        logger.info("Best macro F1 : %.4f", study.best_value)
        logger.info("Best params   : %s", best_params)

        mlflow.log_metric("best_val_macro_f1", study.best_value)
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

    # Retrain best model (no early-stopping set, use n_estimators from search)
    best_clf = xgb.XGBClassifier(
        **best_params,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        verbosity=0,
        random_state=cfg.random_seed,
    )
    best_clf.fit(X_train, y_train)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, "model.pkl")
    joblib.dump(best_clf, ckpt_path)
    logger.info("Model saved  → %s", ckpt_path)

    out = {
        "best_params":       best_params,
        "best_val_macro_f1": study.best_value,
        "feature_idx":       cfg.feature_idx,
        "n_features":        len(cfg.feature_idx) if cfg.feature_idx is not None else 22,
    }
    out_path = cfg.results_path("best_params")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Params saved → %s", out_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw = load_config(args.config)
    cfg = ExperimentConfig(**raw.get("experiment", {}))
    n_trials = raw.get("optuna", {}).get("n_trials", 50)
    main(cfg, n_trials)
