"""Image pipeline Stage 1 — XGBoost baseline with Optuna tuning.

Tunes XGBoost on pre-extracted CNN features to establish a per-frame baseline
before the full TCN pipeline.  A macro F1 above ~0.70 indicates the backbone
produces class-discriminative representations.

Usage
-----
    python -m experiments.gait_detection.img_baselines
    python -m experiments.gait_detection.img_baselines --config configs/experiments/image/baselines.yaml

Output
------
    <output_dir>/results.json                   best params + val metrics
    <checkpoint_dir>/xgboost_best.pkl           best XGBoost model
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import joblib
import numpy as np
import optuna
import xgboost as xgb

import mlflow

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.metrics import per_class_f1
from src.gait.detection.train import seed_everything
from src.gait.gait_data.dataset import train_test_split, tuning_split
from src.gait.image.dataset import load_image_dataset
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)



def _flatten(records) -> tuple[np.ndarray, np.ndarray]:
    feats  = np.concatenate([r.features for r in records], axis=0)
    labels = np.concatenate([r.labels   for r in records], axis=0)
    return feats.astype(np.float32), labels.astype(np.int64)


def _objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    class_names: list[str],
    random_seed: int = 42,
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
        random_state=random_seed,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = clf.predict(X_val)
    f1 = per_class_f1(y_val, y_pred, n_classes=n_classes, class_names=class_names)
    return f1["macro"]


def main(cfg: ExperimentConfig, features_dir: str, video_input_dir: str, n_trials: int, n_test: dict, seed: int) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_image_baselines")

    logger.info("Loading image dataset …")
    all_records = load_image_dataset(cfg.annotations_csv, features_dir, video_input_dir)
    logger.info("%d records loaded.", len(all_records))

    records, _, test_athletes = train_test_split(all_records, n_test=n_test, seed=seed)
    logger.info("Test athletes excluded: %s", test_athletes)
    train_records, val_records = tuning_split(records, cfg.n_val_athletes_tuning, cfg.random_seed)
    logger.info("Train: %d records  Val: %d records", len(train_records), len(val_records))

    X_train, y_train = _flatten(train_records)
    X_val,   y_val   = _flatten(val_records)
    logger.info("Train frames: %d  Val frames: %d  Features: %d",
                X_train.shape[0], X_val.shape[0], X_train.shape[1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.random_seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    with mlflow.start_run(run_name="img_xgb_tune"):
        mlflow.log_params({
            "features_dir":   features_dir,
            "n_trials":       n_trials,
            "n_val_athletes": cfg.n_val_athletes_tuning,
            "n_features":     X_train.shape[1],
        })

        study.optimize(
            lambda trial: _objective(
                trial, X_train, y_train, X_val, y_val, cfg.n_classes, cfg.class_names, cfg.random_seed
            ),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best_params = study.best_params
        logger.info("Best macro F1 : %.4f", study.best_value)
        logger.info("Best params   : %s", best_params)

        mlflow.log_metric("best_val_macro_f1", study.best_value)
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

    # Retrain on full training set with best params
    best_clf = xgb.XGBClassifier(
        **best_params,
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        verbosity=0,
        random_state=cfg.random_seed,
    )
    best_clf.fit(X_train, y_train)

    y_pred = best_clf.predict(X_val)
    f1 = per_class_f1(y_val, y_pred, n_classes=cfg.n_classes, class_names=cfg.class_names)
    logger.info(
        "XGBoost baseline  macro_F1=%.3f  left=%.3f  right=%.3f  flight=%.3f",
        f1["macro"], f1["left_stance"], f1["right_stance"], f1["flight"],
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, "xgboost_best.pkl")
    joblib.dump(best_clf, ckpt_path)
    logger.info("Model saved → %s", ckpt_path)

    out = {
        "best_params":       best_params,
        "best_val_macro_f1": study.best_value,
        "n_features":        int(X_train.shape[1]),
        "n_train_records":   len(train_records),
        "n_val_records":     len(val_records),
        "test_athletes":     test_athletes,
        "val_f1":            f1,
    }
    out_path = cfg.results_path("results")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg             = ExperimentConfig(**raw.get("experiment", {}))
    features_dir    = raw["features_dir"]
    video_input_dir = raw.get("video_input_dir", "data/input/optojump")
    n_trials        = raw.get("optuna", {}).get("n_trials", 50)

    split_cfg = load_config(raw["split_config"])
    n_test    = split_cfg["n_test"]
    seed      = split_cfg.get("seed", 42)

    main(cfg, features_dir, video_input_dir, n_trials, n_test=n_test, seed=seed)
