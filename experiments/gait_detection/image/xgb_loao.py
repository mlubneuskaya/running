"""Leave-one-athlete-out cross-validation for XGBoost (image features).

Uses best hyperparameters from image baseline XGBoost tuning.  For each
held-out athlete, trains XGBoost on all other athletes and saves per-frame
softmax probabilities for the held-out records.

Usage
-----
    python -m experiments.gait_detection.image.xgb_loao \
        --config configs/experiments/image/xgb_loao.yaml

Output
------
    <output_dir>/manifest.json    — per-record probs paths (train records only)
    <output_dir>/probs/           — .npy arrays, shape (T, n_classes) float32
    <output_dir>/loao.json        — aggregate F1 + confusion matrix per fold
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re

import mlflow
import numpy as np
import xgboost as xgb

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.metrics import (
    aggregate_confusion_matrices,
    confusion_matrix,
    per_class_f1,
)
from src.gait.detection.train import seed_everything
from src.gait.gait_data.dataset import loao_splits, train_test_split
from src.gait.image.dataset import load_image_dataset
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)


def _safe_stem(video_path: str) -> str:
    stem = os.path.splitext(video_path)[0]
    return re.sub(r"[/\\]", "_", stem).lstrip("_")


def _flatten(records) -> tuple[np.ndarray, np.ndarray]:
    feats  = np.concatenate([r.features for r in records], axis=0)
    labels = np.concatenate([r.labels   for r in records], axis=0)
    return feats.astype(np.float32), labels.astype(np.int64)


def main(
    cfg: ExperimentConfig,
    best_params: dict,
    features_dir: str,
    video_input_dir: str,
    output_dir: str,
    n_test: dict,
    seed: int,
) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_image_xgb_loao")

    probs_dir = os.path.join(output_dir, "probs")
    os.makedirs(probs_dir, exist_ok=True)

    logger.info("Loading image dataset …")
    all_records = load_image_dataset(cfg.annotations_csv, features_dir, video_input_dir)
    logger.info("%d records loaded.", len(all_records))

    records, _, test_athletes = train_test_split(all_records, n_test=n_test, seed=seed)
    logger.info("Test athletes excluded: %s  (%d train records)", test_athletes, len(records))

    n_athletes = len({r.athlete for r in records})
    fold_results = []
    all_cms = []
    manifest_records = []

    with mlflow.start_run(run_name="image_xgb_loao"):
        mlflow.log_params(best_params)

        for fold_i, (train_records, val_records, athlete) in enumerate(loao_splits(records), 1):
            logger.info(
                "[Fold %d/%d] Held out: %s (%d videos)",
                fold_i, n_athletes, athlete, len(val_records),
            )

            X_train, y_train = _flatten(train_records)
            X_val,   y_val   = _flatten(val_records)

            clf = xgb.XGBClassifier(
                **best_params,
                eval_metric="mlogloss",
                tree_method="hist",
                device="cpu",
                verbosity=0,
                random_state=cfg.random_seed,
            )
            clf.fit(X_train, y_train)

            for rec in val_records:
                probs = clf.predict_proba(rec.features.astype(np.float32)).astype(np.float32)
                out_name = _safe_stem(rec.video_path) + ".npy"
                out_path = os.path.join(probs_dir, out_name)
                np.save(out_path, probs)
                manifest_records.append({
                    "video_path": rec.video_path,
                    "athlete":    rec.athlete,
                    "split":      "train",
                    "n_frames":   int(probs.shape[0]),
                    "probs_path": out_path,
                })

            y_pred = clf.predict(X_val)
            f1 = per_class_f1(y_val, y_pred, n_classes=cfg.n_classes, class_names=cfg.class_names)
            cm = confusion_matrix(y_val, y_pred, n_classes=cfg.n_classes)
            all_cms.append(cm)

            fold_result = {
                "athlete":          athlete,
                "n_val_videos":     len(val_records),
                "f1":               f1,
                "confusion_matrix": cm.tolist(),
                "training":         {"n_estimators": clf.n_estimators},
            }
            fold_results.append(fold_result)

            with mlflow.start_run(run_name=f"fold_{athlete}", nested=True):
                mlflow.log_metric("macro_f1", f1["macro"])
                for cls in cfg.class_names:
                    mlflow.log_metric(f"f1_{cls}", f1[cls])

            logger.info(
                "  F1 macro=%.3f  left=%.3f  right=%.3f  flight=%.3f",
                f1["macro"], f1["left_stance"], f1["right_stance"], f1["flight"],
            )

        macro_f1s = [r["f1"]["macro"] for r in fold_results]
        total_cm = aggregate_confusion_matrices(all_cms)

        loao_summary = {
            "best_params":            best_params,
            "n_classes":              cfg.n_classes,
            "class_names":            cfg.class_names,
            "n_folds":                len(fold_results),
            "macro_f1_mean":          float(np.mean(macro_f1s)),
            "macro_f1_std":           float(np.std(macro_f1s)),
            "per_class_f1_mean": {
                cls: float(np.mean([r["f1"][cls] for r in fold_results]))
                for cls in cfg.class_names
            },
            "total_confusion_matrix": total_cm.tolist(),
            "folds":                  fold_results,
        }

        mlflow.log_metric("macro_f1_mean", loao_summary["macro_f1_mean"])
        mlflow.log_metric("macro_f1_std",  loao_summary["macro_f1_std"])

    loao_path = os.path.join(output_dir, "loao.json")
    with open(loao_path, "w") as f:
        json.dump(loao_summary, f, indent=2)
    logger.info("LOAO results saved → %s", loao_path)

    manifest = {
        "best_params":   best_params,
        "n_classes":     cfg.n_classes,
        "class_names":   cfg.class_names,
        "fps":           cfg.fps,
        "test_athletes": list(test_athletes),
        "records":       manifest_records,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved → %s", manifest_path)

    logger.info(
        "LOAO done  macro_F1=%.3f ± %.3f",
        loao_summary["macro_f1_mean"], loao_summary["macro_f1_std"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg              = ExperimentConfig(**raw.get("experiment", {}))
    best_params_json = raw["best_params_json"]
    features_dir     = raw["features_dir"]
    video_input_dir  = raw.get("video_input_dir", "data/input/optojump")
    output_dir       = raw.get("output_dir", "data/output/gait/image/xgb_loao")

    with open(best_params_json) as _f:
        _bp = json.load(_f)
    best_params = _bp["best_params"]

    split_cfg = load_config(raw["split_config"])
    n_test    = split_cfg["n_test"]
    seed      = split_cfg.get("seed", 42)

    logger.info("Loaded best params from %s: %s", best_params_json, best_params)
    main(cfg, best_params, features_dir, video_input_dir, output_dir, n_test=n_test, seed=seed)
