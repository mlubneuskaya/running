"""XGBoost inference — image features: train final model and save per-record probabilities.

Trains a single XGBoost model on all train athletes using the best hyperparameters
from the image baseline tuning stage, then runs inference on every record (both
train and test).  Saves per-frame softmax probabilities and a manifest.json with
correct split tags.

Unlike xgb_loao.py (which only covers held-out train-set athletes), this script
also covers the held-out test athletes so all records appear in the manifest.

Usage
-----
    python -m experiments.gait_detection.image.xgb_infer \
        --config configs/experiments/image/xgb_inference.yaml

Output
------
    <output_dir>/probs/<safe_stem>.npy  — shape (T, n_classes) float32
    <output_dir>/manifest.json          — per-record metadata with split tags
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

from experiments.gait_detection.config import ExperimentConfig, get_split_config, get_test_dataset_params
from src.gait.detection.train import seed_everything
from src.gait.gait_data.dataset import train_test_split
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
    test_image_cfg: dict | None = None,
) -> None:
    seed_everything(cfg.random_seed)
    mlflow.set_experiment("gait_image_xgb_infer")

    probs_dir = os.path.join(output_dir, "probs")
    os.makedirs(probs_dir, exist_ok=True)

    logger.info("Loading image dataset …")
    all_records = load_image_dataset(cfg.annotations_csv, features_dir, video_input_dir, dataset=cfg.dataset)
    logger.info("%d records loaded.", len(all_records))

    test_ds_params = get_test_dataset_params(cfg.dataset_config)
    if test_ds_params is not None and test_image_cfg is not None:
        train_records = all_records
        test_records  = load_image_dataset(
            test_ds_params["annotations_csv"],
            test_image_cfg["features_dir"],
            test_image_cfg.get("video_input_dir", "data/input/tempos"),
            dataset=test_ds_params["dataset"],
        )
        test_athletes = sorted({r.athlete for r in test_records})
        logger.info("Cross mode: %d train (optojump), %d test (tempos).",
                    len(train_records), len(test_records))
    else:
        train_records, test_records, test_athletes = train_test_split(all_records, n_test=n_test, seed=seed)
        logger.info(
            "Split: %d train records, %d test records (test athletes: %s).",
            len(train_records), len(test_records), test_athletes,
        )

    X_train, y_train = _flatten(train_records)
    logger.info("Fitting XGBoost on %d frames …", len(y_train))

    with mlflow.start_run(run_name="image_xgb_infer"):
        mlflow.log_params(best_params)

        clf = xgb.XGBClassifier(
            **best_params,
            eval_metric="mlogloss",
            tree_method="hist",
            device="cpu",
            verbosity=0,
            random_state=cfg.random_seed,
        )
        clf.fit(X_train, y_train)
        logger.info("Training done. Saving probabilities …")

        manifest_records = []
        for split, split_records in [("train", train_records), ("test", test_records)]:
            for rec in split_records:
                probs = clf.predict_proba(rec.features.astype(np.float32)).astype(np.float32)
                out_name = _safe_stem(rec.video_path) + ".npy"
                out_path = os.path.join(probs_dir, out_name)
                np.save(out_path, probs)
                manifest_records.append({
                    "video_path": rec.video_path,
                    "athlete":    rec.athlete,
                    "split":      split,
                    "n_frames":   int(probs.shape[0]),
                    "probs_path": out_path,
                })
                logger.info(
                    "  [%s] %s → %s (%d frames)", split, rec.athlete, out_name, probs.shape[0]
                )

        mlflow.log_metric("n_train_records", len(train_records))
        mlflow.log_metric("n_test_records",  len(test_records))

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
        "Done — %d train + %d test records written to %s",
        len(train_records), len(test_records), output_dir,
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
    output_dir       = raw.get("output_dir", "data/output/gait/image/xgb_infer")

    with open(best_params_json) as _f:
        _bp = json.load(_f)
    best_params = _bp["best_params"]

    split_cfg     = get_split_config(cfg.dataset_config)
    n_test        = split_cfg["n_test"]
    seed          = split_cfg.get("seed", 42)
    test_image_cfg = raw.get("test_dataset")

    logger.info("Loaded best params from %s: %s", best_params_json, best_params)
    main(cfg, best_params, features_dir, video_input_dir, output_dir,
         n_test=n_test, seed=seed, test_image_cfg=test_image_cfg)
