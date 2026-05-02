"""Image pipeline Stage 6 — inference: save per-record softmax probabilities.

Mirrors tcn_inference.py but uses pre-extracted CNN image features.

Usage
-----
    python -m experiments.gait_detection.img_tcn_infer
    python -m experiments.gait_detection.img_tcn_infer --config configs/experiments/image/tcn_inference.yaml

Output
------
    <output_dir>/probs/<safe_stem>.npy    shape (T, n_classes), float32
    <output_dir>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re

import numpy as np
import optuna
import torch
import torch.nn.functional as F

import src.gait.detection.dilations as dilation_schedules
from experiments.gait_detection.config import ExperimentConfig
from experiments.gait_detection.study_utils import load_study, params_from_trial
from src.gait.detection.model import TCN
from src.gait.detection.train import get_device
from src.gait.gait_data.dataset import train_test_split
from src.gait.image.dataset import load_image_dataset
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)



def _safe_stem(video_path: str) -> str:
    stem = os.path.splitext(video_path)[0]
    return re.sub(r"[/\\]", "_", stem).lstrip("_")


@torch.no_grad()
def _run(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    return F.softmax(model(x).squeeze(0), dim=-1).cpu().numpy().astype(np.float32)


def main(
    cfg: ExperimentConfig,
    study_cfg: dict,
    trial_id: int,
    output_dir: str,
    features_dir: str,
    video_input_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    probs_dir = os.path.join(output_dir, "probs")
    os.makedirs(probs_dir, exist_ok=True)

    logger.info("Loading image dataset …")
    all_records = load_image_dataset(cfg.annotations_csv, features_dir, video_input_dir)
    logger.info("%d records loaded.", len(all_records))

    train_records, test_records, test_athletes = train_test_split(all_records)
    n_features = all_records[0].features.shape[1]
    logger.info(
        "Split: %d train  %d test  n_features=%d",
        len(train_records), len(test_records), n_features,
    )

    study        = load_study(study_cfg, cfg)
    trials_by_id = {t.number: t for t in study.trials}

    if trial_id not in trials_by_id:
        raise KeyError(f"Trial #{trial_id} not found in study '{study.study_name}'.")

    params    = params_from_trial(trials_by_id[trial_id])
    dilations = getattr(dilation_schedules, params["dilation_schedule"])(params["n_blocks"])
    model     = TCN(
        n_features=n_features,
        n_blocks=params["n_blocks"],
        n_filters=params["n_filters"],
        kernel_size=params["kernel_size"],
        dropout=params["dropout"],
        dilations=dilations,
    )

    ckpt_path = cfg.checkpoint_path("checkpoint")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run img_tcn_train (stage 5) first."
        )
    device = get_device()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    logger.info("Loaded checkpoint from %s", ckpt_path)

    manifest = {
        "trial_id":      trial_id,
        "checkpoint":    ckpt_path,
        "params":        params,
        "n_features":    n_features,
        "n_classes":     cfg.n_classes,
        "class_names":   cfg.class_names,
        "fps":           cfg.fps,
        "test_athletes": test_athletes,
        "records":       [],
    }

    for split, split_records in [("train", train_records), ("test", test_records)]:
        for rec in split_records:
            probs    = _run(model, rec.features, device)
            out_name = _safe_stem(rec.video_path) + ".npy"
            out_path = os.path.join(probs_dir, out_name)
            np.save(out_path, probs)
            manifest["records"].append({
                "video_path": rec.video_path,
                "athlete":    rec.athlete,
                "split":      split,
                "n_frames":   int(probs.shape[0]),
                "probs_path": out_path,
            })
            logger.info("  [%s] %s → %s (%d frames)", split, rec.athlete, out_name, probs.shape[0])

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    cfg             = ExperimentConfig(**raw.get("experiment", {}))
    study_cfg       = raw.get("study", {})
    trial_id        = raw.get("trial_id")
    loao_best_json  = raw.get("loao_best_json")
    training_json   = raw.get("training_json")
    output_dir      = raw.get("output_dir", "data/output/gait/image/stage6")
    features_dir    = raw["features_dir"]
    video_input_dir = raw.get("video_input_dir", "data/input/optojump")

    if trial_id is None:
        if training_json:
            with open(training_json) as _f:
                _tr = json.load(_f)
            _entry   = _tr["models"][0]
            trial_id = _entry["trial_id"]
            cfg.checkpoint_dir = os.path.dirname(_entry["checkpoint"])
            logger.info("Auto-selected trial %d from %s", trial_id, training_json)
        elif loao_best_json:
            with open(loao_best_json) as _f:
                _lb = json.load(_f)
            trial_id = _lb["best_trial_id"]
            logger.info("Auto-selected trial %d from LOAO best: %s", trial_id, loao_best_json)
        else:
            raise ValueError("'trial_id' is missing and neither 'training_json' nor 'loao_best_json' is set.")
    if not study_cfg:
        raise ValueError("'study' section is missing.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main(cfg, study_cfg, int(trial_id), output_dir, features_dir, video_input_dir)
