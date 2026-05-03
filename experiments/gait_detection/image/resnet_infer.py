"""Image pipeline — linear head inference: save per-record softmax probabilities.

Mirrors tcn_infer.py but uses the trained LinearHead (per-frame classifier).

Usage
-----
    python -m experiments.gait_detection.image.resnet_infer --config configs/experiments/image/resnet_infer.yaml

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
import torch
import torch.nn.functional as F

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.train import get_device
from src.gait.gait_data.dataset import train_test_split
from src.gait.image.dataset import load_image_dataset
from src.gait.image.finetune import LinearHead
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)


def _safe_stem(video_path: str) -> str:
    stem = os.path.splitext(video_path)[0]
    return re.sub(r"[/\\]", "_", stem).lstrip("_")


@torch.no_grad()
def _run(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X = torch.from_numpy(features).float().to(device)
    return F.softmax(model(X), dim=-1).cpu().numpy().astype(np.float32)


def main(
    cfg: ExperimentConfig,
    params: dict,
    n_features: int,
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
    logger.info(
        "Split: %d train  %d test  n_features=%d",
        len(train_records), len(test_records), n_features,
    )

    device = get_device()
    model  = LinearHead(n_features, n_classes=cfg.n_classes, dropout=params["dropout"]).to(device)

    ckpt_path = cfg.checkpoint_path("checkpoint")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run resnet_train first."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    logger.info("Loaded checkpoint from %s", ckpt_path)

    manifest = {
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
    features_dir    = raw["features_dir"]
    video_input_dir = raw.get("video_input_dir", "data/input/optojump")
    output_dir      = raw.get("output_dir", "data/output/gait/image/resnet/infer")

    training_json    = raw.get("training_json")
    best_params_json = raw.get("best_params_json")

    if training_json:
        with open(training_json) as _f:
            _tr = json.load(_f)
        params     = _tr["params"]
        n_features = _tr["n_features"]
        cfg.checkpoint_dir = os.path.dirname(_tr["checkpoint"])
        logger.info("Loaded training info from %s", training_json)
    elif best_params_json:
        with open(best_params_json) as _f:
            _bp = json.load(_f)
        params     = _bp["best_params"]
        n_features = _bp["n_features"]
        logger.info("Loaded params from %s", best_params_json)
    else:
        raise ValueError("Either 'training_json' or 'best_params_json' must be set.")

    main(cfg, params, n_features, output_dir, features_dir, video_input_dir)
