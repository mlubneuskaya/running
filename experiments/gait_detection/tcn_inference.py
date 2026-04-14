"""TCN inference — run a trained checkpoint on a dataset and save per-record probabilities.

Loads model architecture from the Optuna study (same as tcn_training.py), then runs the
checkpoint forward pass on every record in the dataset.  Saves:

  - one ``.npy`` per record: shape ``(T, n_classes)`` float32 softmax probabilities
  - ``manifest.json`` listing every record with its athlete, video path, and output path

No postprocessing (argmax / min_duration_filter / derive_events) is applied here —
that is left entirely to the caller.

Usage
-----
    python -m experiments.gait_detection.tcn_inference
    python -m experiments.gait_detection.tcn_inference --config configs/experiments/tcn_inference.yaml

Output
------
    <output_dir>/<safe_stem>.npy
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

import src.gait.detection.dilations as dilation_schedules
from experiments.gait_detection.config import ExperimentConfig
from experiments.gait_detection.study_utils import load_study, params_from_trial
from src.gait.detection.model import TCN
from src.gait.gait_data.dataset import load_dataset
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/experiments/tcn_inference.yaml"


def _safe_stem(video_path: str) -> str:
    """Convert a video path to a safe flat filename (no slashes, no extension)."""
    stem = os.path.splitext(video_path)[0]
    return re.sub(r"[/\\]", "_", stem).lstrip("_")



@torch.no_grad()
def _run(
    model: torch.nn.Module,
    features: np.ndarray,
    feature_idx: list[int] | None,
    device: torch.device,
) -> np.ndarray:
    """Return softmax probabilities, shape (T, n_classes), float32."""
    model.eval()
    feats = features if feature_idx is None else features[:, feature_idx]
    x     = torch.from_numpy(feats).unsqueeze(0).to(device)   # (1, T, C)
    logits = model(x).squeeze(0)                               # (T, n_classes)
    return F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)



def main(
    cfg: ExperimentConfig,
    study_cfg: dict,
    trial_id: int,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading dataset …")
    records = load_dataset(cfg.annotations_csv, fps=cfg.fps)
    logger.info("%d records loaded.", len(records))

    logger.info("Loading Optuna study …")
    study        = load_study(study_cfg, cfg)
    trials_by_id = {t.number: t for t in study.trials}

    if trial_id not in trials_by_id:
        raise KeyError(f"Trial #{trial_id} not found in study '{study.study_name}'.")

    trial  = trials_by_id[trial_id]
    params = params_from_trial(trial)
    logger.info(
        "Trial %d  val_loss=%.4f  %s",
        trial_id, trial.value, params,
    )

    n_features = len(cfg.feature_idx) if cfg.feature_idx is not None else 22
    dilations  = getattr(dilation_schedules, params["dilation_schedule"])(params["n_blocks"])

    model = TCN(
        n_features=n_features,
        n_blocks=params["n_blocks"],
        n_filters=params["n_filters"],
        kernel_size=params["kernel_size"],
        dropout=params["dropout"],
        dilations=dilations,
    )

    ckpt_path = cfg.checkpoint_path(f"trial_{trial_id}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run 'make stage5' (tcn_training) first."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    logger.info("Loaded checkpoint from %s  (device: %s)", ckpt_path, device)

    manifest = {
        "trial_id":       trial_id,
        "checkpoint":     ckpt_path,
        "params":         params,
        "feature_idx":    cfg.feature_idx,
        "n_classes":      cfg.n_classes,
        "class_names":    cfg.class_names,
        "fps":            cfg.fps,
        "records":        [],
    }

    for rec in records:
        probs    = _run(model, rec.features, cfg.feature_idx, device)
        out_name = _safe_stem(rec.video_path) + ".npy"
        out_path = os.path.join(output_dir, out_name)
        np.save(out_path, probs)

        manifest["records"].append({
            "video_path": rec.video_path,
            "athlete":    rec.athlete,
            "n_frames":   int(probs.shape[0]),
            "probs_path": out_path,
        })
        logger.info("  %s  →  %s  (%d frames)", rec.athlete, out_name, probs.shape[0])

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", manifest_path)
    logger.info("Done — %d records written to %s", len(records), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw = load_config(args.config)

    cfg        = ExperimentConfig(**raw.get("experiment", {}))
    study_cfg  = raw.get("study", {})
    trial_id   = raw.get("trial_id")
    output_dir = raw.get("output_dir", "data/output/experiments/gait/tcn/predictions")

    if trial_id is None:
        raise ValueError("'trial_id' is missing in the config.")
    if not study_cfg:
        raise ValueError("'study' section is missing in the config.")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main(cfg, study_cfg, int(trial_id), output_dir)
