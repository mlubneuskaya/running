"""Image pipeline — full ResNet inference: save per-record softmax probabilities.

Mirrors tcn_infer.py but uses the trained ResNetFinetune (backbone + head).
Crops are loaded from video frames on-the-fly per record (one record at a time
to keep peak memory bounded).

Usage
-----
    python -m experiments.gait_detection.image.resnet_infer \
        --config configs/experiments/image/resnet_infer.yaml

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
from torch.utils.data import DataLoader

from experiments.gait_detection.config import ExperimentConfig
from src.gait.detection.train import get_device
from src.gait.gait_data.dataset import train_test_split
from src.gait.image.dataset import load_image_records_for_finetune
from src.gait.image.finetune import FrameCropDataset, ResNetFinetune
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)


def _safe_stem(video_path: str) -> str:
    stem = os.path.splitext(video_path)[0]
    return re.sub(r"[/\\]", "_", stem).lstrip("_")


@torch.no_grad()
def _run_record(
    model: torch.nn.Module,
    rec_info: tuple,
    img_size: int,
    bbox_padding: float,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    ds     = FrameCropDataset([rec_info], img_size=img_size, bbox_padding=bbox_padding)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs_list = []
    for X, _ in loader:
        X = X.to(device)
        probs_list.append(F.softmax(model(X), dim=-1).cpu().numpy())
    return np.concatenate(probs_list, axis=0).astype(np.float32)


def main(
    cfg: ExperimentConfig,
    params: dict,
    backbone_name: str,
    img_size: int,
    bbox_padding: float,
    output_dir: str,
    pose_dir: str,
    video_input_dir: str,
    n_test: dict,
    seed: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    probs_dir = os.path.join(output_dir, "probs")
    os.makedirs(probs_dir, exist_ok=True)

    logger.info("Loading image records …")
    all_records_info = load_image_records_for_finetune(
        cfg.annotations_csv, pose_dir, video_input_dir
    )
    all_records = [r for r, _, _ in all_records_info]
    logger.info("%d records loaded.", len(all_records))

    _, _, test_athletes = train_test_split(all_records, n_test=n_test, seed=seed)
    test_set = set(test_athletes)

    device = get_device()
    model  = ResNetFinetune(
        backbone_name=backbone_name,
        n_classes=cfg.n_classes,
        dropout=params["dropout"],
    ).to(device)

    ckpt_path = cfg.checkpoint_path("checkpoint")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nRun resnet_train first."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    logger.info("Loaded checkpoint from %s", ckpt_path)

    manifest = {
        "params":        params,
        "backbone_name": backbone_name,
        "img_size":      img_size,
        "n_classes":     cfg.n_classes,
        "class_names":   cfg.class_names,
        "fps":           cfg.fps,
        "test_athletes": list(test_athletes),
        "records":       [],
    }

    for rec_info in all_records_info:
        rec   = rec_info[0]
        split = "test" if rec.athlete in test_set else "train"

        probs    = _run_record(model, rec_info, img_size, bbox_padding, cfg.batch_size, device)
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
    pose_dir        = raw["pose_dir"]
    video_input_dir = raw.get("video_input_dir", "data/input/optojump")
    output_dir      = raw.get("output_dir", "data/output/gait/image/resnet/infer")

    training_json    = raw.get("training_json")
    best_params_json = raw.get("best_params_json")

    if training_json:
        with open(training_json) as _f:
            _tr = json.load(_f)
        params        = _tr["params"]
        backbone_name = _tr.get("backbone_name", "resnet18")
        img_size      = _tr.get("img_size",      224)
        bbox_padding  = _tr.get("bbox_padding",  0.1)
        cfg.checkpoint_dir = os.path.dirname(_tr["checkpoint"])
        logger.info("Loaded training info from %s", training_json)
    elif best_params_json:
        with open(best_params_json) as _f:
            _bp = json.load(_f)
        params        = _bp["best_params"]
        backbone_name = _bp.get("backbone_name", "resnet18")
        img_size      = _bp.get("img_size",      224)
        bbox_padding  = _bp.get("bbox_padding",  0.1)
        logger.info("Loaded params from %s", best_params_json)
    else:
        raise ValueError("Either 'training_json' or 'best_params_json' must be set.")

    split_cfg = load_config(raw["split_config"])
    n_test    = split_cfg["n_test"]
    seed      = split_cfg.get("seed", 42)

    main(cfg, params, backbone_name, img_size, bbox_padding, output_dir, pose_dir, video_input_dir,
         n_test=n_test, seed=seed)
