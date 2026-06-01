"""Image feature extraction — run CNN backbone on video crops from pose bboxes.

Reads the active dataset mode from configs/dataset.yaml to determine which
videos to process, mirroring how every other image stage loads its data.

Mode → datasets processed:
  optojump → optojump only
  tempos   → tempos only
  cross    → optojump (train) + tempos (test)
  smoke    → smoke subset (saves into the optojump features dir)

Usage
-----
    python -m experiments.gait_detection.image.feature_extract --config configs/experiments/image/feature_extract.yaml

Output
------
    <features_dir>/<relative_video_stem>.npy   shape (T, feature_dim), float32
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from src.gait.detection.train import get_device
from src.gait.image.extractor import ImageFeatureExtractor
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)


def _pose_json_path(video_path: str, pose_dir: str, video_input_dir: str) -> str:
    base = os.path.splitext(video_path)[0]
    rel  = os.path.relpath(base, video_input_dir)
    return os.path.join(pose_dir, rel + ".json")


def _feature_out_path(video_path: str, features_dir: str, video_input_dir: str) -> str:
    base = os.path.splitext(video_path)[0]
    rel  = os.path.relpath(base, video_input_dir)
    return os.path.join(features_dir, rel + ".npy")


def extract_dataset(
    annotations_csv: str,
    video_input_dir: str,
    pose_dir: str,
    features_dir: str,
    extractor: ImageFeatureExtractor,
) -> None:
    ann_df = pd.read_csv(annotations_csv)
    video_paths = ann_df["video_path"].unique().tolist()
    logger.info("%d videos to process from %s.", len(video_paths), annotations_csv)

    for i, video_path in enumerate(video_paths, 1):
        pose_path = _pose_json_path(video_path, pose_dir, video_input_dir)
        out_path  = _feature_out_path(video_path, features_dir, video_input_dir)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose JSON not found: {pose_path}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        logger.info("[%d/%d] %s", i, len(video_paths), video_path)
        features = extractor.extract_video(video_path, pose_path)
        np.save(out_path, features)
        logger.info("  Saved %s  shape=%s", out_path, features.shape)

    logger.info("Done. Features saved to: %s", features_dir)


def _datasets_for_mode(mode: str, datasets_cfg: dict) -> list[dict]:
    """Return the list of dataset configs to process for the given mode."""
    if mode == "cross":
        return [datasets_cfg["optojump"], datasets_cfg["tempos"]]
    if mode in datasets_cfg:
        return [datasets_cfg[mode]]
    raise ValueError(
        f"Unknown mode {mode!r}. Add an entry under 'datasets' in feature_extract.yaml "
        f"or choose from: {list(datasets_cfg)}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    dc   = load_config(raw["dataset_config"])
    mode = dc["mode"]
    logger.info("Active mode: %s", mode)

    device    = raw.get("device") or str(get_device())
    extractor = ImageFeatureExtractor(
        backbone=raw["backbone"],
        device=device,
        img_size=raw["img_size"],
        bbox_padding=raw["bbox_padding"],
    )

    # Ensure every configured output directory exists so DVC never sees a
    # missing output, even for datasets that aren't processed in this mode.
    for ds_cfg in raw["datasets"].values():
        os.makedirs(ds_cfg["features_dir"], exist_ok=True)

    for ds in _datasets_for_mode(mode, raw["datasets"]):
        extract_dataset(
            annotations_csv=ds["annotations_csv"],
            video_input_dir=ds["video_input_dir"],
            pose_dir=ds["pose_dir"],
            features_dir=ds["features_dir"],
            extractor=extractor,
        )
