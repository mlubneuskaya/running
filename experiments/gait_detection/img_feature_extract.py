"""Image feature extraction — run CNN backbone on video crops from pose bboxes.

Reads the annotations CSV to find all videos, maps each to its pose JSON (for
bounding boxes), extracts per-frame CNN features, and saves one .npy file per
video under features_dir.

Usage
-----
    python -m experiments.gait_detection.img_feature_extract
    python -m experiments.gait_detection.img_feature_extract --config configs/experiments/img_feature_extract.yaml

Output
------
    <features_dir>/<relative_video_stem>.npy   shape (T, feature_dim), float32
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from src.gait.image.extractor import ImageFeatureExtractor
from src.pose.utils.load_config import load_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "configs/experiments/img_feature_extract.yaml"


def _pose_json_path(video_path: str, pose_dir: str, video_input_dir: str) -> str:
    base = os.path.splitext(video_path)[0]
    rel  = os.path.relpath(base, video_input_dir)
    return os.path.join(pose_dir, rel + ".json")


def _feature_out_path(video_path: str, features_dir: str, video_input_dir: str) -> str:
    base = os.path.splitext(video_path)[0]
    rel  = os.path.relpath(base, video_input_dir)
    return os.path.join(features_dir, rel + ".npy")


def main(
    annotations_csv: str,
    video_input_dir: str,
    pose_dir: str,
    features_dir: str,
    extractor: ImageFeatureExtractor,
) -> None:
    ann_df = pd.read_csv(annotations_csv)
    video_paths = ann_df["video_path"].unique().tolist()
    logger.info("%d videos to process.", len(video_paths))

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

        import numpy as np
        np.save(out_path, features)
        logger.info("  Saved %s  shape=%s", out_path, features.shape)

    logger.info("Feature extraction complete. Features in: %s", features_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raw = load_config(args.config)

    extractor = ImageFeatureExtractor(
        backbone=raw["backbone"],
        device=raw["device"],
        img_size=raw["img_size"],
        bbox_padding=raw["bbox_padding"],
    )
    main(
        annotations_csv=raw["annotations_csv"],
        video_input_dir=raw["video_input_dir"],
        pose_dir=raw["pose_dir"],
        features_dir=raw["features_dir"],
        extractor=extractor,
    )
