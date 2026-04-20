"""Dataset loading for image-based gait phase detection.

Loads VideoRecord objects backed by pre-extracted CNN feature files (.npy)
instead of computing kinematic features from smoothed pose data.

The label-building logic is identical to gait_data/dataset.py.  All training
dataset classes (GaitWindowDataset, GaitSequenceDataset, loao_splits, etc.)
work unchanged because they only access record.features and record.labels.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from src.gait.gait_data.dataset import (
    ANNOTATION_PADDING,
    VideoRecord,
    _annotation_to_labels,
    _athlete_from_path,
)

logger = logging.getLogger(__name__)


def _feature_path(video_path: str, features_dir: str, video_input_dir: str) -> str:
    """Map a video path to its pre-extracted .npy feature file."""
    base = os.path.splitext(video_path)[0]
    rel  = os.path.relpath(base, video_input_dir)
    return os.path.join(features_dir, rel + ".npy")


def load_image_dataset(
    annotations_csv: str,
    features_dir: str,
    video_input_dir: str = "data/input/optojump",
) -> list[VideoRecord]:
    """Load annotated videos using pre-extracted CNN feature files.

    Videos whose .npy feature file is missing raise FileNotFoundError.
    Features are clipped to the annotation window ± ANNOTATION_PADDING frames,
    matching the windowing in load_dataset().

    Parameters
    ----------
    annotations_csv : str
    features_dir : str
        Directory where img_feature_extract.py saved the .npy files.
    video_input_dir : str
        Root of the video input tree (used to compute relative paths).

    Returns
    -------
    list[VideoRecord]
        features shape: (T, feature_dim)   labels shape: (T,)
    """
    ann_df  = pd.read_csv(annotations_csv)
    records: list[VideoRecord] = []

    for video_path, group in ann_df.groupby("video_path"):
        feat_path = _feature_path(video_path, features_dir, video_input_dir)

        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                f"Feature file not found: {feat_path}\n"
                f"Run img_feature_extract before loading the image dataset."
            )

        features = np.load(feat_path)    # (T, feature_dim), float32
        T        = len(features)

        # Frame index 0 in the feature array corresponds to video frame 0.
        # Annotation frame_numbers are 0-indexed 120fps frame numbers — they
        # align directly with the feature array index.
        first_frame = 0
        labels = _annotation_to_labels(group, first_frame, T)

        first_ann = int(group["frame_number"].min())
        last_ann  = int(group["frame_number"].max())
        start_idx = max(0, first_ann - ANNOTATION_PADDING - first_frame)
        end_idx   = min(T, last_ann  + ANNOTATION_PADDING - first_frame + 1)

        if start_idx >= end_idx:
            raise ValueError(
                f"Empty clip window for {video_path}: "
                f"first_ann={first_ann}, last_ann={last_ann}, T={T}"
            )

        records.append(VideoRecord(
            video_path=video_path,
            athlete=_athlete_from_path(video_path),
            features=features[start_idx:end_idx],
            labels=labels[start_idx:end_idx],
        ))

    logger.info("Loaded %d image records from %s", len(records), features_dir)
    return records
