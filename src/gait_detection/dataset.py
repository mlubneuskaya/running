"""Dataset utilities for gait phase detection.

Label mapping
-------------
    0 = left_stance
    1 = right_stance
    2 = flight

VideoRecord
    A loaded, feature-extracted video ready for training/evaluation.

GaitWindowDataset
    Random-window sampler for training (window_size random frames per video per epoch).

GaitSequenceDataset
    Yields entire sequences for inference / validation.

loao_splits
    Generator of (train_records, val_records) for leave-one-athlete-out CV.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.gait_analysis.data_cleaning.data_smoothing import smooth_pose_data
from src.gait_detection.features import extract_features

LABEL_MAP = {"left_stance": 0, "right_stance": 1, "flight": 2}
# contact + side → stance label
CONTACT_SIDE_MAP = {"left": 0, "right": 1}

RECORDING_FPS = 120


@dataclass
class VideoRecord:
    """Feature array and frame-level labels for one video."""

    video_path: str
    athlete: str
    features: np.ndarray     # (T, 22) float32
    labels: np.ndarray       # (T,)    int64


def _athlete_from_path(video_path: str) -> str:
    parts = os.path.splitext(os.path.basename(video_path))[0].split("_")
    return "_".join(parts[:-1])  # strip trailing test-id


def _pose_path(video_path: str) -> str:
    """Map a video_path (as stored in the CSV) to the corresponding pose JSON."""
    base = os.path.splitext(video_path)[0]
    # video_path is like  data/input/optojump/study_N/athlete_M.mov
    # pose JSON is at     data/output/tuned_yolo/000332/optojump/study_N/athlete_M.json
    rel = os.path.relpath(base, "data/input/optojump")
    return os.path.join("data/output/tuned_yolo/000332/optojump", rel + ".json")


_KEYPOINTS = [
    "left_heel", "right_heel",
    "left_big_toe", "right_big_toe",
    "left_ankle", "right_ankle",
    "left_knee", "right_knee",
    "left_hip", "right_hip",
]
_ANCHORS = [
    "left_heel", "right_heel",
    "left_big_toe", "right_big_toe",
]


def _load_smooth(pose_path: str) -> pd.DataFrame | None:
    if not os.path.exists(pose_path):
        return None
    with open(pose_path) as f:
        pose_data = json.load(f)['pose_data']
    smooth = smooth_pose_data(
        pose_data,
        keypoints=_KEYPOINTS,
        anchors=_ANCHORS,
        keys_to_exclude={"bbox"},
        fps=RECORDING_FPS,
        cutoff=6.0,
    )
    if smooth.empty:
        return None
    # Correct timestamps: cv2 reports 30 fps but actual rate is 120
    CV2_FPS = 30
    smooth["timestamp_ms"] = smooth["timestamp_ms"] * CV2_FPS / RECORDING_FPS
    smooth["frame_index"] = (smooth["timestamp_ms"] / 1000 * RECORDING_FPS).round().astype(int)
    return smooth


def _annotation_to_labels(ann: pd.DataFrame, n_frames: int) -> np.ndarray:
    """Convert annotation DataFrame rows to a per-frame label array."""
    labels = np.full(n_frames, LABEL_MAP["flight"], dtype=np.int64)
    for _, row in ann.iterrows():
        fi = int(row["frame_number"])
        if fi >= n_frames:
            continue
        if row["label"] == "contact":
            side = row.get("side", None)
            labels[fi] = CONTACT_SIDE_MAP.get(side, LABEL_MAP["flight"])
    return labels


def load_dataset(annotations_csv: str, fps: float = float(RECORDING_FPS)) -> list[VideoRecord]:
    """Load all annotated videos into VideoRecord objects.

    Videos whose pose JSON is missing or whose smoothed DataFrame is empty
    are silently skipped.
    """
    ann_df = pd.read_csv(annotations_csv)
    records: list[VideoRecord] = []

    for video_path, group in ann_df.groupby("video_path"):
        pose_path = _pose_path(video_path)
        smooth = _load_smooth(pose_path)
        if smooth is None:
            continue

        feats = extract_features(smooth, fps=fps)
        T = len(feats)

        # Align annotations by frame_index present in smooth
        first_frame = int(smooth["frame_index"].iloc[0])
        labels_full = _annotation_to_labels(group, first_frame + T)
        labels = labels_full[first_frame: first_frame + T]

        athlete = _athlete_from_path(video_path)
        records.append(VideoRecord(
            video_path=video_path,
            athlete=athlete,
            features=feats,
            labels=labels,
        ))

    return records


# ── datasets ────────────────────────────────────────────────────────────────

class GaitWindowDataset(Dataset):
    """Training dataset: random window of fixed length sampled per video."""

    def __init__(self, records: list[VideoRecord], window_size: int = 75):
        self.records = records
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        T = len(rec.features)
        if T <= self.window_size:
            # Pad with edge values
            pad = self.window_size - T
            feats = np.pad(rec.features, ((0, pad), (0, 0)), mode="edge")
            labels = np.pad(rec.labels, (0, pad), mode="edge")
            mask = np.array([True] * T + [False] * pad, dtype=bool)
        else:
            start = np.random.randint(0, T - self.window_size + 1)
            feats = rec.features[start: start + self.window_size]
            labels = rec.labels[start: start + self.window_size]
            mask = np.ones(self.window_size, dtype=bool)

        return (
            torch.from_numpy(feats),
            torch.from_numpy(labels),
            torch.from_numpy(mask),
        )


class GaitSequenceDataset(Dataset):
    """Inference/validation dataset: whole sequences, padded to batch max length."""

    def __init__(self, records: list[VideoRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        return (
            torch.from_numpy(rec.features),
            torch.from_numpy(rec.labels),
            torch.ones(len(rec.features), dtype=torch.bool),
        )

    @staticmethod
    def collate(batch):
        """Pad sequences in batch to the length of the longest one."""
        feats, labels, masks = zip(*batch)
        max_len = max(f.shape[0] for f in feats)
        n_feat = feats[0].shape[1]

        padded_feats = torch.zeros(len(feats), max_len, n_feat)
        padded_labels = torch.zeros(len(labels), max_len, dtype=torch.long)
        padded_masks = torch.zeros(len(masks), max_len, dtype=torch.bool)

        for i, (f, l, m) in enumerate(zip(feats, labels, masks)):
            t = f.shape[0]
            padded_feats[i, :t] = f
            padded_labels[i, :t] = l
            padded_masks[i, :t] = m

        return padded_feats, padded_labels, padded_masks


# ── splits ───────────────────────────────────────────────────────────────────

def loao_splits(records: list[VideoRecord]) -> Iterator[tuple[list[VideoRecord], list[VideoRecord], str]]:
    """Leave-one-athlete-out cross-validation splits.

    Yields
    ------
    (train_records, val_records, held_out_athlete)
    """
    athletes = sorted({r.athlete for r in records})
    for athlete in athletes:
        train = [r for r in records if r.athlete != athlete]
        val = [r for r in records if r.athlete == athlete]
        yield train, val, athlete


def tuning_split(
    records: list[VideoRecord],
    n_val_athletes: int = 4,
    seed: int = 42,
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    """Hold out a fixed set of athletes for hyperparameter tuning.

    Returns
    -------
    (train_records, val_records)
    """
    rng = np.random.default_rng(seed)
    athletes = sorted({r.athlete for r in records})
    val_athletes = set(rng.choice(athletes, size=n_val_athletes, replace=False).tolist())
    train = [r for r in records if r.athlete not in val_athletes]
    val = [r for r in records if r.athlete in val_athletes]
    return train, val


def compute_class_weights(records: list[VideoRecord], n_classes: int = 3) -> torch.Tensor:
    """Compute class weights for weighted cross-entropy loss.

    weight_c = total_frames / (n_classes * count_c)
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    for r in records:
        for c in range(n_classes):
            counts[c] += (r.labels == c).sum()
    total = counts.sum()
    weights = total / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
