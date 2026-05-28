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
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.gait.detection.features import extract_features
from src.gait.gait_data.smoothing import smooth_pose_data

LABEL_MAP = {"left_stance": 0, "right_stance": 1, "flight": 2}
ANNOTATION_PADDING = 0  # annotations now cover the full visibility window; no extra padding needed
# contact + side → stance label
CONTACT_SIDE_MAP = {"left": 0, "right": 1}

RECORDING_FPS = 120


@dataclass
class VideoRecord:
    """Feature array and frame-level labels for one video."""

    video_path: str
    athlete: str          # unique athlete key — "{dataset}:{person_id}" when dataset is set
    features: np.ndarray  # (T, 22) float32
    labels: np.ndarray    # (T,)    int64
    person_id: int = -1   # numeric athlete ID from the annotation CSV
    dataset: str = ""     # source dataset name (e.g. "optojump", "tempos")


def _athlete_from_path(video_path: str) -> str:
    """Legacy name-based athlete key, used only when no dataset tag is given."""
    parts = os.path.splitext(os.path.basename(video_path))[0].split("_")
    return "_".join(parts[:-1])  # strip trailing test-id


def _athlete_key(video_path: str, person_id: int, dataset: str) -> str:
    """Return the appropriate athlete key.

    When *dataset* is set, returns ``"{dataset}:{person_id}"`` — a stable,
    cross-dataset-unique identifier that works even when filenames carry no
    name (e.g. tempos: ``1.MP4``).

    Falls back to the legacy path-derived name when *dataset* is empty so that
    callers that have not yet adopted the dataset tag are unaffected.
    """
    if dataset:
        return f"{dataset}:{person_id}"
    return _athlete_from_path(video_path)


def _pose_path(video_path: str) -> str:
    """Map a video_path (as stored in the CSV) to the corresponding pose JSON."""
    base = os.path.splitext(video_path)[0]
    # video_path is like  data/input/optojump/study_N/athlete_M.mov
    # pose JSON is at     data/output/tuned_yolo/000332/optojump/study_N/athlete_M.json
    rel = os.path.relpath(base, "data/input/optojump")
    return os.path.join("data/output/pose/tuned_yolo/000332/optojump", rel + ".json")  # TODO make model id independent


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


def _pad_label(ann_sorted: pd.DataFrame, end: str) -> int:
    """Return the label to use for padding frames outside the annotation window.

    Leading padding (end="first"):
        If the first annotated frame is contact → before it was flight.
        If the first annotated frame is flight  → before it was contact;
          infer side as the opposite of the first contact seen in the sequence
          (since feet alternate, the contact before an initial flight is the
          other foot).

    Trailing padding (end="last"):
        Continues the state of the last annotated frame.
          last = contact right/left → pad as contact same side
          last = flight             → pad as flight
    """
    if end == "last":
        last_row = ann_sorted.iloc[-1]
        if last_row["label"] == "contact":
            return CONTACT_SIDE_MAP.get(last_row.get("side"), LABEL_MAP["flight"])
        return LABEL_MAP["flight"]

    # end == "first"
    first_row = ann_sorted.iloc[0]
    if first_row["label"] == "contact":
        # video starts mid-contact → frames before it were flight
        return LABEL_MAP["flight"]
    # video starts mid-flight → frames before it were contact
    contact_rows = ann_sorted[ann_sorted["label"] == "contact"]
    if contact_rows.empty:
        return LABEL_MAP["flight"]
    first_contact_side = contact_rows.iloc[0].get("side", None)
    other_side = "right" if first_contact_side == "left" else "left"
    return CONTACT_SIDE_MAP.get(other_side, LABEL_MAP["flight"])


def _annotation_to_labels(
    ann: pd.DataFrame,
    first_frame: int,
    n_frames: int,
) -> np.ndarray:
    """Build a per-frame label array for the pose window [first_frame, first_frame+n_frames).

    Only the annotated range is taken from the CSV.  Pose frames that fall
    before or after the annotation window are filled with a padding label
    inferred from the state at the annotation boundary (see _pad_label).
    """
    ann_sorted = ann.sort_values("frame_number")
    first_ann  = int(ann_sorted["frame_number"].iloc[0])
    last_ann   = int(ann_sorted["frame_number"].iloc[-1])

    leading_label  = _pad_label(ann_sorted, "first")
    trailing_label = _pad_label(ann_sorted, "last")

    labels = np.full(n_frames, LABEL_MAP["flight"], dtype=np.int64)

    # leading padding: pose frames before the annotation window
    leading_end = max(0, min(first_ann - first_frame, n_frames))
    labels[:leading_end] = leading_label

    # annotated frames
    for _, row in ann_sorted.iterrows():
        fi = int(row["frame_number"]) - first_frame
        if not (0 <= fi < n_frames):
            continue
        if row["label"] == "contact":
            labels[fi] = CONTACT_SIDE_MAP.get(row.get("side"), LABEL_MAP["flight"])
        else:
            labels[fi] = LABEL_MAP["flight"]

    # trailing padding: pose frames after the annotation window
    trailing_start = max(0, min(last_ann - first_frame + 1, n_frames))
    labels[trailing_start:] = trailing_label

    return labels


def load_dataset_with_pose(
    annotations_csv: str,
    fps: float = float(RECORDING_FPS),
    dataset: str = "",
) -> list[tuple[VideoRecord, pd.DataFrame]]:
    """Like :func:`load_dataset` but also returns the clipped pose DataFrame.

    The pose DataFrame is clipped to exactly the same window as the VideoRecord's
    features and labels, so ``len(pose_df) == len(record.labels)`` for every pair.

    Parameters
    ----------
    dataset : str
        Dataset tag written into every ``VideoRecord.dataset``.  When set,
        ``record.athlete`` is ``"{dataset}:{person_id}"``; otherwise the legacy
        path-derived name is used.

    Returns
    -------
    list of (VideoRecord, pose_df) tuples
    """
    ann_df = pd.read_csv(annotations_csv)
    out: list[tuple[VideoRecord, pd.DataFrame]] = []

    for video_path, group in ann_df.groupby("video_path"):
        pose_path = _pose_path(video_path)
        smooth = _load_smooth(pose_path)
        if smooth is None:
            continue

        feats = extract_features(smooth, fps=fps)
        T = len(feats)
        first_frame = int(smooth["frame_index"].iloc[0])
        labels = _annotation_to_labels(group, first_frame, T)

        # Clip to the annotated visibility window exactly
        first_ann = int(group["frame_number"].min())
        last_ann  = int(group["frame_number"].max())
        start_idx = max(0, first_ann - ANNOTATION_PADDING - first_frame)
        end_idx   = min(T, last_ann  + ANNOTATION_PADDING - first_frame + 1)

        feats  = feats[start_idx:end_idx]
        labels = labels[start_idx:end_idx]
        pose_df = smooth.iloc[start_idx:end_idx].reset_index(drop=True)

        person_id = int(group["person_id"].iloc[0]) if "person_id" in group.columns else -1
        record = VideoRecord(
            video_path=video_path,
            athlete=_athlete_key(video_path, person_id, dataset),
            features=feats,
            labels=labels,
            person_id=person_id,
            dataset=dataset,
        )
        out.append((record, pose_df))

    return out


def load_dataset(
    annotations_csv: str,
    fps: float = float(RECORDING_FPS),
    dataset: str = "",
) -> list[VideoRecord]:
    """Load all annotated videos into VideoRecord objects.

    Videos whose pose JSON is missing or whose smoothed DataFrame is empty
    are silently skipped.

    Parameters
    ----------
    dataset : str
        Dataset tag written into every ``VideoRecord.dataset``.  When set,
        ``record.athlete`` is ``"{dataset}:{person_id}"``; otherwise the legacy
        path-derived name is used.
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
        first_frame = int(smooth["frame_index"].iloc[0])
        labels = _annotation_to_labels(group, first_frame, T)

        # Clip to the annotated visibility window exactly
        first_ann = int(group["frame_number"].min())
        last_ann  = int(group["frame_number"].max())
        start_idx = max(0, first_ann - ANNOTATION_PADDING - first_frame)
        end_idx   = min(T, last_ann  + ANNOTATION_PADDING - first_frame + 1)
        feats  = feats[start_idx:end_idx]
        labels = labels[start_idx:end_idx]

        person_id = int(group["person_id"].iloc[0]) if "person_id" in group.columns else -1
        records.append(VideoRecord(
            video_path=video_path,
            athlete=_athlete_key(video_path, person_id, dataset),
            features=feats,
            labels=labels,
            person_id=person_id,
            dataset=dataset,
        ))

    return records


def load_datasets(
    csv_paths: dict[str, str],
    fps: float = float(RECORDING_FPS),
) -> list[VideoRecord]:
    """Load and combine multiple datasets.

    Parameters
    ----------
    csv_paths : dict[str, str]
        Mapping from dataset name to annotation CSV path.  Each record will
        have ``record.dataset`` set to the dict key and ``record.athlete`` set
        to ``"{dataset_name}:{person_id}"``.
    fps : float
        Recording frame rate.

    Returns
    -------
    list[VideoRecord]
        Combined records from all datasets, tagged by dataset name.

    Example
    -------
    >>> records = load_datasets(
    ...     {"optojump": "data/output/annotations/optojump/visibility_annotations.csv",
    ...      "tempos":   "data/output/annotations/tempos/ml_labels.csv"},
    ...     fps=120.0,
    ... )
    """
    records: list[VideoRecord] = []
    for dataset_name, csv_path in csv_paths.items():
        records.extend(load_dataset(csv_path, fps=fps, dataset=dataset_name))
    return records


# ── datasets ────────────────────────────────────────────────────────────────

def _select(features: np.ndarray, feature_idx: list[int] | None) -> np.ndarray:
    return features if feature_idx is None else features[:, feature_idx]


class GaitWindowDataset(Dataset):
    """Training dataset: random window of fixed length sampled per video."""

    def __init__(
        self,
        records: list[VideoRecord],
        window_size: int = 75,
        feature_idx: list[int] | None = None,
    ):
        self.records = records
        self.window_size = window_size
        self.feature_idx = feature_idx

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        T = len(rec.features)
        if T <= self.window_size:
            pad = self.window_size - T
            feats = np.pad(_select(rec.features, self.feature_idx), ((0, pad), (0, 0)), mode="edge")
            labels = np.pad(rec.labels, (0, pad), mode="edge")
            mask = np.array([True] * T + [False] * pad, dtype=bool)
        else:
            start = np.random.randint(0, T - self.window_size + 1)
            feats = _select(rec.features, self.feature_idx)[start: start + self.window_size]
            labels = rec.labels[start: start + self.window_size]
            mask = np.ones(self.window_size, dtype=bool)

        return (
            torch.from_numpy(feats),
            torch.from_numpy(labels),
            torch.from_numpy(mask),
        )


class GaitSequenceDataset(Dataset):
    """Inference/validation dataset: whole sequences, padded to batch max length."""

    def __init__(self, records: list[VideoRecord], feature_idx: list[int] | None = None):
        self.records = records
        self.feature_idx = feature_idx

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        return (
            torch.from_numpy(_select(rec.features, self.feature_idx)),
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

def train_test_split(
    records: list[VideoRecord],
    n_test: dict[str, int],
    seed: int = 42,
) -> tuple[list[VideoRecord], list[VideoRecord], list[str]]:
    """Randomly split records into train and test sets at the athlete level.

    The split is fully deterministic: athlete lists are sorted before sampling
    so adding new videos for existing athletes never shifts the test set.

    Parameters
    ----------
    records : list[VideoRecord]
        All records to split.
    n_test : dict[str, int]
        How many athletes from each dataset go to the test split.
        Keys must match the ``record.dataset`` values of the records you pass.
        Example: ``{"optojump": 3, "tempos": 2}`` — draws 3 athletes from the
        optojump subset and 2 from the tempos subset independently.
        Set a value to 0 to keep all athletes of a dataset in training.
        Load this from ``configs/split.yaml`` via the ``split_config`` key in
        your experiment config.
    seed : int
        Random seed for reproducibility.  Load from ``configs/split.yaml``.

    Returns
    -------
    (train_records, test_records, test_athletes)
        ``test_athletes`` is a sorted list of ``record.athlete`` strings.
    """
    if not isinstance(n_test, dict) or len(n_test) == 0:
        raise ValueError(
            "'n_test' must be a dict with at least one dataset entry, "
            "e.g. {\"optojump\": 2, \"tempos\": 0}.\n"
            "Use 0 to keep all athletes of a dataset in training.\n"
            "Add 'split_config: \"configs/split.yaml\"' to your experiment config."
        )
    rng = np.random.default_rng(seed)
    test_set: set[str] = set()
    for ds_name, k in n_test.items():
        ds_athletes = sorted({r.athlete for r in records if r.dataset == ds_name})
        if k > len(ds_athletes):
            raise ValueError(
                f"Dataset '{ds_name}' has {len(ds_athletes)} athlete(s) but "
                f"n_test['{ds_name}']={k} was requested."
            )
        if k > 0:
            test_set.update(rng.choice(ds_athletes, k, replace=False).tolist())

    train = [r for r in records if r.athlete not in test_set]
    test  = [r for r in records if r.athlete in test_set]
    return train, test, sorted(test_set)


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
