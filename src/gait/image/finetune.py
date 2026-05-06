"""Per-frame classifiers on video crops — linear head and full fine-tune variants."""
from __future__ import annotations

import json
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.gait.image.extractor import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _xywh_to_xyxy,
    build_backbone,
)

logger = logging.getLogger(__name__)

_MEAN = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
_STD  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)


# ── Models ────────────────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    """Trainable linear head on pre-extracted CNN features.

    Backbone is NOT loaded here — feature extraction runs separately.
    A dropout + linear layer maps each frame's feature vector to
    log-class probabilities.
    """

    def __init__(self, in_features: int, n_classes: int = 3, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))


class ResNetFinetune(nn.Module):
    """Full ResNet backbone + linear head — all weights trainable.

    Parameters
    ----------
    backbone_name : str
        "resnet18" (512-dim) or "efficientnet_b0" (1280-dim).
    n_classes : int
    dropout : float
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        n_classes: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        backbone, in_features = build_backbone(backbone_name)
        self.backbone   = backbone
        self.drop       = nn.Dropout(dropout)
        self.fc         = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)  ImageNet-normalised crops

        Returns
        -------
        (B, n_classes) log-probabilities
        """
        return self.fc(self.drop(self.backbone(x)))


# ── Datasets ──────────────────────────────────────────────────────────────────

class GaitFrameDataset(Dataset):
    """Frame-level dataset: one (feature_vec, label) pair per video frame."""

    def __init__(self, records: list):
        feats  = np.concatenate([r.features for r in records], axis=0)
        labels = np.concatenate([r.labels   for r in records], axis=0)
        self.X = torch.from_numpy(feats).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class FrameCropDataset(Dataset):
    """Loads all video crops into RAM as uint8 at init; normalises in __getitem__.

    Memory usage ≈ n_frames × img_size² × 3 bytes.
    For 18 k frames at 224 px that is ≈ 2.7 GB.

    Parameters
    ----------
    records_info : list of (VideoRecord, pose_json_path, frame_start)
        ``frame_start`` is the first video-frame index included in
        ``record.labels``.
    img_size : int
        Crop size fed to the backbone (224 for ResNet18 / EfficientNet-B0).
    bbox_padding : float
        Same fractional padding used during feature extraction.
    """

    def __init__(
        self,
        records_info: list,
        img_size: int = 224,
        bbox_padding: float = 0.1,
    ):
        all_crops:  list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for rec, pose_json_path, frame_start in records_info:
            T     = len(rec.labels)
            crops = self._read_crops(
                rec.video_path, pose_json_path, frame_start, T, img_size, bbox_padding
            )
            all_crops.append(crops)
            all_labels.append(rec.labels)

        self.crops  = np.concatenate(all_crops, axis=0)   # (N, H, W, 3) uint8
        self.labels = torch.from_numpy(np.concatenate(all_labels)).long()

    @staticmethod
    def _read_crops(
        video_path: str,
        pose_json_path: str,
        frame_start: int,
        n_frames: int,
        img_size: int,
        bbox_padding: float,
    ) -> np.ndarray:
        with open(pose_json_path) as f:
            pose_data = json.load(f)["pose_data"]

        crops = np.zeros((n_frames, img_size, img_size, 3), dtype=np.uint8)
        cap   = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.warning("Cannot open video: %s", video_path)
            return crops

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

        for fi in range(n_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                logger.warning(
                    "Video ended early at frame %d in %s", frame_start + fi, video_path
                )
                break
            video_fi  = frame_start + fi
            if video_fi >= len(pose_data):
                continue
            frame_data = pose_data[video_fi]
            if not (frame_data and "bbox" in frame_data):
                continue
            h, w = frame_bgr.shape[:2]
            x1, y1, x2, y2 = _xywh_to_xyxy(frame_data["bbox"], h, w, bbox_padding)
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops[fi] = cv2.resize(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                (img_size, img_size),
                interpolation=cv2.INTER_LINEAR,
            )

        cap.release()
        return crops

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        crop   = self.crops[idx]                                   # (H, W, 3) uint8
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float().div(255.0)
        tensor = (tensor - _MEAN) / _STD
        return tensor, self.labels[idx]
