"""Per-frame linear classifier on top of frozen ResNet features."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
        """
        Parameters
        ----------
        x : (*, in_features)

        Returns
        -------
        (*, n_classes) log-probabilities
        """
        return torch.log_softmax(self.fc(self.drop(x)), dim=-1)


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
