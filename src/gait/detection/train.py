"""Training loop for the TCN gait phase detector."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Run one training epoch.  Returns mean loss over batches."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for feats, labels, masks in loader:
        feats = feats.to(device)          # (B, T, 22)
        labels = labels.to(device)        # (B, T)
        masks = masks.to(device)          # (B, T)

        optimizer.zero_grad()
        loss = calculate_loss(criterion, feats, labels, masks, model)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def calculate_loss(criterion: nn.Module, feats, labels, masks, model: nn.Module) -> nn.modules.loss._Loss:
    logits = model(feats)  # (B, T, C)

    # Flatten, apply mask
    B, T, C = logits.shape
    logits_flat = logits.reshape(B * T, C)[masks.reshape(-1)]
    labels_flat = labels.reshape(B * T)[masks.reshape(-1)]

    loss = criterion(logits_flat, labels_flat)
    return loss


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one validation epoch.  Returns mean loss over batches."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for feats, labels, masks in loader:
        feats = feats.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        loss = calculate_loss(criterion, feats, labels, masks, model)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@dataclass
class TrainerConfig:
    lr: float = 1e-3
    batch_size: int = 16
    max_epochs: int = 200
    early_stopping_patience: int = 20
    lr_schedule_factor: float = 0.5
    lr_schedule_patience: int = 10
    max_grad_norm: float = 1.0
    dropout: float = 0.2
    window_size: int = 75
    checkpoint_path: str | None = None
    device: str | None = None  # None = auto-detect


class Trainer:
    """Encapsulates the training loop with early stopping and LR scheduling.

    Parameters
    ----------
    model : nn.Module
    class_weights : torch.Tensor
        Per-class weights for weighted cross-entropy loss.
    config : TrainerConfig
    trial : optuna.Trial | None
        If provided, calls ``trial.report`` and ``trial.should_prune`` each epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        class_weights: torch.Tensor,
        config: TrainerConfig | None = None,
        trial=None,
    ):
        self.config = config or TrainerConfig()
        self.device = get_device(self.config.device)
        self.model = model.to(self.device)
        self.trial = trial

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device)
        )
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.lr_schedule_factor,
            patience=self.config.lr_schedule_patience,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epoch_callback: Callable[[int, float, float], None] | None = None,
    ) -> dict:
        """Train until max_epochs or early stopping.

        Returns
        -------
        dict
            ``{"best_val_loss": float, "best_epoch": int, "epochs_trained": int,
               "train_losses": list[float], "val_losses": list[float]}``
        """
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        train_losses: list[float] = []
        val_losses:   list[float] = []
        epoch = 0

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = train_epoch(
                self.model, train_loader, self.optimizer,
                self.criterion, self.device, self.config.max_grad_norm,
            )
            val_loss = val_epoch(self.model, val_loader, self.criterion, self.device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self.scheduler.step(val_loss)

            if epoch_callback is not None:
                epoch_callback(epoch, train_loss, val_loss)

            # Optuna pruning
            if self.trial is not None:
                self.trial.report(val_loss, epoch)
                import optuna
                if self.trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                if self.config.checkpoint_path:
                    os.makedirs(os.path.dirname(self.config.checkpoint_path) or ".", exist_ok=True)
                    torch.save(self.model.state_dict(), self.config.checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info("Early stopping at epoch %d (best val loss %.4f at epoch %d)", epoch, best_val_loss, best_epoch)
                    break

        if self.config.checkpoint_path and os.path.exists(self.config.checkpoint_path):
            self.model.load_state_dict(torch.load(self.config.checkpoint_path, map_location=self.device))

        return {
            "best_val_loss":  best_val_loss,
            "best_epoch":     best_epoch,
            "epochs_trained": epoch,
            "train_losses":   train_losses,
            "val_losses":     val_losses,
        }
