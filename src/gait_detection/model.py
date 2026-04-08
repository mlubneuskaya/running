"""Temporal Convolutional Network for per-frame gait phase classification.

Architecture
------------
Input: (batch, T, 22)
  TCN Block 1: Conv1d(64, kernel=3, dilation=1)  + WeightNorm + ReLU + Dropout + residual
  TCN Block 2: Conv1d(64, kernel=3, dilation=2)  + WeightNorm + ReLU + Dropout + residual
  TCN Block 3: Conv1d(64, kernel=3, dilation=4)  + WeightNorm + ReLU + Dropout + residual
  TCN Block 4: Conv1d(64, kernel=3, dilation=8)  + WeightNorm + ReLU + Dropout + residual
  Conv1d(3, kernel=1) → log-softmax per frame

Receptive field at default settings (4 blocks, kernel=3, dilations 1/2/4/8):
  RF = 1 + 2 * (kernel-1) * sum(dilations) = 1 + 2*2*(1+2+4+8) = 31 frames
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TCNBlock(nn.Module):
    """Single dilated causal TCN block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        # Causal padding: pad only on the left so output length == input length
        padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # 1×1 projection for residual when channels differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self._padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        out = self.conv(x)
        # Remove the extra causal padding on the right
        if self._padding > 0:
            out = out[:, :, : -self._padding]
        out = self.relu(out)
        out = self.dropout(out)
        return out + self.residual(x)


class TCN(nn.Module):
    """Temporal Convolutional Network for gait phase classification.

    Parameters
    ----------
    n_features : int
        Number of input features per frame (default 22).
    n_classes : int
        Number of output classes (default 3).
    n_blocks : int
        Number of dilated TCN blocks.  Dilations are 1, 2, 4, … 2^(n_blocks-1).
    n_filters : int
        Number of convolutional filters in each block.
    kernel_size : int
        Kernel size for all dilated convolutions.
    dropout : float
        Dropout probability applied after each block's activation.
    """

    def __init__(
        self,
        n_features: int = 22,
        n_classes: int = 3,
        n_blocks: int = 4,
        n_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            in_ch = n_features if i == 0 else n_filters
            blocks.append(
                TCNBlock(
                    in_channels=in_ch,
                    out_channels=n_filters,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Conv1d(n_filters, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, n_features).

        Returns
        -------
        torch.Tensor
            Log-probabilities, shape (B, T, n_classes).
        """
        # Conv1d expects (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.blocks(x)
        logits = self.classifier(x)          # (B, n_classes, T)
        return torch.log_softmax(logits, dim=1).permute(0, 2, 1)  # (B, T, n_classes)

    @property
    def receptive_field(self) -> int:
        """Number of frames the model's context window covers.

        Causal (one-sided) padding: RF = 1 + sum_i (kernel-1) * dilation_i
        """
        rf = 1
        for i, block in enumerate(self.blocks):
            dilation = 2 ** i
            rf += (block.conv.weight.shape[2] - 1) * dilation
        return rf
