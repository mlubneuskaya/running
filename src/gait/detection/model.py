"""Temporal Convolutional Network for per-frame gait phase classification.

Non-causal (offline) design: each block uses symmetric padding so every output
frame can attend to both past and future context.  This is appropriate for
offline detection where the full sequence is available at inference time.

Architecture
------------
Input: (batch, T, n_features)
  TCN Block 1: Conv1d(64, kernel=3, dilation=1)  + WeightNorm + ReLU + Dropout + residual
  TCN Block 2: Conv1d(64, kernel=3, dilation=2)  + WeightNorm + ReLU + Dropout + residual
  TCN Block 3: Conv1d(64, kernel=3, dilation=4)  + WeightNorm + ReLU + Dropout + residual
  TCN Block 4: Conv1d(64, kernel=3, dilation=8)  + WeightNorm + ReLU + Dropout + residual
  Conv1d(3, kernel=1) → log-softmax per frame

Receptive field at default settings (4 blocks, kernel=3, dilations 1/2/4/8):
  RF = 1 + 2 * (kernel-1) * sum(dilations) = 1 + 2*2*15 = 61 frames
  (30 frames past + current frame + 30 frames future)
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class TCNBlock(nn.Module):
    """Single dilated non-causal TCN block with residual connection.

    Symmetric padding keeps output length == input length for any odd kernel size.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        # Symmetric padding: (kernel-1)//2 * dilation on each side.
        # Works exactly for odd kernel sizes (3, 5, 7, …).
        padding = (kernel_size - 1) // 2 * dilation
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
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — output is exactly (B, out_channels, T)
        out = self.conv(x)
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
        Kernel size for all dilated convolutions.  Must be odd.
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
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
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
        x = x.permute(0, 2, 1)               # (B, n_features, T)
        x = self.blocks(x)                    # (B, n_filters, T)
        logits = self.classifier(x)           # (B, n_classes, T)
        return torch.log_softmax(logits, dim=1).permute(0, 2, 1)  # (B, T, n_classes)

    @property
    def receptive_field(self) -> int:
        """Total number of frames in the context window (past + current + future).

        Symmetric padding: RF = 1 + 2 * sum_i (kernel-1) * dilation_i
        """
        rf = 1
        for i, block in enumerate(self.blocks):
            dilation = 2 ** i
            rf += 2 * (block.conv.weight.shape[2] - 1) * dilation
        return rf
