"""
CNN Downsampling Bridge for the Sandwich architecture.

This module sits between two stages of a sequence model, downsampling
continuous hidden states from full sequence length to a compressed length.

Unlike the CNN Stem (which operates on token IDs), this bridge operates on
continuous representations (B, L, D) from encoder hidden states.

Architecture:
    Hidden states (B, L, D)
    → LayerNorm (stabilize input)
    → transpose → (B, D, L)
    → N × ResidualDownsampleBlock (each halves L via MaxPool(2))
    → transpose → (B, L // 2^N, D)
    → LayerNorm (stabilize output)
    → (B, L_down, D)

Design choices:
    - Input/output LayerNorm for training stability at the stage boundary
    - Reuses ResidualDownsampleBlock from cnn_stem.py for consistent downsampling
    - Maintains d_model dimension throughout (no channel expansion/reduction)
    - Default: 7 stages of 2x = 128x total (131072 → 1024)
"""

import torch
import torch.nn as nn

from src.models.cnn_stem import ResidualDownsampleBlock


class CNNDownsampleBridge(nn.Module):
    """CNN bridge for downsampling hidden states between encoder stages.

    Takes continuous hidden states (B, L, D) and downsamples the sequence
    length by a factor of 2^num_downsample_stages, producing (B, L/factor, D).

    This is designed to sit between a "pre-downsample" encoder stage operating
    at full sequence length and a "post-downsample" stage operating at reduced
    sequence length, forming a "sandwich" architecture.

    Args:
        d_model: Hidden dimension of the encoder (input and output dimension).
        num_downsample_stages: Number of 2x downsampling stages (default: 7 → 128x).
        kernel_size: Convolution kernel size in residual blocks (default: 5).
        dropout: Dropout rate in convolution blocks (default: 0.1).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_downsample_stages: int = 7,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_downsample_stages = num_downsample_stages
        self.total_downsample_factor = 2 ** num_downsample_stages

        # Input normalization to stabilize the transition from encoder hidden states
        self.input_norm = nn.LayerNorm(d_model)

        # Progressive downsampling blocks
        # Each block: Conv1d → BN → GELU → Conv1d → BN → (+skip) → GELU → MaxPool(2)
        self.blocks = nn.ModuleList()
        for _ in range(num_downsample_stages):
            self.blocks.append(
                ResidualDownsampleBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )

        # Output normalization for stable input to the next encoder stage
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample hidden states.

        Args:
            x: (B, L, D) encoder hidden states at full sequence length.

        Returns:
            (B, L // 2^N, D) downsampled hidden states.
        """
        # Normalize input
        x = self.input_norm(x)                 # (B, L, D)

        # Transpose for Conv1d: (B, L, D) → (B, D, L)
        x = x.transpose(1, 2)

        # Progressive 2x downsampling
        for block in self.blocks:
            x = block(x)                       # (B, D, L) → (B, D, L//2) each stage

        # Transpose back: (B, D, L_down) → (B, L_down, D)
        x = x.transpose(1, 2)

        # Normalize output
        x = self.output_norm(x)                # (B, L_down, D)

        return x

    def get_output_length(self, input_length: int) -> int:
        """Compute the output sequence length for a given input length."""
        return input_length // self.total_downsample_factor
