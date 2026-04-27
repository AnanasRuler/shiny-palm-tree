"""
CNN Downsampling Stem and Simple Projection Head for genomic sequence modeling.

Architecture overview:
  Input tokens (B, L) → Embedding → Conv tower with progressive 2x downsampling
  → (B, L // 2^N, d_model) downsampled continuous representations

This replaces the post-encoder pooling in SFFuseHead, moving all spatial
downsampling to the beginning of the model. The encoder then processes
much shorter sequences (e.g., 1024 instead of 131072), greatly reducing
computational cost.

Design choices:
  - Residual convolution blocks with MaxPool for stable 2x downsampling
  - Wide initial kernel (15) for capturing local DNA patterns
  - BatchNorm + GELU activation throughout
  - Progressive channel expansion (optional)
  - LayerNorm before final projection for training stability
"""

import torch
import torch.nn as nn


class TargetLengthCrop1D(nn.Module):
    """Crops the center of a sequence to the target length.
    
    Works on tensors of shape (B, L, C), cropping along dimension 1.
    """
    
    def __init__(self, target_length: int):
        super().__init__()
        self.target_length = target_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len < self.target_length:
            raise ValueError(
                f"Sequence length {seq_len} is smaller than target length {self.target_length}"
            )
        start = (seq_len - self.target_length) // 2
        return x[:, start : start + self.target_length, :]


class ResidualDownsampleBlock(nn.Module):
    """Residual convolution block with 2x downsampling via MaxPool.
    
    Architecture:
        x → Conv1d → BN → GELU → Dropout → Conv1d → BN → (+skip) → GELU → MaxPool(2)
    
    The skip connection uses a 1x1 conv if input/output channels differ.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size (default: 5).
        dropout: Dropout rate between conv layers (default: 0.1).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        
        # Skip connection with optional channel projection
        self.skip_proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        
        self.activation = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L)
        Returns:
            (B, C_out, L // 2)
        """
        residual = self.skip_proj(x)
        out = self.conv_block(x)
        out = out + residual
        out = self.activation(out)
        out = self.pool(out)
        return out


class CNNDownsampleStem(nn.Module):
    """CNN downsampling stem for genomic sequences.
    
    Takes raw token IDs and produces downsampled continuous representations.
    Uses progressive 2x downsampling via residual convolution blocks.
    
    Default configuration: 7 stages of 2x = 128x total downsampling.
    For input length 131072: output length = 131072 / 128 = 1024.
    
    Architecture:
        Token IDs (B, L)
        → Embedding (B, L, C0)
        → Initial wide-kernel Conv1d for local pattern extraction
        → N × ResidualDownsampleBlock (each halves the sequence length)
        → LayerNorm + Linear projection to d_model
        → (B, L // 2^N, d_model)
    
    Args:
        vocab_size: Number of token types (default: 12 for DNA vocabulary).
        d_model: Output feature dimension (should match encoder's d_model).
        stem_channels: Hidden channel widths per stage. Can be:
            - None: all stages use d_model channels.
            - int: all stages use this channel width.
            - list[int]: per-stage channel widths (length must equal num_downsample_stages).
        num_downsample_stages: Number of 2x downsampling stages (default: 7 for 128x).
        kernel_size: Conv kernel size in residual blocks (default: 5).
        initial_kernel_size: Initial conv kernel size for pattern extraction (default: 15).
        dropout: Dropout rate in conv blocks (default: 0.1).
    """
    
    def __init__(
        self,
        vocab_size: int = 12,
        d_model: int = 256,
        stem_channels=None,
        num_downsample_stages: int = 7,
        kernel_size: int = 5,
        initial_kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_downsample_stages = num_downsample_stages
        self.total_downsample_factor = 2 ** num_downsample_stages
        
        # Build channel schedule
        if stem_channels is None:
            channel_list = [d_model] * num_downsample_stages
        elif isinstance(stem_channels, int):
            channel_list = [stem_channels] * num_downsample_stages
        elif isinstance(stem_channels, (list, tuple)):
            channel_list = list(stem_channels)
        else:
            channel_list = [d_model] * num_downsample_stages
        
        assert len(channel_list) == num_downsample_stages, (
            f"stem_channels length ({len(channel_list)}) must equal "
            f"num_downsample_stages ({num_downsample_stages})"
        )
        
        initial_ch = channel_list[0]
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, initial_ch)
        
        # Initial wide-kernel conv for capturing local DNA motifs
        initial_padding = initial_kernel_size // 2
        self.initial_conv = nn.Sequential(
            nn.Conv1d(initial_ch, initial_ch, initial_kernel_size, padding=initial_padding),
            nn.BatchNorm1d(initial_ch),
            nn.GELU(),
        )
        
        # Progressive downsampling blocks
        self.blocks = nn.ModuleList()
        in_ch = initial_ch
        for i in range(num_downsample_stages):
            out_ch = channel_list[i]
            self.blocks.append(
                ResidualDownsampleBlock(in_ch, out_ch, kernel_size, dropout)
            )
            in_ch = out_ch
        
        # Final normalization and projection to d_model
        final_ch = channel_list[-1]
        self.final_norm = nn.LayerNorm(final_ch)
        self.final_proj = (
            nn.Linear(final_ch, d_model)
            if final_ch != d_model
            else nn.Identity()
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) integer token IDs.
            
        Returns:
            (B, L // 2^N, d_model) downsampled embeddings.
        """
        # Embed tokens
        x = self.embedding(input_ids)   # (B, L, C)
        x = x.transpose(1, 2)          # (B, C, L)
        
        # Initial feature extraction
        x = self.initial_conv(x)        # (B, C, L)
        
        # Progressive downsampling
        for block in self.blocks:
            x = block(x)                # (B, C, L // 2) at each stage
        
        # Project to d_model
        x = x.transpose(1, 2)          # (B, L_down, C)
        x = self.final_norm(x)         # (B, L_down, C)
        x = self.final_proj(x)         # (B, L_down, d_model)
        
        return x

    def get_output_length(self, input_length: int) -> int:
        """Compute the output sequence length for a given input length."""
        return input_length // self.total_downsample_factor


class SimpleProjectionHead(nn.Module):
    """Simple projection head for genomic track prediction.
    
    Crops the sequence to target length, then applies a two-layer MLP
    to project from d_model to num_tracks. No spatial pooling is needed
    since downsampling is handled by the CNN stem.
    
    Architecture:
        (B, L, D) → CenterCrop(target_len) → Linear → Dropout → GELU → Linear → Softplus
        → (B, target_len, num_tracks)
    
    Args:
        d_model: Input feature dimension from encoder.
        num_tracks: Number of output genomic tracks (e.g., 5313 for human).
        hidden_dim: MLP hidden dimension (default: 2 * d_model).
        target_len: Target output sequence length after cropping (default: 896).
        dropout: Dropout rate in MLP (default: 0.4).
    """
    
    def __init__(
        self,
        d_model: int,
        num_tracks: int,
        hidden_dim: int = None,
        target_len: int = 896,
        dropout: float = 0.4,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = d_model * 2
        
        self.crop = TargetLengthCrop1D(target_len)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tracks),
            nn.Softplus(),  # Ensure positive outputs for Poisson NLL loss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) encoder output.
            
        Returns:
            (B, target_len, num_tracks) track predictions.
        """
        x = self.crop(x)   # (B, target_len, D)
        x = self.head(x)   # (B, target_len, num_tracks)
        return x
