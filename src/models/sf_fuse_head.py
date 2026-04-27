import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetLengthCrop1D(nn.Module):
    """
    Crops the center of the sequence to the target length, assuming 1D input (B, C, L) or (B, L, C).
    """
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        # Check if shape is (B, L, C) or (B, C, L)
        # Assuming (B, L, C) from LLM usually
        seq_len = x.shape[1] 
        if seq_len < self.target_length:
            raise ValueError(f"Sequence length {seq_len} is smaller than target length {self.target_length}")
        
        start = (seq_len - self.target_length) // 2
        end = start + self.target_length
        return x[:, start:end, :]

class SFFuseHead(nn.Module):
    """
    SF-Fuse-style head for predicting genomic tracks.
    Adapts from a sequence embedding to track predictions.
    
    Structure:
    1. Pointwise Conv (Project raw embedding dim to hidden dim)
    2. Pooling (Reduce resolution, e.g., from 1bp to 128bp bins)
    3. Crop (Remove edges to match target length)
    4. Two-layer MLP (Hidden -> 2*Hidden -> Num_Tracks)
    """
    def __init__(
        self,
        d_model: int,
        num_tracks: int,
        head_hidden_dim: int = 96, # SF-Fuse uses ~96*8 or similar
        pooling_factor: int = 128, # SF-Fuse bins are 128bp
        target_len: int = 896,     # SF-Fuse target bins
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_tracks = num_tracks
        self.pooling_factor = pooling_factor
        
        # 1. Pointwise Conv / Projection
        self.projection = nn.Linear(d_model, head_hidden_dim)
        
        # 2. Pooling
        # If input is (B, L, C), we want to pool over L.
        # Average pooling is standard for downsampling in this context
        self.pooling = nn.AvgPool1d(kernel_size=pooling_factor, stride=pooling_factor)
        
        # 3. Crop
        self.crop = TargetLengthCrop1D(target_length=target_len)
        
        # 4. Final MLP
        self.head_mlp = nn.Sequential(
            nn.Linear(head_hidden_dim, head_hidden_dim * 2),
            nn.Dropout(dropout_rate),
            nn.GELU(),
            nn.Linear(head_hidden_dim * 2, num_tracks),
            nn.Softplus() # Ensure positive predictions for Poisson Loss
        )

    def forward(self, x):
        # x: (Batch, Length, D_model)
        
        # Project
        x = self.projection(x) # (B, L, H)
        
        # Rearrange for Pooling (B, C, L) required for Conv/Pool layers usually
        x = x.transpose(1, 2) # (B, H, L)
        
        # Pool
        x = self.pooling(x) # (B, H, L/pool)
        
        # Back to (B, L_new, H)
        x = x.transpose(1, 2)
        
        # Crop
        x = self.crop(x)
        
        # Final MLP
        output = self.head_mlp(x)
        
        return output
