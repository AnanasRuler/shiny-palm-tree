import torch
import torch.nn as nn

class SimpleConvModel(nn.Module):
    """
    A simple CNN model for testing the data pipeline and training loop.
    Mimics the interface of a Transformer/Mamba backbone but is lightweight.
    """
    def __init__(self, d_model=256, vocab_size=4, **kwargs):
        super().__init__()
        self.d_model = d_model
        # Simple embedding: (B, L) -> (B, L, D)
        self.embedding = nn.Embedding(vocab_size + 1, d_model) # +1 for padding/mask
        
        # 3 layers of Conv1d to process sequence
        self.conv_tower = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=15, padding=7),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=15, padding=7),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=15, padding=7),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

    def forward(self, input_ids, **kwargs):
        # input_ids: (B, L)
        x = self.embedding(input_ids) # (B, L, D)
        
        # Permute for Conv1d: (B, D, L)
        x = x.transpose(1, 2)
        
        x = self.conv_tower(x)
        
        # Permute back: (B, L, D) for the Head to consume
        x = x.transpose(1, 2)
        
        return x


# Alias for backward compatibility
SimpleCNN = SimpleConvModel


class SimpleConvConfig:
    def __init__(self, d_model=256, vocab_size=4):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self._name_ = "simple_cnn" # Registry key
