"""
Gated Attention Decoder for DualRep Architecture.

This module implements an 11-layer gated attention decoder as specified in
the thesis Chapter 4 (DualRep architecture).

The gated attention mechanism uses query-dependent scalar gates to modulate
attention outputs, enabling sparse and efficient feature selection.

Architecture Overview (Section 4.2.4):
    - 11 layers of gated multi-head self-attention
    - Each layer: GatedAttention -> Dropout -> Add -> LayerNorm -> FFN -> Dropout -> Add -> LayerNorm
    - Gate computation: g = Q @ W_g, where W_g projects query to scalar per head
    - Gate activation: sigmoid(g) element-wise multiplies attention output

Headwise Gating (Section 2.2.3):
    - Each attention head shares a single scalar gate value
    - Gate scores computed from query vectors via linear projection
    - Minimal parameter overhead (d_h parameters per head)
    - Formula: GatedAttn(Q, K, V) = σ(g) ⊙ Attn(Q, K, V)

References:
    - Thesis Chapter 4: DualRep Architecture Specification
    - Formula 4-1: Gated Attention Mechanism
    - Section 2.2.3: Headwise Gating Strategy
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GatedMultiHeadAttention(nn.Module):
    """Multi-head self-attention with query-dependent headwise gating.

    This module implements standard scaled dot-product attention with an additional
    gating mechanism that modulates attention outputs based on query content.

    Gating Mechanism (Formula 4-1):
        gate_score = Q @ W_g  # Shape: (B, L, num_heads)
        gate = sigmoid(gate_score)  # Shape: (B, L, num_heads, 1)
        gated_output = gate * attention_output

    The gate acts as a content-dependent filter, allowing the model to selectively
    attend to different positions based on the query representation.

    Args:
        d_model: Model dimension (default: 768).
        num_heads: Number of attention heads (default: 16).
        dropout: Dropout probability (default: 0.1).
        bias: Whether to use bias in linear projections (default: True).
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 16,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Gate projection: query -> scalar per head (headwise gating)
        # W_g in Formula 4-1: projects query to gate scores
        self.gate_proj = nn.Linear(d_model, num_heads, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for gated multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask: Optional attention mask of shape
                (batch_size, 1, seq_len, seq_len) or (batch_size, seq_len).
                Causal mask should be upper triangular with -inf for future positions.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of:
                - output: Gated attention output of shape (batch_size, seq_len, d_model).
                - attention_weights: Optional attention weights (batch_size, num_heads, seq_len, seq_len).
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V projections
        # Shape: (B, L, D) -> (B, L, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Compute gate scores from queries
        # Shape: (B, L, D) -> (B, L, num_heads)
        gate_scores = self.gate_proj(q)

        # Reshape for multi-head attention: (B, L, D) -> (B, num_heads, L, head_dim)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_dim)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_dim)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads, d=self.head_dim)

        # Scaled dot-product attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) @ V
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Support both (B, L) and (B, 1, L, L) mask formats
            if attention_mask.dim() == 2:
                # Convert (B, L) to (B, 1, 1, L) for broadcasting
                attention_mask = attention_mask[:, None, None, :]
            attention_scores = attention_scores + attention_mask

        # Softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Compute attention output: (B, num_heads, L, L) @ (B, num_heads, L, head_dim)
        # -> (B, num_heads, L, head_dim)
        attention_output = torch.matmul(attention_weights, v)

        # Reshape back: (B, num_heads, L, head_dim) -> (B, L, D)
        attention_output = rearrange(
            attention_output, "b h l d -> b l (h d)", h=self.num_heads, d=self.head_dim
        )

        # Apply gating: GatedAttn(Q, K, V) = σ(g) ⊙ Attn(Q, K, V)
        # gate_scores shape: (B, L, num_heads)
        # gate shape after sigmoid and unsqueeze: (B, L, num_heads, 1)
        gates = torch.sigmoid(gate_scores).unsqueeze(-1)

        # Reshape attention_output for gating: (B, L, num_heads, head_dim)
        attention_output_reshaped = rearrange(
            attention_output, "b l (h d) -> b l h d", h=self.num_heads, d=self.head_dim
        )

        # Apply gate: element-wise multiplication
        # Each head gets multiplied by its corresponding scalar gate
        gated_output = gates * attention_output_reshaped

        # Reshape back to (B, L, D)
        gated_output = rearrange(gated_output, "b l h d -> b l (h d)")

        # Output projection
        output = self.out_proj(gated_output)

        if output_attentions:
            return output, attention_weights
        return output, None


class GatedAttentionLayer(nn.Module):
    """Single layer of gated attention with feed-forward network.

    Implements pre-normalization Transformer encoder layer with gated attention:
        x = x + GatedAttention(LN(x))
        x = x + FFN(LN(x))

    The gated attention mechanism enables query-dependent sparse attention,
    where each attention head's output is modulated by a learned gate.

    Args:
        d_model: Model dimension (default: 768).
        num_heads: Number of attention heads (default: 16).
        dim_ff: Feed-forward hidden dimension (default: 3072, 4x d_model).
        dropout: Dropout probability (default: 0.1).
        activation: Activation function for FFN (default: GELU).
        norm_eps: Epsilon for layer normalization (default: 1e-5).
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 16,
        dim_ff: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Pre-attention layer norm
        self.norm1 = nn.LayerNorm(d_model, eps=norm_eps)

        # Gated multi-head attention
        self.gated_attn = GatedMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Pre-FFN layer norm
        self.norm2 = nn.LayerNorm(d_model, eps=norm_eps)

        # Feed-forward network: two-layer MLP with activation
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for gated attention layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask: Optional attention mask.
            output_attentions: Whether to return attention weights.

        Returns:
            Tuple of:
                - output: Layer output of shape (batch_size, seq_len, d_model).
                - attention_weights: Optional attention weights.
        """
        # Pre-norm
        x_norm = self.norm1(x)

        # Gated attention with residual connection
        attn_output, attn_weights = self.gated_attn(
            x_norm,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        x = x + attn_output

        # Pre-norm + FFN with residual connection
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output

        return x, attn_weights


class GatedAttentionDecoder(nn.Module):
    """11-layer gated attention decoder for DualRep architecture.

    This is the main decoder component specified in thesis Chapter 4 (DualRep).
    It consists of 11 stacked gated attention layers that process sequence
    representations at 1024 resolution (after CNN bridge downsampling).

    Architecture (Section 4.2.4):
        Input (B, L, d_model)
        → [GatedAttentionLayer x 11]
        → Output (B, L, d_model)

    Each layer contains:
        - LayerNorm (pre-norm)
        - Gated Multi-Head Self-Attention (query-dependent headwise gating)
        - Residual connection
        - LayerNorm (pre-norm)
        - Feed-Forward Network (GELU activation)
        - Residual connection

    Gating Mechanism (Formula 4-1):
        GatedAttn(Q, K, V) = σ(Q @ W_g) ⊙ Attention(Q, K, V)

    This enables the model to learn sparse, content-dependent attention patterns
    that are more interpretable and parameter-efficient than standard attention.

    Args:
        d_model: Model dimension (default: 768).
        num_heads: Number of attention heads (default: 16).
        num_layers: Number of gated attention layers (default: 11).
        dim_ff: Feed-forward hidden dimension (default: 3072).
        dropout: Dropout probability (default: 0.1).
        activation: Activation function for FFN (default: "gelu").
        norm_eps: Epsilon for layer normalization (default: 1e-5).

    Example:
        >>> decoder = GatedAttentionDecoder(d_model=768, num_heads=16, num_layers=11)
        >>> x = torch.randn(2, 1024, 768)  # (batch, seq_len, d_model)
        >>> output = decoder(x)  # (2, 1024, 768)
    """

    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 16,
        num_layers: int = 11,
        dim_ff: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Stack of 11 gated attention layers
        self.layers = nn.ModuleList([
            GatedAttentionLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_ff=dim_ff,
                dropout=dropout,
                activation=activation,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model, eps=norm_eps)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights using Xavier/Glorot initialization.

        This initialization strategy is consistent with standard Transformer
        practices and helps stabilize training for deep architectures.

        Args:
            module: The module to initialize.
        """
        if isinstance(module, nn.Linear):
            # Xavier normal initialization for linear layers
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            # Standard LayerNorm initialization
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """Forward pass through the gated attention decoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            attention_mask: Optional attention mask of shape
                (batch_size, seq_len) or (batch_size, 1, seq_len, seq_len).
                Use -inf values for masked positions (e.g., padding or future tokens).
            output_attentions: Whether to return attention weights from all layers.

        Returns:
            Tuple of:
                - output: Decoder output of shape (batch_size, seq_len, d_model).
                - all_attention_weights: Optional tuple of attention weights from each layer.

        Example:
            >>> decoder = GatedAttentionDecoder()
            >>> x = torch.randn(4, 1024, 768)
            >>> output, attn_weights = decoder(x, output_attentions=True)
            >>> print(output.shape)  # torch.Size([4, 1024, 768])
            >>> print(len(attn_weights))  # 11 (one per layer)
            >>> print(attn_weights[0].shape)  # (4, 16, 1024, 1024)
        """
        all_attention_weights = [] if output_attentions else None

        # Pass through each gated attention layer
        for i, layer in enumerate(self.layers):
            layer_output, layer_attn_weights = layer(
                x,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            x = layer_output

            if output_attentions and layer_attn_weights is not None:
                all_attention_weights.append(layer_attn_weights)

        # Final layer normalization
        x = self.final_norm(x)

        if output_attentions:
            return x, tuple(all_attention_weights)
        return x, None

    def get_num_params(self) -> int:
        """Get the total number of parameters in the decoder.

        Returns:
            Total parameter count.
        """
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get the number of trainable parameters.

        Returns:
            Number of parameters with requires_grad=True.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """Create a causal (triangular) attention mask.

    This mask prevents attention to future positions, useful for autoregressive
    decoding or causal language modeling.

    Args:
        seq_len: Sequence length.
        device: Target device for the mask.

    Returns:
        Causal mask tensor of shape (seq_len, seq_len) with 0 for valid
        positions and -inf for masked (future) positions.

    Example:
        >>> mask = create_causal_mask(4)
        >>> print(mask)
        tensor([[ 0., -inf, -inf, -inf],
                [ 0.,   0., -inf, -inf],
                [ 0.,   0.,   0., -inf],
                [ 0.,   0.,   0.,   0.]])
    """
    mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    if device is not None:
        mask = mask.to(device)
    return mask


def create_padding_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    """Convert a padding mask to attention mask format.

    Args:
        padding_mask: Boolean mask of shape (batch_size, seq_len) where
            True indicates padding positions to be masked.

    Returns:
        Attention mask of shape (batch_size, 1, 1, seq_len) with 0 for valid
        positions and -inf for masked (padding) positions.
    """
    # (B, L) -> (B, 1, 1, L)
    mask = padding_mask[:, None, None, :].float()
    mask = mask * float("-inf")
    return mask
