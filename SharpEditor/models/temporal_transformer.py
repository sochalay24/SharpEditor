"""Temporal Transformer shot encoder (trainable).

Fuses per-frame actor embeddings (512-D) and scene embeddings (768-D) into a
single 1024-D shot-level embedding using a small Transformer encoder with
learned positional encoding and a [CLS] token for aggregation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """Transformer-based temporal shot encoder.

    Input:  per-frame feature vectors of size ``input_dim`` (actor + scene = 1280-D).
    Output: a single ``output_dim``-D (default 1024) shot embedding.

    Architecture:
        1. Linear projection from input_dim → d_model
        2. Prepend a learnable [CLS] token
        3. Add learned positional embeddings
        4. N Transformer encoder layers
        5. Take [CLS] output → linear projection to output_dim
    """

    def __init__(
        self,
        input_dim: int = 1280,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_frames: int = 64,
        output_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.max_frames = max_frames

        # Project input features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learned positional embeddings (CLS + up to max_frames positions)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_frames + 1, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Layer norm before projection
        self.norm = nn.LayerNorm(d_model)

        # Final projection to output_dim
        self.output_proj = nn.Linear(d_model, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        frame_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of shot frame sequences into shot embeddings.

        Args:
            frame_features: [B, T, input_dim] per-frame feature vectors.
            padding_mask: [B, T] bool tensor where True = padded (ignored).

        Returns:
            [B, output_dim] shot embeddings.
        """
        B, T, _ = frame_features.shape

        # Project to d_model
        x = self.input_proj(frame_features)  # [B, T, d_model]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)  # [B, T+1, d_model]

        # Add positional embeddings (truncate if T+1 < max)
        x = x + self.pos_embed[:, : T + 1, :]

        # Build attention mask for transformer: True = ignored position
        if padding_mask is not None:
            # Add False for CLS token (never masked)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            src_key_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            src_key_padding_mask = None

        # Encode
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Take CLS output
        cls_out = self.norm(x[:, 0, :])  # [B, d_model]

        # Project to output dim
        shot_emb = self.output_proj(cls_out)  # [B, output_dim]

        return shot_emb
