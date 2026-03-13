"""Embedding utilities (saving/loading, normalization, similarity search)."""

from __future__ import annotations

from typing import Optional

import torch


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize a tensor along a given dimension."""
    norm = torch.linalg.norm(x, ord=2, dim=dim, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return x / norm


