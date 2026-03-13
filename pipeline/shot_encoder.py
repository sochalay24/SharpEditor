"""Temporal shot encoding stage.

Consumes per-frame features (actor embeddings, scene embeddings, optional extras)
and produces a 1024-D shot embedding via a trainable Transformer encoder.
"""

from __future__ import annotations


class ShotEncoder:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Temporal shot encoder will be implemented after CLIP scene features.")

