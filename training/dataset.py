"""Training dataset for temporal shot encoder."""

from __future__ import annotations


class ShotPairDataset:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Dataset will be implemented alongside the training objective.")

