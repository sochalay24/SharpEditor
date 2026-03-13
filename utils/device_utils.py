from __future__ import annotations

from typing import Literal

import torch


DeviceType = Literal["cpu", "cuda", "mps"]


def get_best_device(prefer_mps: bool = True) -> torch.device:
    """Select the best available torch.device for this machine.

    Preference order:
    - MPS (Apple Silicon) if prefer_mps and available
    - CUDA if available
    - CPU otherwise
    """
    if prefer_mps and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_from_config(device_cfg: dict | None) -> torch.device:
    """Helper to build a device from a nested config section."""
    device_cfg = device_cfg or {}
    prefer_mps = bool(device_cfg.get("prefer_mps", True))
    return get_best_device(prefer_mps=prefer_mps)

