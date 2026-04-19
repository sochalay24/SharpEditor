from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata for a video file."""

    path: str
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


def ensure_dir(path: str | Path) -> str:
    """Create a directory if it doesn't exist. Returns the string path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def open_video(path: str) -> cv2.VideoCapture:
    """Open a video via OpenCV with basic validation."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def read_video_metadata(path: str) -> VideoMetadata:
    """Read fps, frame count, and dimensions from a video file."""
    cap = open_video(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    if fps <= 0:
        # Some codecs/container combos on macOS report 0; fall back to a sane default.
        fps = 30.0

    duration_s = (frame_count / fps) if frame_count > 0 else 0.0
    return VideoMetadata(
        path=os.path.abspath(path),
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
        width=width,
        height=height,
    )


def compute_sample_indices(
    src_fps: float,
    frame_count: int,
    target_fps: float,
    max_frames: Optional[int] = None,
) -> list[int]:
    """Compute frame indices to sample at a target FPS from a source FPS stream."""
    if frame_count <= 0:
        return []
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    stride = max(int(round(src_fps / target_fps)), 1)
    indices = list(range(0, frame_count, stride))
    if max_frames is not None and max_frames > 0:
        indices = indices[:max_frames]
    return indices


def resize_frame(
    frame_bgr, *, width: int, height: int
):
    """Resize a BGR frame to fixed dimensions."""
    return cv2.resize(frame_bgr, (int(width), int(height)), interpolation=cv2.INTER_AREA)

