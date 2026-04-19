from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from utils.video_utils import (
    VideoMetadata,
    compute_sample_indices,
    ensure_dir,
    open_video,
    read_video_metadata,
    resize_frame,
    write_json,
)


@dataclass(frozen=True)
class FrameExtractorConfig:
    """Configuration for extracting representative frames from a shot."""

    sampling_fps: float = 2.0
    max_frames_per_shot: int = 30
    resize_enabled: bool = True
    resize_width: int = 640
    resize_height: int = 360
    image_format: str = "jpg"
    jpg_quality: int = 92
    keyframes_enabled: bool = True
    max_keyframes: int = 12
    min_scene_change_score: float = 0.25


def _histogram_scene_change_score(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> float:
    """Compute a lightweight scene-change score based on HSV histogram divergence."""
    prev_hsv = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2HSV)
    curr_hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)

    hist_bins = (32, 32)
    prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, hist_bins, [0, 180, 0, 256])
    curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, hist_bins, [0, 180, 0, 256])
    cv2.normalize(prev_hist, prev_hist)
    cv2.normalize(curr_hist, curr_hist)
    # Bhattacharyya distance in [0, 1] (roughly), higher = more different
    score = float(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA))
    return score


def _encode_image(path: str, frame_bgr: np.ndarray, *, image_format: str, jpg_quality: int) -> None:
    """Write a frame to disk with deterministic encoding params."""
    ext = image_format.lower().lstrip(".")
    if ext not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported image_format: {image_format}")

    if ext in {"jpg", "jpeg"}:
        ok = cv2.imwrite(path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])
    else:
        ok = cv2.imwrite(path, frame_bgr)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


class FrameExtractor:
    """Extract representative frames from a shot (video file).

    Outputs:
    - Sampled frames (uniform sampling at configurable FPS)
    - Optional keyframes (subset of sampled frames based on scene-change scoring)
    - `manifest.json` containing metadata and file lists
    """

    def __init__(self, cfg: FrameExtractorConfig):
        self.cfg = cfg

    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> "FrameExtractor":
        """Build from a top-level YAML config dict."""
        resize = d.get("frame_resize", {}) or {}
        keyframes = d.get("keyframes", {}) or {}
        return FrameExtractor(
            FrameExtractorConfig(
                sampling_fps=float(d.get("frame_sampling_rate_fps", 2)),
                max_frames_per_shot=int(d.get("max_frames_per_shot", 30)),
                resize_enabled=bool(resize.get("enabled", True)),
                resize_width=int(resize.get("width", 640)),
                resize_height=int(resize.get("height", 360)),
                image_format=str(d.get("image_format", "jpg")),
                jpg_quality=int(d.get("jpg_quality", 92)),
                keyframes_enabled=bool(keyframes.get("enabled", True)),
                max_keyframes=int(keyframes.get("max_keyframes", 12)),
                min_scene_change_score=float(keyframes.get("min_scene_change_score", 0.25)),
            )
        )

    def extract_shot(
        self,
        video_path: str,
        output_dir: str,
        *,
        shot_id: Optional[str] = None,
        overwrite: bool = False,
        write_keyframes: bool = True,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """Extract frames for a single shot."""
        video_path = os.path.abspath(video_path)
        meta = read_video_metadata(video_path)

        shot_id = shot_id or Path(video_path).stem
        shot_dir = Path(output_dir) / shot_id
        sampled_dir = shot_dir / "sampled"
        keyframes_dir = shot_dir / "keyframes"

        ensure_dir(sampled_dir)
        ensure_dir(keyframes_dir)

        manifest_path = shot_dir / "manifest.json"
        if manifest_path.exists() and not overwrite:
            # Fast path for iterating during research.
            return {
                "shot_id": shot_id,
                "video_path": video_path,
                "output_dir": str(shot_dir),
                "skipped": True,
                "reason": "manifest_exists",
            }

        sample_indices = compute_sample_indices(
            src_fps=meta.fps,
            frame_count=meta.frame_count,
            target_fps=self.cfg.sampling_fps,
            max_frames=self.cfg.max_frames_per_shot,
        )

        cap = open_video(video_path)
        sampled_files: List[str] = []
        sampled_frame_indices: List[int] = []
        sampled_scene_change_scores: List[float] = []

        try:
            prev_frame = None
            it = sample_indices
            if not quiet:
                it = tqdm(it, desc=f"Sampling frames: {shot_id}", leave=False)

            for out_i, frame_idx in enumerate(it):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                if self.cfg.resize_enabled:
                    frame = resize_frame(frame, width=self.cfg.resize_width, height=self.cfg.resize_height)

                score = 0.0
                if prev_frame is not None:
                    score = _histogram_scene_change_score(prev_frame, frame)
                prev_frame = frame

                filename = f"frame_{out_i:04d}.{self.cfg.image_format.lower().lstrip('.')}"
                out_path = sampled_dir / filename
                _encode_image(
                    str(out_path),
                    frame,
                    image_format=self.cfg.image_format,
                    jpg_quality=self.cfg.jpg_quality,
                )
                sampled_files.append(str(out_path))
                sampled_frame_indices.append(int(frame_idx))
                sampled_scene_change_scores.append(float(score))
        finally:
            cap.release()

        keyframe_files: List[str] = []
        keyframe_source_indices: List[int] = []
        if self.cfg.keyframes_enabled and write_keyframes and sampled_files:
            # Always include first frame; then pick top scene-change peaks.
            scored = list(enumerate(sampled_scene_change_scores))
            # Exclude first frame score=0.0 from ranking; we add it explicitly.
            scored = [(i, s) for i, s in scored if i != 0 and s >= self.cfg.min_scene_change_score]
            scored.sort(key=lambda x: x[1], reverse=True)

            selected = [0] + [i for i, _ in scored[: max(0, self.cfg.max_keyframes - 1)]]
            selected = sorted(set(selected))

            for k_i, sample_i in enumerate(selected):
                src_path = Path(sampled_files[sample_i])
                dst_path = keyframes_dir / f"keyframe_{k_i:03d}.{src_path.suffix.lstrip('.')}"
                # Re-encode to keep artifacts consistent even if sampled frames are deleted later.
                frame = cv2.imread(str(src_path))
                if frame is None:
                    continue
                _encode_image(
                    str(dst_path),
                    frame,
                    image_format=self.cfg.image_format,
                    jpg_quality=self.cfg.jpg_quality,
                )
                keyframe_files.append(str(dst_path))
                keyframe_source_indices.append(int(sample_i))

        manifest = {
            "shot_id": shot_id,
            "video_path": video_path,
            "output_dir": str(shot_dir),
            "video_metadata": {
                "fps": meta.fps,
                "frame_count": meta.frame_count,
                "duration_s": meta.duration_s,
                "width": meta.width,
                "height": meta.height,
            },
            "sampling": {
                "target_fps": self.cfg.sampling_fps,
                "max_frames_per_shot": self.cfg.max_frames_per_shot,
                "resize": {
                    "enabled": self.cfg.resize_enabled,
                    "width": self.cfg.resize_width,
                    "height": self.cfg.resize_height,
                },
                "image_format": self.cfg.image_format,
                "jpg_quality": self.cfg.jpg_quality,
            },
            "sampled": {
                "files": sampled_files,
                "source_frame_indices": sampled_frame_indices,
                "scene_change_scores": sampled_scene_change_scores,
            },
            "keyframes": {
                "enabled": bool(self.cfg.keyframes_enabled and write_keyframes),
                "min_scene_change_score": self.cfg.min_scene_change_score,
                "max_keyframes": self.cfg.max_keyframes,
                "files": keyframe_files,
                "source_sample_indices": keyframe_source_indices,
            },
        }
        write_json(manifest_path, manifest)
        return manifest

    def extract_dataset(
        self,
        raw_shots_dir: str,
        frames_out_dir: str,
        *,
        extensions: Tuple[str, ...] = (".mov", ".mp4", ".mkv", ".avi", ".mts", ".m2ts", ".m4v"),
        overwrite: bool = False,
        quiet: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract frames for all shots in a directory."""
        raw_shots_dir = os.path.abspath(raw_shots_dir)
        frames_out_dir = os.path.abspath(frames_out_dir)
        ensure_dir(frames_out_dir)

        videos: List[str] = []
        for name in sorted(os.listdir(raw_shots_dir)):
            p = os.path.join(raw_shots_dir, name)
            if not os.path.isfile(p):
                continue
            if Path(p).suffix.lower() in {e.lower() for e in extensions}:
                videos.append(p)

        results: List[Dict[str, Any]] = []
        it = videos if quiet else tqdm(videos, desc="Extracting shots")
        for vp in it:
            results.append(
                self.extract_shot(
                    vp,
                    frames_out_dir,
                    overwrite=overwrite,
                    quiet=quiet,
                )
            )
        return results

