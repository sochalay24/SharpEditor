"""Face detection stage (frames → face boxes/crops).

Implementation: facenet-pytorch MTCNN with Apple Silicon (MPS) support.

This module operates on the per-shot frame folders produced by the frame
extraction stage and writes face crops + detection metadata manifests. It does
not perform any actor recognition; it purely localizes faces and prepares
normalized crops for downstream embedding models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import cv2
import numpy as np
import torch
from retinaface import RetinaFace
from PIL import Image
from tqdm import tqdm

from utils.device_utils import device_from_config
from utils.video_utils import ensure_dir, write_json


@dataclass(frozen=True)
class FaceDetectorConfig:
    """Configuration wrapper for MTCNN-based face detection."""

    frame_source: str = "keyframes"  # 'keyframes' | 'sampled' | 'both'
    image_size: int = 160
    margin: int = 20
    min_face_size: int = 20
    thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.7)
    factor: float = 0.709
    keep_all: bool = True
    postprocess: bool = True
    min_score: float = 0.8
    save_aligned: bool = True


class FaceDetector:
    """Run MTCNN over shot frames and save face crops + metadata.

    Expected shot layout (from frame extractor):

        <frames_root>/<shot_id>/
            manifest.json
            sampled/
              frame_0000.jpg
              ...
            keyframes/
              keyframe_000.jpg
              ...

    This detector will create:

        <frames_root>/<shot_id>/
            faces/
              frame_0000_face_000.jpg
              ...
            faces.json
    """

    def __init__(self, cfg: FaceDetectorConfig, *, device_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg
        self.device = device_from_config(device_cfg)
        # RetinaFace handles its own model loading and execution.
        # We don't need to forcefully place it on a specific device here 
        # as it automatically utilizes available hardware via tf/keras.

    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> "FaceDetector":
        """Instantiate from the global YAML config dict."""
        fd = d.get("face_detection", {}) or {}
        device_cfg = d.get("device", {}) or {}
        thresholds = fd.get("thresholds", [0.6, 0.7, 0.7])
        if len(thresholds) != 3:
            raise ValueError("face_detection.thresholds must be a list of length 3.")

        cfg = FaceDetectorConfig(
            frame_source=str(fd.get("frame_source", "keyframes")).lower(),
            image_size=int(fd.get("image_size", 160)),
            margin=int(fd.get("margin", 20)),
            min_face_size=int(fd.get("min_face_size", 20)),
            thresholds=(float(thresholds[0]), float(thresholds[1]), float(thresholds[2])),
            factor=float(fd.get("factor", 0.709)),
            keep_all=bool(fd.get("keep_all", True)),
            postprocess=bool(fd.get("postprocess", True)),
            min_score=float(fd.get("min_score", 0.8)),
            save_aligned=bool(fd.get("save_aligned", True)),
        )
        return FaceDetector(cfg, device_cfg=device_cfg)

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------

    def _select_frame_paths(self, shot_dir: Path) -> List[Path]:
        """Select which frame images to use for detection based on config."""
        sampled_dir = shot_dir / "sampled"
        keyframes_dir = shot_dir / "keyframes"

        def list_images(d: Path) -> List[Path]:
            if not d.exists() or not d.is_dir():
                return []
            exts = {".jpg", ".jpeg", ".png"}
            return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts])

        fs = self.cfg.frame_source
        if fs == "keyframes":
            return list_images(keyframes_dir)
        if fs == "sampled":
            return list_images(sampled_dir)
        if fs == "both":
            return list_images(keyframes_dir) + list_images(sampled_dir)
        raise ValueError(f"Unsupported frame_source: {fs}")

    def _detect_faces_in_image(self, img_path: Path, *, faces_dir: Path, shot_id: str) -> List[Dict[str, Any]]:
        """Run RetinaFace on a single image and save face crops.

        - Saves raw BGR crops (for visualization/debugging)
        - Optionally saves RetinaFace-aligned RGB crops resized to target dim
        """
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            return []

        try:
            resp = RetinaFace.detect_faces(str(img_path))
        except Exception:
            return []

        base_name = img_path.stem
        aligned_dir = faces_dir.parent / "faces_aligned"
        if self.cfg.save_aligned:
            ensure_dir(aligned_dir)

        # RetinaFace returns empty tuple if no faces found
        if isinstance(resp, tuple) and len(resp) == 0:
            return []
            
        if not isinstance(resp, dict):
            return []

        try:
            aligned_crops = RetinaFace.extract_faces(str(img_path), align=True)
        except Exception:
            aligned_crops = []

        use_aligned = len(aligned_crops) == len(resp)
        detections: List[Dict[str, Any]] = []

        for idx, (face_key, face_info) in enumerate(resp.items()):
            score = face_info.get("score", 0)
            if score < self.cfg.min_score:
                continue

            box = face_info["facial_area"]
            x1, y1, x2, y2 = box
            
            # Apply margin similar to MTCNN
            margin = self.cfg.margin
            h, w = bgr.shape[:2]
            
            x1_crop = max(0, int(x1) - margin // 2)
            y1_crop = max(0, int(y1) - margin // 2)
            x2_crop = min(w, int(x2) + margin // 2)
            y2_crop = min(h, int(y2) + margin // 2)

            if x2_crop <= x1_crop or y2_crop <= y1_crop:
                continue

            raw_crop = bgr[y1_crop:y2_crop, x1_crop:x2_crop]
            face_filename = f"{base_name}_face_{idx:03d}.jpg"
            face_path = faces_dir / face_filename
            cv2.imwrite(str(face_path), raw_crop)

            det: Dict[str, Any] = {
                "shot_id": shot_id,
                "source_image": str(img_path),
                "face_image": str(face_path),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
            }

            if use_aligned and self.cfg.save_aligned:
                aligned_crop_rgb = aligned_crops[idx]
                if aligned_crop_rgb is not None and aligned_crop_rgb.size > 0:
                    aligned_crop_bgr = cv2.cvtColor(aligned_crop_rgb, cv2.COLOR_RGB2BGR)
                    aligned_crop_resized = cv2.resize(aligned_crop_bgr, (self.cfg.image_size, self.cfg.image_size))
                    
                    aligned_filename = f"{base_name}_aligned_{idx:03d}.jpg"
                    aligned_path = aligned_dir / aligned_filename
                    cv2.imwrite(str(aligned_path), aligned_crop_resized)
                    det["aligned_face_image"] = str(aligned_path)

            detections.append(det)

        return detections

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def process_shot(
        self,
        shot_dir: str,
        *,
        overwrite: bool = False,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """Run face detection for a single shot directory."""
        shot_dir_path = Path(shot_dir)
        shot_id = shot_dir_path.name
        faces_dir = shot_dir_path / "faces"
        ensure_dir(faces_dir)

        manifest_path = shot_dir_path / "faces.json"
        if manifest_path.exists() and not overwrite:
            return {
                "shot_id": shot_id,
                "shot_dir": str(shot_dir_path),
                "skipped": True,
                "reason": "faces_manifest_exists",
            }

        frame_paths = self._select_frame_paths(shot_dir_path)
        all_detections: List[Dict[str, Any]] = []

        it = frame_paths
        if not quiet:
            it = tqdm(it, desc=f"Detecting faces: {shot_id}", leave=False)

        for img_path in it:
            dets = self._detect_faces_in_image(img_path, faces_dir=faces_dir, shot_id=shot_id)
            all_detections.extend(dets)

        by_image: Dict[str, List[Dict[str, Any]]] = {}
        for det in all_detections:
            by_image.setdefault(det["source_image"], []).append(det)

        manifest: Dict[str, Any] = {
            "shot_id": shot_id,
            "shot_dir": str(shot_dir_path),
            "faces_dir": str(faces_dir),
            "config": {
                "frame_source": self.cfg.frame_source,
                "image_size": self.cfg.image_size,
                "margin": self.cfg.margin,
                "min_face_size": self.cfg.min_face_size,
                "thresholds": list(self.cfg.thresholds),
                "factor": self.cfg.factor,
                "keep_all": self.cfg.keep_all,
                "postprocess": self.cfg.postprocess,
                "device_global": str(self.device),
            },
            "num_frames_processed": len(frame_paths),
            "num_faces": len(all_detections),
            "detections_by_image": by_image,
        }
        write_json(manifest_path, manifest)
        return manifest

    def process_dataset(
        self,
        frames_root: str,
        *,
        overwrite: bool = False,
        quiet: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run face detection for all shots under a frames root directory."""
        frames_root_path = Path(frames_root)
        if not frames_root_path.exists():
            raise FileNotFoundError(f"Frames root not found: {frames_root}")

        shot_dirs = sorted([p for p in frames_root_path.iterdir() if p.is_dir()])
        results: List[Dict[str, Any]] = []

        it = shot_dirs
        if not quiet:
            it = tqdm(it, desc="Face detection on dataset")

        for sd in it:
            results.append(self.process_shot(str(sd), overwrite=overwrite, quiet=quiet))
        return results

