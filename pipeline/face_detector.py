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
from facenet_pytorch import MTCNN
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
        # Global preferred device (may be MPS or CUDA for other models).
        self.device = device_from_config(device_cfg)

        # facenet-pytorch MTCNN uses adaptive pooling operations that are not
        # fully supported on MPS; to avoid runtime errors on Apple Silicon,
        # we force the detector itself onto CPU when the global device is MPS.
        mtcnn_device = self.device
        if mtcnn_device.type == "mps":
            warnings.warn(
                "MTCNN face detector is not fully compatible with MPS; "
                "falling back to CPU for face detection to avoid adaptive "
                "pooling errors.",
                RuntimeWarning,
            )
            mtcnn_device = torch.device("cpu")

        self.mtcnn = MTCNN(
            image_size=self.cfg.image_size,
            margin=self.cfg.margin,
            min_face_size=self.cfg.min_face_size,
            thresholds=self.cfg.thresholds,
            factor=self.cfg.factor,
            keep_all=self.cfg.keep_all,
            post_process=self.cfg.postprocess,
            device=mtcnn_device,
        )
        self.detector_device = mtcnn_device

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
        """Run MTCNN on a single image and save face crops.

        - Saves raw BGR crops (for visualization/debugging)
        - Optionally saves MTCNN-aligned RGB crops suitable for embedding
        """
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            return []

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Get aligned faces + detection scores
        aligned_faces, probs = self.mtcnn(pil_img, return_prob=True)
        if aligned_faces is None or probs is None:
            return []

        # Also fetch boxes/landmarks for metadata
        boxes, _, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
        if boxes is None:
            return []

        detections: List[Dict[str, Any]] = []
        base_name = img_path.stem

        # Directory for aligned faces (if enabled)
        aligned_dir = faces_dir.parent / "faces_aligned"
        if self.cfg.save_aligned:
            ensure_dir(aligned_dir)

        for idx, (box, prob) in enumerate(zip(boxes, probs)):
            if box is None or prob is None:
                continue
            if float(prob) < self.cfg.min_score:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            # Clamp to image bounds
            h, w = bgr.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = bgr[y1:y2, x1:x2]
            face_filename = f"{base_name}_face_{idx:03d}.jpg"
            face_path = faces_dir / face_filename
            ok = cv2.imwrite(str(face_path), crop)
            if not ok:
                continue

            aligned_path_str: Optional[str] = None
            if self.cfg.save_aligned and aligned_faces is not None and idx < len(aligned_faces):
                aligned_tensor = aligned_faces[idx].detach().cpu()
                # Convert CHW float tensor in [0, 1] (after fixed_image_standardization inverse-ish)
                aligned_img = aligned_tensor.permute(1, 2, 0).numpy()
                aligned_img = (np.clip(aligned_img, 0.0, 1.0) * 255.0).astype("uint8")
                aligned_bgr = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)
                aligned_filename = f"{base_name}_aligned_{idx:03d}.jpg"
                aligned_path = aligned_dir / aligned_filename
                ok_aligned = cv2.imwrite(str(aligned_path), aligned_bgr)
                if ok_aligned:
                    aligned_path_str = str(aligned_path)

            det: Dict[str, Any] = {
                "shot_id": shot_id,
                "source_image": str(img_path),
                "face_image": str(face_path),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(prob),
            }
            if aligned_path_str is not None:
                det["aligned_face_image"] = aligned_path_str
            if landmarks is not None:
                lm = landmarks[idx]
                if lm is not None:
                    det["landmarks"] = lm.tolist()
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
                "device_detector": str(self.detector_device),
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

