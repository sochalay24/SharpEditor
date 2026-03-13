"""Scene context encoding stage (frames → CLIP scene embeddings).

Implementation: CLIP Vision Transformer (openai/clip-vit-base-patch32).

This stage consumes per-shot frames (typically keyframes), encodes them with
the CLIP vision encoder, and aggregates per-shot scene embeddings. Outputs are:

- Dataset-level scene embeddings under ``data/embeddings/scene_embeddings.npz``
- Per-shot ``scene.json`` files summarizing scene embeddings per shot
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

from utils.device_utils import device_from_config
from utils.video_utils import ensure_dir, write_json


@dataclass(frozen=True)
class SceneEncoderConfig:
    """Configuration for CLIP-based scene encoding."""

    model_name: str = "openai/clip-vit-base-patch32"
    frame_source: str = "keyframes"  # 'keyframes' | 'sampled' | 'both'
    batch_size: int = 16


class SceneEncoder:
    """Encode scene context from frames using CLIP Vision Transformer."""

    def __init__(self, cfg: SceneEncoderConfig, *, device_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg
        self.device = device_from_config(device_cfg)

        self.processor = CLIPImageProcessor.from_pretrained(self.cfg.model_name)
        self.model = CLIPVisionModel.from_pretrained(self.cfg.model_name).to(self.device)
        self.model.eval()

        # Hidden size (e.g. 768 for ViT-B/32) inferred from config
        self.hidden_size = self.model.config.hidden_size

    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> "SceneEncoder":
        """Instantiate from the global YAML config dict."""
        se = d.get("scene_encoding", {}) or {}
        device_cfg = d.get("device", {}) or {}
        cfg = SceneEncoderConfig(
            model_name=str(se.get("model_name", "openai/clip-vit-base-patch32")),
            frame_source=str(se.get("frame_source", "keyframes")).lower(),
            batch_size=int(se.get("batch_size", 16)),
        )
        return SceneEncoder(cfg, device_cfg=device_cfg)

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------

    def _select_frame_paths(self, shot_dir: Path) -> List[Path]:
        """Select which frame images to use for scene encoding."""
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

    def _encode_images(self, image_paths: List[Path], *, quiet: bool) -> np.ndarray:
        """Encode images into CLIP vision pooled embeddings."""
        if not image_paths:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        all_embs: List[torch.Tensor] = []

        it = image_paths if quiet else tqdm(image_paths, desc="Encoding scene frames")

        batch_images: List[Image.Image] = []
        for path in it:
            img = Image.open(path).convert("RGB")
            batch_images.append(img)
            if len(batch_images) >= self.cfg.batch_size:
                inputs = self.processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values)
                    pooled = outputs.pooler_output  # [B, hidden_size]
                all_embs.append(pooled.cpu())
                batch_images = []

        if batch_images:
            inputs = self.processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)
                pooled = outputs.pooler_output
            all_embs.append(pooled.cpu())

        if not all_embs:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        embs = torch.cat(all_embs, dim=0)
        return embs.numpy().astype(np.float32)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def process_dataset(
        self,
        frames_root: str,
        embeddings_root: str,
        *,
        overwrite: bool = False,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """Compute scene embeddings for all shots in a dataset."""
        frames_root_path = Path(frames_root)
        if not frames_root_path.exists():
            raise FileNotFoundError(f"Frames root not found: {frames_root}")

        ensure_dir(embeddings_root)
        embeddings_root_path = Path(embeddings_root)

        shot_dirs = sorted([p for p in frames_root_path.iterdir() if p.is_dir()])

        shot_ids: List[str] = []
        shot_scene_embs: List[np.ndarray] = []
        shot_num_frames: List[int] = []

        it = shot_dirs if quiet else tqdm(shot_dirs, desc="Scene encoding for dataset")
        for shot_dir in it:
            shot_id = shot_dir.name
            frame_paths = self._select_frame_paths(shot_dir)
            if not frame_paths:
                continue

            per_frame_embs = self._encode_images(frame_paths, quiet=quiet)
            if per_frame_embs.shape[0] == 0:
                continue

            # Aggregate per-shot scene embedding as mean of frame embeddings.
            scene_emb = per_frame_embs.mean(axis=0)

            shot_ids.append(shot_id)
            shot_scene_embs.append(scene_emb)
            shot_num_frames.append(int(per_frame_embs.shape[0]))

            # Write per-shot scene summary JSON.
            scene_summary = {
                "shot_id": shot_id,
                "num_frames_used": int(per_frame_embs.shape[0]),
                "frame_source": self.cfg.frame_source,
                "embedding_dim": int(scene_emb.shape[0]),
                "scene_embedding": scene_emb.tolist(),
            }
            out_path = shot_dir / "scene.json"
            if out_path.exists() and not overwrite:
                continue
            write_json(out_path, scene_summary)

        if not shot_scene_embs:
            return {
                "num_shots": 0,
                "embeddings_file": None,
            }

        all_scene_embs = np.stack(shot_scene_embs, axis=0)
        embeddings_path = embeddings_root_path / "scene_embeddings.npz"
        np.savez_compressed(
            embeddings_path,
            embeddings=all_scene_embs,
            shot_ids=np.array(shot_ids, dtype=object),
            num_frames=np.array(shot_num_frames, dtype=np.int32),
        )

        return {
            "num_shots": len(shot_ids),
            "embeddings_file": str(embeddings_path),
        }

