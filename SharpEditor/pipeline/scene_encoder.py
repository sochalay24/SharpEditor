"""Scene context encoding stage (frames → CLIP scene embeddings + location labels).

Implementation: CLIP Vision + Text (openai/clip-vit-base-patch32).

This stage consumes per-shot frames (typically keyframes), encodes them with
the CLIP vision encoder, aggregates per-shot scene embeddings, and performs
zero-shot location classification against a configurable list of candidate
location labels.

Outputs:
- Dataset-level scene embeddings under ``data/embeddings/scene_embeddings.npz``
- Per-shot ``scene.json`` files with scene embedding + location label
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from utils.device_utils import device_from_config
from utils.video_utils import ensure_dir, write_json


# Default location candidates for zero-shot classification
DEFAULT_LOCATION_CANDIDATES = [
    "park",
    "living room",
    "bedroom",
    "kitchen",
    "office",
    "street",
    "restaurant",
    "car interior",
    "hallway",
    "rooftop",
    "beach",
    "forest",
    "hospital",
    "school",
    "courtyard",
    "garden",
    "parking lot",
    "warehouse",
    "bar",
    "studio",
    "balcony",
    "staircase",
    "bathroom",
    "church",
    "library",
    "gym",
    "subway",
    "airport",
    "bridge",
    "countryside",
]


@dataclass(frozen=True)
class SceneEncoderConfig:
    """Configuration for CLIP-based scene encoding."""

    model_name: str = "openai/clip-vit-base-patch32"
    frame_source: str = "keyframes"  # 'keyframes' | 'sampled' | 'both'
    batch_size: int = 16
    location_candidates: Tuple[str, ...] = tuple(DEFAULT_LOCATION_CANDIDATES)


class SceneEncoder:
    """Encode scene context from frames using CLIP Vision Transformer.

    Also performs zero-shot location classification using CLIP text encoder.
    """

    def __init__(self, cfg: SceneEncoderConfig, *, device_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg
        self.device = device_from_config(device_cfg)

        # Load the full CLIP model (vision + text) for zero-shot classification
        self.processor = CLIPProcessor.from_pretrained(self.cfg.model_name)
        self.model = CLIPModel.from_pretrained(self.cfg.model_name).to(self.device)
        self.model.eval()

        # Hidden size (e.g. 768 for ViT-B/32) inferred from config
        self.hidden_size = self.model.config.vision_config.hidden_size

        # Pre-compute text embeddings for location candidates
        self._location_labels = list(self.cfg.location_candidates)
        self._location_text_embs = self._encode_location_texts()

    def _encode_location_texts(self) -> torch.Tensor:
        """Pre-compute CLIP text embeddings for all location candidates."""
        prompts = [f"a photo of a {loc}" for loc in self._location_labels]
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            text_outputs = self.model.text_model(**inputs)
            # Use pooler_output (CLS token) and project to shared space
            text_embs = self.model.text_projection(text_outputs.pooler_output)  # [N_locs, proj_dim]
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
        return text_embs

    def _classify_location(self, image_features: torch.Tensor) -> Tuple[str, float]:
        """Zero-shot classify the location using CLIP similarity.

        Args:
            image_features: [N_frames, emb_dim] CLIP vision embeddings for the shot.

        Returns:
            (location_label, confidence_score)
        """
        # Ensure features are on the same device as the text embeddings
        image_features = image_features.to(self._location_text_embs.device)

        # Mean-pool frame features to get a single shot-level visual embedding
        mean_feat = image_features.mean(dim=0, keepdim=True)  # [1, emb_dim]
        mean_feat = mean_feat / mean_feat.norm(dim=-1, keepdim=True)

        # Cosine similarity with all location text embeddings
        similarities = (mean_feat @ self._location_text_embs.T).squeeze(0)  # [N_locs]
        probs = similarities.softmax(dim=-1)

        best_idx = probs.argmax().item()
        best_label = self._location_labels[best_idx]
        best_score = float(probs[best_idx].item())

        return best_label, best_score

    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> "SceneEncoder":
        """Instantiate from the global YAML config dict."""
        se = d.get("scene_encoding", {}) or {}
        device_cfg = d.get("device", {}) or {}

        location_candidates = se.get("location_candidates", None)
        if location_candidates is None:
            location_candidates = DEFAULT_LOCATION_CANDIDATES
        location_candidates = tuple(str(c) for c in location_candidates)

        cfg = SceneEncoderConfig(
            model_name=str(se.get("model_name", "openai/clip-vit-base-patch32")),
            frame_source=str(se.get("frame_source", "keyframes")).lower(),
            batch_size=int(se.get("batch_size", 16)),
            location_candidates=location_candidates,
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

    def _encode_images(self, image_paths: List[Path], *, quiet: bool) -> Tuple[np.ndarray, torch.Tensor]:
        """Encode images into CLIP vision embeddings.

        Returns:
            (pooled_embs_np [N, hidden_size], projected_features [N, proj_dim])
        """
        if not image_paths:
            return np.zeros((0, self.hidden_size), dtype=np.float32), torch.zeros(0, 512)

        all_pooled: List[torch.Tensor] = []
        all_projected: List[torch.Tensor] = []

        it = image_paths if quiet else tqdm(image_paths, desc="Encoding scene frames")

        batch_images: List[Image.Image] = []
        for path in it:
            img = Image.open(path).convert("RGB")
            batch_images.append(img)
            if len(batch_images) >= self.cfg.batch_size:
                inputs = self.processor(images=batch_images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                with torch.no_grad():
                    outputs = self.model.vision_model(pixel_values=pixel_values)
                    pooled = outputs.pooler_output  # [B, hidden_size]
                    # Get projected features for zero-shot classification
                    projected = self.model.visual_projection(pooled)  # [B, proj_dim]
                all_pooled.append(pooled.cpu())
                all_projected.append(projected.cpu())
                batch_images = []

        if batch_images:
            inputs = self.processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.no_grad():
                outputs = self.model.vision_model(pixel_values=pixel_values)
                pooled = outputs.pooler_output
                projected = self.model.visual_projection(pooled)
            all_pooled.append(pooled.cpu())
            all_projected.append(projected.cpu())

        if not all_pooled:
            return np.zeros((0, self.hidden_size), dtype=np.float32), torch.zeros(0, 512)

        pooled_embs = torch.cat(all_pooled, dim=0)
        projected_embs = torch.cat(all_projected, dim=0)
        return pooled_embs.numpy().astype(np.float32), projected_embs

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
        """Compute scene embeddings and location labels for all shots in a dataset."""
        frames_root_path = Path(frames_root)
        if not frames_root_path.exists():
            raise FileNotFoundError(f"Frames root not found: {frames_root}")

        ensure_dir(embeddings_root)
        embeddings_root_path = Path(embeddings_root)

        shot_dirs = sorted([p for p in frames_root_path.iterdir() if p.is_dir()])

        shot_ids: List[str] = []
        shot_scene_embs: List[np.ndarray] = []
        shot_num_frames: List[int] = []
        shot_locations: List[Dict[str, Any]] = []

        it = shot_dirs if quiet else tqdm(shot_dirs, desc="Scene encoding for dataset")
        for shot_dir in it:
            shot_id = shot_dir.name
            frame_paths = self._select_frame_paths(shot_dir)
            if not frame_paths:
                continue

            per_frame_embs, projected_features = self._encode_images(frame_paths, quiet=quiet)
            if per_frame_embs.shape[0] == 0:
                continue

            # Aggregate per-shot scene embedding as mean of frame embeddings.
            scene_emb = per_frame_embs.mean(axis=0)

            # Zero-shot location classification
            location_label, location_score = self._classify_location(projected_features)

            shot_ids.append(shot_id)
            shot_scene_embs.append(scene_emb)
            shot_num_frames.append(int(per_frame_embs.shape[0]))
            shot_locations.append({
                "label": location_label,
                "score": location_score,
            })

            # Write per-shot scene summary JSON (now with location).
            scene_summary = {
                "shot_id": shot_id,
                "num_frames_used": int(per_frame_embs.shape[0]),
                "frame_source": self.cfg.frame_source,
                "embedding_dim": int(scene_emb.shape[0]),
                "scene_embedding": scene_emb.tolist(),
                "location_label": location_label,
                "location_score": round(location_score, 4),
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

        if not quiet:
            for sid, loc in zip(shot_ids, shot_locations):
                print(f"    {sid}: location={loc['label']} (score={loc['score']:.3f})")

        return {
            "num_shots": len(shot_ids),
            "embeddings_file": str(embeddings_path),
            "locations": dict(zip(shot_ids, shot_locations)),
        }
