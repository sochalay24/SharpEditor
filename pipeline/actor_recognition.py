"""Actor recognition stage (faces → actor embeddings/IDs).

Implementation: facenet-pytorch InceptionResnetV1 (ArcFace-style embeddings).

This stage consumes the face crops and detection manifests produced by the
face detector, computes 512-D face embeddings, clusters them into actor IDs
using HDBSCAN, and writes:

- Dataset-level embedding arrays under ``data/embeddings/``
- Per-shot ``actors.json`` files summarizing which actors appear in each shot
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import hdbscan
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

from utils.device_utils import device_from_config
from utils.embedding_utils import l2_normalize
from utils.video_utils import ensure_dir, write_json


@dataclass(frozen=True)
class ActorRecognitionConfig:
    """Configuration for actor recognition and clustering."""

    model: str = "inception_resnet_v1"
    pretrained: str = "vggface2"
    batch_size: int = 64
    l2_normalize: bool = True
    # HDBSCAN parameters
    min_cluster_size: int = 5
    min_samples: int = 1
    cluster_selection_epsilon: float = 0.0


class ActorRecognizer:
    """Compute actor embeddings and cluster them into actor IDs."""

    def __init__(self, cfg: ActorRecognitionConfig, *, device_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg
        self.device = device_from_config(device_cfg)

        if self.cfg.model != "inception_resnet_v1":
            raise ValueError(f"Unsupported actor model: {self.cfg.model}")

        self.model = (
            InceptionResnetV1(pretrained=self.cfg.pretrained)
            .eval()
            .to(self.device)
        )

        self.transform = T.Compose(
            [
                T.Resize((160, 160)),
                T.ToTensor(),
                fixed_image_standardization,
            ]
        )

    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> "ActorRecognizer":
        """Instantiate from the global YAML config dict."""
        ar = d.get("actor_recognition", {}) or {}
        device_cfg = d.get("device", {}) or {}
        clustering = ar.get("clustering", {}) or {}

        cfg = ActorRecognitionConfig(
            model=str(ar.get("model", "inception_resnet_v1")),
            pretrained=str(ar.get("pretrained", "vggface2")),
            batch_size=int(ar.get("batch_size", 64)),
            l2_normalize=bool(ar.get("l2_normalize", True)),
            min_cluster_size=int(clustering.get("min_cluster_size", 5)),
            min_samples=int(clustering.get("min_samples", 1)),
            cluster_selection_epsilon=float(clustering.get("cluster_selection_epsilon", 0.0)),
        )
        return ActorRecognizer(cfg, device_cfg=device_cfg)

    # -------------------------------------------------------------------------
    # Core helpers
    # -------------------------------------------------------------------------

    def _load_faces_from_shot(self, shot_dir: Path) -> List[Dict[str, Any]]:
        """Load face records from ``faces.json`` inside a shot directory.

        Prefer MTCNN-aligned face images when available; otherwise fall back
        to the raw cropped faces.
        """
        faces_manifest = shot_dir / "faces.json"
        if not faces_manifest.exists():
            return []
        data = json.loads(faces_manifest.read_text(encoding="utf-8"))
        records: List[Dict[str, Any]] = []
        for _, dets in data.get("detections_by_image", {}).items():
            for det in dets:
                # Skip detections that somehow lack a face image
                if not det.get("face_image"):
                    continue
                records.append(det)
        return records

    def _compute_embeddings(self, face_paths: List[str], *, quiet: bool) -> np.ndarray:
        """Compute 512-D embeddings for a list of face image paths."""
        embeddings: List[torch.Tensor] = []
        batch: List[torch.Tensor] = []

        paths_iter = face_paths if quiet else tqdm(face_paths, desc="Computing face embeddings")

        for path in paths_iter:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
            batch.append(tensor)
            if len(batch) >= self.cfg.batch_size:
                batch_tensor = torch.stack(batch).to(self.device)
                with torch.no_grad():
                    emb = self.model(batch_tensor)
                    if self.cfg.l2_normalize:
                        emb = l2_normalize(emb, dim=1)
                embeddings.append(emb.cpu())
                batch = []

        if batch:
            batch_tensor = torch.stack(batch).to(self.device)
            with torch.no_grad():
                emb = self.model(batch_tensor)
                if self.cfg.l2_normalize:
                    emb = l2_normalize(emb, dim=1)
            embeddings.append(emb.cpu())

        if not embeddings:
            return np.zeros((0, 512), dtype=np.float32)

        all_emb = torch.cat(embeddings, dim=0)
        return all_emb.numpy().astype(np.float32)

    def _cluster_actors(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings into actor IDs using HDBSCAN.

        We operate on L2-normalized embeddings and use Euclidean distance; on
        the unit hypersphere this is monotonically related to cosine distance,
        so clustering behavior is effectively cosine-like while remaining
        compatible with the installed hdbscan/ sklearn stack.
        """
        if embeddings.shape[0] == 0:
            return np.empty((0,), dtype=int)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            cluster_selection_epsilon=self.cfg.cluster_selection_epsilon,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)
        return labels

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
        """Compute actor embeddings and clusters for an entire dataset.

        Args:
            frames_root: Root directory containing per-shot frame folders.
            embeddings_root: Directory where actor embedding files will be written.
        """
        frames_root_path = Path(frames_root)
        if not frames_root_path.exists():
            raise FileNotFoundError(f"Frames root not found: {frames_root}")

        ensure_dir(embeddings_root)
        embeddings_root_path = Path(embeddings_root)

        shot_dirs = sorted([p for p in frames_root_path.iterdir() if p.is_dir()])

        # Collect all face records across shots.
        all_face_paths: List[str] = []
        all_shot_ids: List[str] = []

        it = shot_dirs if quiet else tqdm(shot_dirs, desc="Loading face records")
        for shot_dir in it:
            shot_id = shot_dir.name
            records = self._load_faces_from_shot(shot_dir)
            for r in records:
                # Prefer aligned faces when available
                face_path = r.get("aligned_face_image") or r.get("face_image")
                if not face_path:
                    continue
                all_face_paths.append(face_path)
                all_shot_ids.append(shot_id)

        if not all_face_paths:
            return {
                "num_faces": 0,
                "num_actors": 0,
                "embeddings_file": None,
                "actors_file": None,
            }

        # 1) Compute embeddings for all faces.
        emb_array = self._compute_embeddings(all_face_paths, quiet=quiet)

        # 2) Cluster embeddings into actor IDs.
        labels = self._cluster_actors(emb_array)

        # Map cluster labels to contiguous actor IDs (skip noise label -1).
        unique_labels = sorted({int(l) for l in labels if int(l) >= 0})
        label_to_actor_id = {lab: f"actor_{i:03d}" for i, lab in enumerate(unique_labels)}
        actor_id_per_face: List[Optional[str]] = []
        for lab in labels:
            if int(lab) < 0:
                actor_id_per_face.append(None)
            else:
                actor_id_per_face.append(label_to_actor_id[int(lab)])

        # 3) Save dataset-level embeddings.
        embeddings_path = embeddings_root_path / "actor_embeddings.npz"
        np.savez_compressed(
            embeddings_path,
            embeddings=emb_array,
            face_paths=np.array(all_face_paths, dtype=object),
            shot_ids=np.array(all_shot_ids, dtype=object),
            cluster_labels=labels,
        )

        # 4) Build dataset-level actor summary, including mean embeddings.
        actors: Dict[str, Dict[str, Any]] = {}
        for idx, actor_id in enumerate(actor_id_per_face):
            if actor_id is None:
                continue
            entry = actors.setdefault(
                actor_id,
                {
                    "faces": [],
                    "cluster_label": int(labels[idx]),
                    "_embedding_sum": np.zeros(emb_array.shape[1], dtype=np.float32),
                    "_count": 0,
                },
            )
            entry["faces"].append(
                {
                    "face_image": all_face_paths[idx],
                    "shot_id": all_shot_ids[idx],
                    "index": idx,
                }
            )
            entry["_embedding_sum"] += emb_array[idx]
            entry["_count"] += 1

        # Finalize actor-level mean embeddings (L2-normalized).
        for actor_id, entry in actors.items():
            count = max(int(entry.pop("_count", 1)), 1)
            emb_sum = entry.pop("_embedding_sum")
            mean_emb = emb_sum / float(count)
            mean_emb = mean_emb / max(np.linalg.norm(mean_emb) + 1e-12, 1e-12)
            entry["embedding_mean"] = mean_emb.tolist()

        actors_file = embeddings_root_path / "actors.json"
        write_json(
            actors_file,
            {
                "num_faces": len(all_face_paths),
                "num_actors": len(actors),
                "actors": actors,
            },
        )

        # 5) Write per-shot actor summaries, with per-shot mean embeddings.
        per_shot: Dict[str, Dict[str, Any]] = {}
        for idx, (shot_id, actor_id) in enumerate(zip(all_shot_ids, actor_id_per_face)):
            shot_entry = per_shot.setdefault(
                shot_id,
                {
                    "shot_id": shot_id,
                    "actors": {},
                    "faces": [],
                    "_actor_embedding_sum": {},
                    "_actor_count": {},
                },
            )
            shot_entry["faces"].append(
                {
                    "face_image": all_face_paths[idx],
                    "actor_id": actor_id,
                    "cluster_label": int(labels[idx]),
                }
            )
            if actor_id is not None:
                shot_entry["actors"].setdefault(actor_id, 0)
                shot_entry["actors"][actor_id] += 1
                # accumulate embeddings per actor within this shot
                shot_entry["_actor_embedding_sum"].setdefault(
                    actor_id, np.zeros(emb_array.shape[1], dtype=np.float32)
                )
                shot_entry["_actor_count"].setdefault(actor_id, 0)
                shot_entry["_actor_embedding_sum"][actor_id] += emb_array[idx]
                shot_entry["_actor_count"][actor_id] += 1

        for shot_id, summary in per_shot.items():
            # finalize per-shot actor mean embeddings
            actor_embs: Dict[str, List[float]] = {}
            for actor_id, emb_sum in summary["_actor_embedding_sum"].items():
                count = max(int(summary["_actor_count"].get(actor_id, 1)), 1)
                mean_emb = emb_sum / float(count)
                mean_emb = mean_emb / max(np.linalg.norm(mean_emb) + 1e-12, 1e-12)
                actor_embs[actor_id] = mean_emb.tolist()

            summary.pop("_actor_embedding_sum", None)
            summary.pop("_actor_count", None)
            summary["actor_embeddings"] = actor_embs

            shot_dir = frames_root_path / shot_id
            out_path = shot_dir / "actors.json"
            if out_path.exists() and not overwrite:
                continue
            write_json(out_path, summary)

        return {
            "num_faces": len(all_face_paths),
            "num_actors": len(actors),
            "embeddings_file": str(embeddings_path),
            "actors_file": str(actors_file),
        }

