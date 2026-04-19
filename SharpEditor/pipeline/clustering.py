"""Scene clustering stage (shot embeddings → scene groups).

Clusters shot embeddings using HDBSCAN, optionally boosted by actor
co-occurrence similarity. Produces scene group assignments and a
dataset-level scene manifest enriched with actor names, location labels,
and original video filenames.

Outputs:
- Dataset-level ``data/embeddings/scenes.json`` with enriched scene groups
- Per-shot ``scene_group.json`` files with assigned scene ID
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import hdbscan
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from utils.video_utils import ensure_dir, write_json


@dataclass(frozen=True)
class ClusteringConfig:
    """Configuration for scene clustering."""

    algorithm: str = "hdbscan"
    min_cluster_size: int = 2
    min_samples: int = 1
    cluster_selection_epsilon: float = 0.0
    # Weight for actor co-occurrence in the fused distance matrix.
    # 0.0 = shot embeddings only, 1.0 = actor similarity only.
    actor_weight: float = 0.3


class SceneClusterer:
    """Cluster shots into scene groups based on shot embeddings."""

    def __init__(self, cfg: ClusteringConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> "SceneClusterer":
        """Instantiate from the global YAML config dict."""
        cl = d.get("clustering", {}) or {}
        cfg = ClusteringConfig(
            algorithm=str(cl.get("algorithm", "hdbscan")),
            min_cluster_size=int(cl.get("min_cluster_size", 2)),
            min_samples=int(cl.get("min_samples", 1)),
            cluster_selection_epsilon=float(cl.get("cluster_selection_epsilon", 0.0)),
            actor_weight=float(cl.get("actor_weight", 0.3)),
        )
        return SceneClusterer(cfg)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_actor_similarity(
        self, shot_ids: List[str], frames_root: Path
    ) -> np.ndarray:
        """Build a shot-by-shot actor co-occurrence similarity matrix."""
        n = len(shot_ids)
        actor_sets: List[set] = []
        all_actors: set = set()

        for sid in shot_ids:
            actors_path = frames_root / sid / "actors.json"
            actors: set = set()
            if actors_path.exists():
                data = json.loads(actors_path.read_text(encoding="utf-8"))
                actors = set(data.get("actors", {}).keys())
            actor_sets.append(actors)
            all_actors.update(actors)

        if not all_actors:
            return np.zeros((n, n), dtype=np.float32)

        actor_list = sorted(all_actors)
        actor_idx = {a: i for i, a in enumerate(actor_list)}
        presence = np.zeros((n, len(actor_list)), dtype=np.float32)
        for i, actors in enumerate(actor_sets):
            for a in actors:
                presence[i, actor_idx[a]] = 1.0

        sim = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                intersection = np.minimum(presence[i], presence[j]).sum()
                union = np.maximum(presence[i], presence[j]).sum()
                if union > 0:
                    s = intersection / union
                else:
                    s = 0.0
                sim[i, j] = s
                sim[j, i] = s

        return sim

    def _cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Run HDBSCAN on a precomputed distance matrix."""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.cfg.min_cluster_size,
            min_samples=self.cfg.min_samples,
            cluster_selection_epsilon=self.cfg.cluster_selection_epsilon,
            metric="precomputed",
        )
        labels = clusterer.fit_predict(distance_matrix)
        return labels

    def _get_shot_actors(self, shot_id: str, frames_root: Path) -> List[str]:
        """Get list of actor IDs present in a shot."""
        actors_path = frames_root / shot_id / "actors.json"
        if not actors_path.exists():
            return []
        data = json.loads(actors_path.read_text(encoding="utf-8"))
        return list(data.get("actors", {}).keys())

    def _get_shot_location(self, shot_id: str, frames_root: Path) -> Optional[str]:
        """Get location label for a shot from scene.json."""
        scene_path = frames_root / shot_id / "scene.json"
        if not scene_path.exists():
            return None
        data = json.loads(scene_path.read_text(encoding="utf-8"))
        return data.get("location_label")

    def _get_original_filename(self, shot_id: str, frames_root: Path) -> str:
        """Get original video filename from the shot's manifest.json."""
        manifest_path = frames_root / shot_id / "manifest.json"
        if not manifest_path.exists():
            return shot_id
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        video_path = data.get("video_path", "")
        if video_path:
            return Path(video_path).name
        return shot_id

    def _determine_scene_location(self, locations: List[Optional[str]]) -> str:
        """Determine the dominant location for a scene via majority vote."""
        valid = [loc for loc in locations if loc is not None]
        if not valid:
            return "unknown"
        counter = Counter(valid)
        return counter.most_common(1)[0][0]

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
        """Cluster shots into scene groups with enriched metadata."""
        frames_root_path = Path(frames_root)
        embeddings_root_path = Path(embeddings_root)
        ensure_dir(embeddings_root)

        # Load shot embeddings
        shot_emb_file = embeddings_root_path / "shot_embeddings.npz"
        if not shot_emb_file.exists():
            raise FileNotFoundError(
                f"Shot embeddings not found at {shot_emb_file}. "
                "Run the shot encoding stage first."
            )

        data = np.load(shot_emb_file, allow_pickle=True)
        embeddings = data["embeddings"]  # [N, 1024]
        shot_ids = [str(s) for s in data["shot_ids"]]
        n = len(shot_ids)

        if not quiet:
            print(f"  Clustering {n} shots (embedding dim={embeddings.shape[1]})")

        # 1. Cosine similarity from shot embeddings
        shot_sim = cosine_similarity(embeddings)  # [N, N] in [-1, 1]

        # 2. Actor co-occurrence similarity
        actor_sim = self._build_actor_similarity(shot_ids, frames_root_path)

        # 3. Fuse similarities
        w = self.cfg.actor_weight
        fused_sim = (1.0 - w) * shot_sim + w * actor_sim

        # Convert similarity to distance (clamp to [0, 2])
        distance_matrix = np.clip(1.0 - fused_sim, 0.0, 2.0).astype(np.float64)
        # Zero out diagonal
        np.fill_diagonal(distance_matrix, 0.0)

        # 4. Cluster
        labels = self._cluster(distance_matrix)

        # Map to scene IDs
        unique_labels = sorted({int(l) for l in labels if int(l) >= 0})
        label_to_scene = {l: f"scene_{i:03d}" for i, l in enumerate(unique_labels)}

        scenes: Dict[str, List[str]] = {}
        ungrouped: List[str] = []

        for sid, lab in zip(shot_ids, labels):
            lab = int(lab)
            if lab < 0:
                ungrouped.append(sid)
            else:
                scene_id = label_to_scene[lab]
                scenes.setdefault(scene_id, []).append(sid)

        # 5. Enrich scene groups with actors, location, and original filenames
        enriched_scenes: Dict[str, Dict[str, Any]] = {}
        for scene_id, members in scenes.items():
            # Collect actors across all shots in this scene
            all_scene_actors: set = set()
            locations: List[Optional[str]] = []
            original_files: List[str] = []

            for sid in members:
                all_scene_actors.update(self._get_shot_actors(sid, frames_root_path))
                locations.append(self._get_shot_location(sid, frames_root_path))
                original_files.append(self._get_original_filename(sid, frames_root_path))

            location = self._determine_scene_location(locations)

            enriched_scenes[scene_id] = {
                "actors": sorted(all_scene_actors),
                "location": location,
                "shot_ids": members,
                "original_files": original_files,
                "num_shots": len(members),
            }

        # Get original filenames for ungrouped shots too
        ungrouped_files = [
            self._get_original_filename(sid, frames_root_path) for sid in ungrouped
        ]

        # 6. Build actor-to-scenes reverse mapping
        actor_scene_map: Dict[str, List[str]] = {}
        for scene_id, info in enriched_scenes.items():
            for actor_id in info["actors"]:
                actor_scene_map.setdefault(actor_id, []).append(scene_id)

        # 7. Write dataset-level scene manifest
        scene_manifest = {
            "num_shots": n,
            "num_scenes": len(enriched_scenes),
            "num_ungrouped": len(ungrouped),
            "algorithm": self.cfg.algorithm,
            "actor_weight": self.cfg.actor_weight,
            "scenes": enriched_scenes,
            "actor_scene_map": actor_scene_map,
            "ungrouped": ungrouped,
            "ungrouped_files": ungrouped_files,
        }
        scenes_file = embeddings_root_path / "scenes.json"
        write_json(scenes_file, scene_manifest)

        # 7. Write per-shot scene group assignment
        pairs = list(zip(shot_ids, labels))
        it = pairs if quiet else tqdm(pairs, desc="Writing scene assignments")
        for sid_val, lab_val in it:
            shot_dir = frames_root_path / str(sid_val)
            out_path = shot_dir / "scene_group.json"
            if out_path.exists() and not overwrite:
                continue

            lab_int = int(lab_val)
            scene_id = label_to_scene.get(lab_int)
            original_file = self._get_original_filename(str(sid_val), frames_root_path)
            write_json(out_path, {
                "shot_id": str(sid_val),
                "original_file": original_file,
                "scene_id": scene_id,
                "cluster_label": lab_int,
                "is_ungrouped": lab_int < 0,
            })

        if not quiet:
            print(f"  Found {len(enriched_scenes)} scenes from {n} shots ({len(ungrouped)} ungrouped)")
            for scene_id, info in sorted(enriched_scenes.items()):
                actors_str = ", ".join(info["actors"]) if info["actors"] else "none detected"
                print(f"    {scene_id}: {info['num_shots']} shots | Location: {info['location']} | Actors: {actors_str}")

        return {
            "num_shots": n,
            "num_scenes": len(enriched_scenes),
            "num_ungrouped": len(ungrouped),
            "scenes_file": str(scenes_file),
        }
