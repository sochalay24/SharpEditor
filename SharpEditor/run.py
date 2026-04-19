from __future__ import annotations

"""Unified pipeline entrypoint.

Running:

    python run.py

executes the dailies scene grouping pipeline in the same order and with the
same code paths that the final system will use. As additional stages are
implemented (actor recognition, scene encoding, temporal shot encoder,
clustering), they will be wired into this script without changing how you run
the project.
"""

import argparse
from pathlib import Path

from pipeline.clustering import SceneClusterer
from pipeline.face_detector import FaceDetector
from pipeline.frame_extractor import FrameExtractor
from pipeline.actor_recognition import ActorRecognizer
from pipeline.scene_encoder import SceneEncoder
from pipeline.shot_encoder import ShotEncoder
from pipeline.report_generator import ReportGenerator
from utils.config_utils import load_config
from utils.video_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dailies Scene AI full pipeline runner.")
    p.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config.",
    )
    p.add_argument(
        "--raw-shots",
        type=str,
        default="data/raw_shots",
        help="Directory containing raw video shots.",
    )
    p.add_argument(
        "--frames-out",
        type=str,
        default="data/frames",
        help="Directory to write extracted frames and per-shot manifests.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing intermediate manifests (frames, faces, etc.).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging/progress bars.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Stage runners (kept small and testable)
# -----------------------------------------------------------------------------

def run_frame_extraction(cfg: dict, raw_shots_dir: str, frames_out_dir: str, *, overwrite: bool, quiet: bool) -> None:
    """Stage 1: Shot preprocessing / frame sampling."""
    ensure_dir(raw_shots_dir)
    ensure_dir(frames_out_dir)

    extractor = FrameExtractor.from_config_dict(cfg)
    extractor.extract_dataset(
        raw_shots_dir=raw_shots_dir,
        frames_out_dir=frames_out_dir,
        overwrite=overwrite,
        quiet=quiet,
    )


def run_face_detection(cfg: dict, frames_out_dir: str, *, overwrite: bool, quiet: bool) -> None:
    """Stage 2: Face detection on extracted frames."""
    detector = FaceDetector.from_config_dict(cfg)
    detector.process_dataset(
        frames_root=frames_out_dir,
        overwrite=overwrite,
        quiet=quiet,
    )


def run_actor_recognition(cfg: dict, frames_out_dir: str, *, overwrite: bool, quiet: bool) -> None:
    """Stage 3: Actor recognition (faces → actor embeddings/IDs)."""
    recognizer = ActorRecognizer.from_config_dict(cfg)
    recognizer.process_dataset(
        frames_root=frames_out_dir,
        embeddings_root="data/embeddings",
        overwrite=overwrite,
        quiet=quiet,
    )


def run_scene_encoding(cfg: dict, frames_out_dir: str, *, overwrite: bool, quiet: bool) -> None:
    """Stage 4: Scene context encoding with CLIP."""
    encoder = SceneEncoder.from_config_dict(cfg)
    encoder.process_dataset(
        frames_root=frames_out_dir,
        embeddings_root="data/embeddings",
        overwrite=overwrite,
        quiet=quiet,
    )


def run_temporal_shot_encoding(cfg: dict, frames_out_dir: str, *, overwrite: bool, quiet: bool) -> None:
    """Stage 5: Temporal shot encoder producing 1024-D shot embeddings."""
    encoder = ShotEncoder.from_config_dict(cfg)
    encoder.process_dataset(
        frames_root=frames_out_dir,
        embeddings_root="data/embeddings",
        overwrite=overwrite,
        quiet=quiet,
    )


def run_scene_clustering(cfg: dict, frames_out_dir: str, *, overwrite: bool, quiet: bool) -> None:
    """Stage 6: Scene clustering (shot embeddings → scene groups)."""
    clusterer = SceneClusterer.from_config_dict(cfg)
    clusterer.process_dataset(
        frames_root=frames_out_dir,
        embeddings_root="data/embeddings",
        overwrite=overwrite,
        quiet=quiet,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    raw_shots_dir = args.raw_shots
    frames_out_dir = args.frames_out

    # 1. Shot preprocessing (frame sampling)
    print("=== Stage 1: Frame extraction ===")
    run_frame_extraction(
        cfg,
        raw_shots_dir=raw_shots_dir,
        frames_out_dir=frames_out_dir,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )
    print(f"Frames written under: {Path(frames_out_dir).resolve()}")

    # 2. Face detection
    print("=== Stage 2: Face detection ===")
    run_face_detection(
        cfg,
        frames_out_dir=frames_out_dir,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )
    print(f"Face crops and manifests written under shot dirs in: {Path(frames_out_dir).resolve()}")

    # 3. Actor recognition
    print("=== Stage 3: Actor recognition ===")
    run_actor_recognition(
        cfg,
        frames_out_dir=frames_out_dir,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )
    print(f"Actor embeddings and clusters written under: {Path('data/embeddings').resolve()}")

    # 4. Scene context encoding
    print("=== Stage 4: Scene encoding (CLIP) ===")
    run_scene_encoding(
        cfg,
        frames_out_dir=frames_out_dir,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )
    print(f"Scene embeddings written under: {Path('data/embeddings').resolve()}")

    # 5. Temporal shot encoding
    print("=== Stage 5: Temporal shot encoding ===")
    run_temporal_shot_encoding(
        cfg,
        frames_out_dir=frames_out_dir,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )
    print(f"Shot embeddings written under: {Path('data/embeddings').resolve()}")

    # 6. Scene clustering
    print("=== Stage 6: Scene clustering ===")
    run_scene_clustering(
        cfg,
        frames_out_dir=frames_out_dir,
        overwrite=args.overwrite,
        quiet=args.quiet,
    )
    print(f"Scene groups written under: {Path('data/embeddings').resolve()}")

    # 7. Report generation
    print("=== Stage 7: Generating scene report ===")
    reporter = ReportGenerator.from_config_dict(cfg)
    reporter.generate(
        embeddings_root="data/embeddings",
        quiet=args.quiet,
    )
    print(f"Report written to: {Path('data/embeddings/scene_report.txt').resolve()}")


if __name__ == "__main__":
    main()

