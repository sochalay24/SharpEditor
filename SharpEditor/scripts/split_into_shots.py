#!/usr/bin/env python3
"""Split full-length films into short clips to simulate raw dailies shots.

Uses ffmpeg scene-change detection to find natural cut points, then extracts
segments between those cuts as individual shot files.

Usage:
    python scripts/split_into_shots.py \
        --input-dir data/raw_downloads \
        --output-dir data/raw_shots \
        --max-shots-per-film 20 \
        --min-duration 3 \
        --max-duration 30
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def detect_scene_changes(video_path: str, threshold: float = 0.3) -> list[float]:
    """Use ffmpeg scene detection to find cut timestamps."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_frames",
        "-of", "json",
        "-f", "lavfi",
        f"movie={video_path},select='gt(scene\\,{threshold})'",
    ]
    # Fallback: use ffmpeg filter to detect scenes
    cmd2 = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(
            cmd2, capture_output=True, text=True, timeout=300
        )
        # Parse timestamps from showinfo output
        timestamps = []
        for line in result.stderr.split("\n"):
            if "showinfo" in line and "pts_time:" in line:
                for part in line.split():
                    if part.startswith("pts_time:"):
                        ts = float(part.split(":")[1])
                        timestamps.append(ts)
        return sorted(timestamps)
    except Exception as e:
        print(f"  Scene detection failed: {e}")
        return []


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_segment(
    video_path: str, output_path: str, start: float, duration: float
) -> bool:
    """Extract a segment from video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0


def split_film(
    video_path: Path,
    output_dir: Path,
    *,
    max_shots: int = 20,
    min_duration: float = 3.0,
    max_duration: float = 30.0,
    scene_threshold: float = 0.3,
) -> list[dict]:
    """Split a film into shot clips using scene detection."""
    film_name = video_path.stem.replace(" ", "_")[:40]
    print(f"\nProcessing: {video_path.name}")

    # Get duration
    total_duration = get_video_duration(str(video_path))
    print(f"  Duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")

    # Detect scene changes
    print(f"  Detecting scene changes (threshold={scene_threshold})...")
    cuts = detect_scene_changes(str(video_path), threshold=scene_threshold)
    print(f"  Found {len(cuts)} scene changes")

    if not cuts:
        # Fallback: uniform segmentation
        print("  Falling back to uniform segmentation...")
        segment_len = 15.0  # 15-second segments
        cuts = [i * segment_len for i in range(int(total_duration / segment_len) + 1)]

    # Build segments from consecutive cut points
    # Add start and end
    all_points = [0.0] + cuts + [total_duration]
    segments = []
    for i in range(len(all_points) - 1):
        start = all_points[i]
        end = all_points[i + 1]
        dur = end - start
        if min_duration <= dur <= max_duration:
            segments.append((start, dur))

    # If too few segments after filtering, relax constraints
    if len(segments) < max_shots // 2:
        segments = []
        for i in range(len(all_points) - 1):
            start = all_points[i]
            end = all_points[i + 1]
            dur = end - start
            if dur >= min_duration:
                # Cap at max_duration
                segments.append((start, min(dur, max_duration)))

    # Sample evenly across the film
    if len(segments) > max_shots:
        step = len(segments) / max_shots
        segments = [segments[int(i * step)] for i in range(max_shots)]

    print(f"  Extracting {len(segments)} shots...")

    results = []
    for idx, (start, dur) in enumerate(segments):
        shot_name = f"{film_name}_shot{idx:03d}"
        out_path = output_dir / f"{shot_name}.mp4"
        ok = extract_segment(str(video_path), str(out_path), start, dur)
        if ok:
            size_mb = out_path.stat().st_size / (1024 * 1024)
            results.append({
                "shot_id": shot_name,
                "source_film": video_path.name,
                "start_time": round(start, 3),
                "duration": round(dur, 3),
                "file": str(out_path.name),
                "size_mb": round(size_mb, 2),
            })
            print(f"    [{idx+1}/{len(segments)}] {shot_name} @ {start:.1f}s ({dur:.1f}s) -> {size_mb:.1f}MB")
        else:
            print(f"    [{idx+1}/{len(segments)}] FAILED: {shot_name}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Split films into shot clips")
    parser.add_argument("--input-dir", default="data/raw_downloads")
    parser.add_argument("--output-dir", default="data/raw_shots")
    parser.add_argument("--max-shots-per-film", type=int, default=20)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--scene-threshold", type=float, default=0.3)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".ogv"}
    videos = sorted(
        f for f in input_dir.iterdir()
        if f.suffix.lower() in video_exts and f.stat().st_size > 1_000_000
    )

    if not videos:
        print(f"No video files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(videos)} films to split")

    all_shots = []
    for v in videos:
        shots = split_film(
            v, output_dir,
            max_shots=args.max_shots_per_film,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            scene_threshold=args.scene_threshold,
        )
        all_shots.extend(shots)

    # Write manifest
    manifest_path = output_dir / "shots_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_shots, f, indent=2)

    total_size = sum(s["size_mb"] for s in all_shots)
    print(f"\n=== Done ===")
    print(f"Total shots: {len(all_shots)}")
    print(f"Total size: {total_size:.1f}MB")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
