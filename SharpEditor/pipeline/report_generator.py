"""Human-readable scene group report generator.

Reads the enriched ``scenes.json`` and produces a clean text report
matching the required output format:

    Scene Group 1
    Actors: Actor_A, Actor_B
    Location: Park

    Shots:
      take_001.mov
      take_003.mov
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class ReportGenerator:
    """Generate a human-readable scene report from scenes.json."""

    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> "ReportGenerator":
        """Instantiate (no config needed, but keeps API consistent)."""
        return ReportGenerator()

    def generate(
        self,
        embeddings_root: str,
        *,
        quiet: bool = False,
    ) -> str:
        """Read scenes.json and produce a text report.

        Returns the report string and writes it to scene_report.txt.
        """
        embeddings_path = Path(embeddings_root)
        scenes_file = embeddings_path / "scenes.json"

        if not scenes_file.exists():
            raise FileNotFoundError(
                f"scenes.json not found at {scenes_file}. "
                "Run the clustering stage first."
            )

        data = json.loads(scenes_file.read_text(encoding="utf-8"))
        scenes = data.get("scenes", {})
        ungrouped = data.get("ungrouped", [])
        ungrouped_files = data.get("ungrouped_files", ungrouped)

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("  SCENE GROUPING REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Total shots analyzed: {data.get('num_shots', 0)}")
        lines.append(f"Scenes found: {data.get('num_scenes', 0)}")
        lines.append(f"Ungrouped shots: {data.get('num_ungrouped', 0)}")
        lines.append("")
        lines.append("-" * 60)

        for idx, (scene_id, info) in enumerate(sorted(scenes.items()), start=1):
            actors = info.get("actors", [])
            location = info.get("location", "unknown")
            original_files = info.get("original_files", info.get("shot_ids", []))
            num_shots = info.get("num_shots", len(original_files))

            actors_display = ", ".join(
                a.replace("_", " ").title() for a in actors
            ) if actors else "None detected"
            location_display = location.replace("_", " ").title()

            lines.append("")
            lines.append(f"Scene Group {idx}")
            lines.append(f"Actors: {actors_display}")
            lines.append(f"Location: {location_display}")
            lines.append("")
            lines.append("Shots:")
            for f in original_files:
                lines.append(f"  {f}")
            lines.append("")
            lines.append("-" * 60)

        if ungrouped_files:
            lines.append("")
            lines.append("Ungrouped Shots (could not be assigned to a scene):")
            for f in ungrouped_files:
                lines.append(f"  {f}")
            lines.append("")
            lines.append("-" * 60)

        # Actor-to-scenes mapping
        actor_scene_map = data.get("actor_scene_map", {})
        if actor_scene_map:
            lines.append("")
            lines.append("=" * 60)
            lines.append("  ACTOR APPEARANCES")
            lines.append("=" * 60)
            lines.append("")

            # Build scene_id -> scene number mapping
            scene_number = {
                sid: idx for idx, (sid, _) in enumerate(sorted(scenes.items()), start=1)
            }

            for actor_id in sorted(actor_scene_map.keys()):
                scene_ids = actor_scene_map[actor_id]
                scene_names = [
                    f"Scene Group {scene_number.get(sid, sid)}"
                    for sid in sorted(scene_ids)
                ]
                actor_display = actor_id.replace("_", " ").title()
                lines.append(f"{actor_display}: {', '.join(scene_names)}")

            lines.append("")
            lines.append("-" * 60)

        report = "\n".join(lines) + "\n"

        # Write to file
        report_path = embeddings_path / "scene_report.txt"
        report_path.write_text(report, encoding="utf-8")

        if not quiet:
            print(report)

        return report
