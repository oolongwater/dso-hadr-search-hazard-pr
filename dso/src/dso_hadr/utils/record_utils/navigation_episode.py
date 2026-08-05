"""Persistence helpers for recorded navigation episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from dso_hadr.types.navigation import NavigationAction, Observation


def prepare_navigation_episode_output(
    output_directory: Path,
) -> tuple[Path, Path]:
    """Create a new episode directory and return it with its trajectory file."""

    resolved_output = output_directory.expanduser().resolve()
    (resolved_output / "rgb").mkdir(parents=True)
    (resolved_output / "depth").mkdir()
    trajectory_path = resolved_output / "trajectory.jsonl"
    trajectory_path.touch()
    return resolved_output, trajectory_path


def write_json(path: Path, document: dict[str, object]) -> None:
    """Write a deterministic, human-readable JSON document."""

    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def record_observation(
    *,
    output_directory: Path,
    trajectory_path: Path,
    scene_id: str,
    step_id: int,
    action: NavigationAction | None,
    observation: Observation,
) -> None:
    """Save one aligned RGB, depth, pose, action, and collision record."""

    rgb_path = output_directory / "rgb" / f"{step_id:06d}.png"
    depth_path = output_directory / "depth" / f"{step_id:06d}.npy"
    Image.fromarray(observation.rgb).save(rgb_path)
    np.save(depth_path, observation.depth)
    record: dict[str, object] = {
        "scene_id": scene_id,
        "step_id": step_id,
        "action": None if action is None else action.value,
        "collision": observation.collision,
        "pose": observation.pose.as_list(),
        "rgb": rgb_path.relative_to(output_directory).as_posix(),
        "depth": depth_path.relative_to(output_directory).as_posix(),
    }
    with trajectory_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


__all__ = [
    "prepare_navigation_episode_output",
    "record_observation",
    "write_json",
]
