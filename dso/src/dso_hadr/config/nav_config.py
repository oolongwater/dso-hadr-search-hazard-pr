"""Configuration for one symbolic navigation and trajectory episode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NavigationEpisodeConfig:
    """All file, scene, graph, controller, and recording parameters for one run."""

    config_path: Path
    corpus_config_path: Path
    scene_manifest_path: Path
    scenes_directory: Path
    simulator_config_path: Path
    output_directory: Path
    scene_id: str
    start_room_id: str
    goal_room_id: str
    seed: int
    meters_per_pixel: float
    move_magnitude: float
    rotation_degrees: float
    navigable_tolerance: float
    waypoint_tolerance: float
    heading_tolerance_degrees: float
    success_distance: float
    max_steps: int


def _resolve(config_path: Path, value: str, *, strict: bool) -> Path:
    return (config_path.parent / value).resolve(strict=strict)


def load_navigation_episode_config(path: Path) -> NavigationEpisodeConfig:
    """Load one episode without supplying hidden defaults."""

    config_path = path.expanduser().resolve(strict=True)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    paths = document["paths"]
    episode = document["episode"]
    scene_graph = document["scene_graph"]
    traversability_map = document["traversability_map"]
    controller = document["controller"]
    return NavigationEpisodeConfig(
        config_path=config_path,
        corpus_config_path=_resolve(config_path, paths["corpus_config"], strict=True),
        scene_manifest_path=_resolve(
            config_path,
            paths["scene_manifest"],
            strict=True,
        ),
        scenes_directory=_resolve(
            config_path,
            paths["scenes_directory"],
            strict=True,
        ),
        simulator_config_path=_resolve(
            config_path,
            paths["simulator_config"],
            strict=True,
        ),
        output_directory=_resolve(
            config_path,
            paths["output_directory"],
            strict=False,
        ),
        scene_id=episode["scene_id"],
        start_room_id=episode["start_room_id"],
        goal_room_id=episode["goal_room_id"],
        seed=episode["seed"],
        meters_per_pixel=scene_graph["meters_per_pixel"],
        move_magnitude=controller["move_magnitude"],
        rotation_degrees=controller["rotation_degrees"],
        navigable_tolerance=traversability_map["navigable_tolerance"],
        waypoint_tolerance=controller["waypoint_tolerance"],
        heading_tolerance_degrees=controller["heading_tolerance_degrees"],
        success_distance=controller["success_distance"],
        max_steps=controller["max_steps"],
    )


__all__ = ["NavigationEpisodeConfig", "load_navigation_episode_config"]
