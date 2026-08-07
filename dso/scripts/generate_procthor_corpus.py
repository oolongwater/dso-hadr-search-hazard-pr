"""Generate a connected ProcTHOR corpus and exact long-horizon episodes."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ai2thor.controller import Controller

from dso_hadr.scenes.long_horizon import select_episode
from dso_hadr.scenes.procthor import create_manifest, load_corpus_config, write_manifest
from dso_hadr.scenes.scene_graph import extract_scene_graph_task
from dso_hadr.scenes.traversability import build_traversability
from dso_hadr.simulator.ai2thor_backend import parse_navmesh

_controller: Controller | None = None


def _resolve(config_path: Path, value: str, *, strict: bool) -> Path:
    return (config_path.parent / value).resolve(strict=strict)


def _controller_for(executable: Path, generation: dict[str, Any]) -> Controller:
    global _controller
    if _controller is None:
        _controller = Controller(
            scene="Procedural",
            local_executable_path=str(executable),
            width=int(generation["width"]),
            height=int(generation["height"]),
            quality=str(generation["quality"]),
            renderDepthImage=False,
            snapToGrid=bool(generation["snap_to_grid"]),
            fullscreen=bool(generation["fullscreen"]),
            gpu_device=0,
        )
        atexit.register(_controller.stop)
    return _controller


def _rejection_kind(error: Exception) -> str:
    message = str(error)
    if "maximum room-pair geodesic is too short" in message:
        return "short_horizon_scene"
    if "disconnected" in message:
        return "disconnected_topology"
    if "at least two rooms" in message or "no distinct room pair" in message:
        return "insufficient_rooms"
    return type(error).__name__


def _generate_scene(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        scene_id,
        scene_index,
        floor_count,
        base_seed,
        output_directory,
        executable,
        generation,
        generator_repository,
        max_attempts,
        move_magnitude,
        meters_per_pixel,
        navigable_tolerance,
        minimum_geodesic_distance,
        navmesh_clearance_radius,
    ) = task
    repository = str(generator_repository)
    if repository not in sys.path:
        sys.path.insert(0, repository)
    from procthor.generation import HouseGenerator
    from procthor.utils.types import InvalidFloorplan, InvalidMultiFloorPlan

    controller = _controller_for(Path(executable), generation)
    output_path = Path(output_directory) / f"{scene_id}.json"
    candidate_path = output_path.with_suffix(f".json.{os.getpid()}.candidate")
    navigation_map_path = output_path.with_suffix(f".json.{os.getpid()}.geojson")
    rejections: Counter[str] = Counter()
    recent_errors: list[str] = []
    for attempt in range(1, int(max_attempts) + 1):
        candidate_seed = int(base_seed) + (attempt - 1) * 1_000_000
        try:
            generator = HouseGenerator(
                split=str(generation["split"]),
                seed=candidate_seed,
                num_floors=int(floor_count),
                controller=controller,
            )
            house, _ = generator.sample()
            warnings = house.validate(controller)
            if warnings:
                raise ValueError(
                    "house validation warnings: " + json.dumps(warnings, sort_keys=True)
                )
            candidate_path.write_text(house.to_json() + "\n", encoding="utf-8")
            triangulated = controller.step(
                action="DSOHADRGetNavMeshTriangulation",
                renderImage=False,
            )
            if not triangulated:
                raise RuntimeError(triangulated.metadata["errorMessage"])
            action_return = triangulated.metadata["actionReturn"]
            if not isinstance(action_return, dict):
                raise TypeError("runtime navmesh action return must be a JSON object")
            navmesh = parse_navmesh(action_return)
            if navmesh.agent_radius + 1e-6 < float(navmesh_clearance_radius):
                raise ValueError(
                    "runtime navmesh agent radius is below the corpus clearance "
                    f"contract: {navmesh.agent_radius} < {navmesh_clearance_radius}"
                )
            traversability = build_traversability(
                navmesh,
                move_magnitude=float(move_magnitude),
            )
            room_ids = tuple(str(room["id"]) for room in house.data["rooms"])
            if len(room_ids) < 2:
                raise ValueError("long-horizon navigation requires at least two rooms")
            graph_task = extract_scene_graph_task(
                candidate_path,
                traversability,
                start_room_id=room_ids[0],
                goal_room_id=room_ids[-1],
                meters_per_pixel=float(meters_per_pixel),
                navigation_map_path=navigation_map_path,
            )
            episode = select_episode(
                graph_task,
                navigable_tolerance=float(navigable_tolerance),
                minimum_geodesic_distance=float(minimum_geodesic_distance),
            )
        except (InvalidFloorplan, InvalidMultiFloorPlan, KeyError, TypeError, ValueError) as error:
            rejections[_rejection_kind(error)] += 1
            recent_errors.append(str(error))
            candidate_path.unlink(missing_ok=True)
            navigation_map_path.unlink(missing_ok=True)
            continue

        house.data["metadata"]["warnings"] = {}
        candidate_path.write_text(house.to_json() + "\n", encoding="utf-8")
        navigation_map_path.unlink(missing_ok=True)
        os.replace(candidate_path, output_path)
        return {
            "scene_id": scene_id,
            "scene_index": scene_index,
            "status": "PASS",
            "attempt": attempt,
            "generation_seed": candidate_seed,
            "floor_count": floor_count,
            "room_count": len(house.data["rooms"]),
            "object_count": len(house.data["objects"]),
            "navmesh_triangle_count": len(navmesh.triangles),
            "navmesh_component_count": 1,
            "navmesh_agent_radius": navmesh.agent_radius,
            "movement_collision_radius": navmesh.movement_radius,
            "clearance_margin": navmesh.agent_radius - navmesh.movement_radius,
            "rejections": dict(sorted(rejections.items())),
            "episode": {
                "scene_id": scene_id,
                "start_room_id": episode.start_room_id,
                "goal_room_id": episode.goal_room_id,
                "start_position": list(episode.start_position),
                "goal_position": list(episode.goal_position),
                "geodesic_distance": episode.geodesic_distance,
                "crosses_floors": episode.crosses_floors,
            },
        }
    candidate_path.unlink(missing_ok=True)
    navigation_map_path.unlink(missing_ok=True)
    return {
        "scene_id": scene_id,
        "scene_index": scene_index,
        "status": "FAILED",
        "floor_count": floor_count,
        "attempt": max_attempts,
        "rejections": dict(sorted(rejections.items())),
        "recent_errors": recent_errors[-5:],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--episodes-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve(strict=True)
    document = json.loads(config_path.read_text(encoding="utf-8"))
    generation = document["generation"]
    selection = document["scene_selection"]
    validation = document["validation"]
    navigation = document["navigation"]
    repository = _resolve(config_path, document["source"]["repository"], strict=True)
    executable = _resolve(config_path, generation["local_executable_path"], strict=True)
    if str(repository) not in sys.path:
        sys.path.insert(0, str(repository))

    output_directory = args.output_directory.expanduser().resolve(strict=False)
    output_directory.mkdir(parents=True, exist_ok=False)
    floor_random = random.Random(int(generation["floor_count_seed"]))
    floor_counts = tuple(
        floor_random.choice(tuple(int(value) for value in generation["floor_count_choices"]))
        for _ in range(int(selection["count"]))
    )
    tasks = []
    for offset, floor_count in enumerate(floor_counts):
        scene_index = int(selection["first_index"]) + offset
        scene_id = Path(str(selection["filename_template"]).format(index=scene_index)).stem
        tasks.append(
            (
                scene_id,
                scene_index,
                floor_count,
                int(generation["seed_start"]) + offset,
                str(output_directory),
                str(executable),
                generation,
                str(repository),
                int(generation["max_attempts_per_scene"]),
                float(navigation["move_magnitude"]),
                float(navigation["meters_per_pixel"]),
                float(navigation["navigable_tolerance"]),
                float(validation["minimum_geodesic_distance"]),
                float(validation["navmesh_clearance_radius"]),
            )
        )

    records = []
    with ProcessPoolExecutor(max_workers=int(generation["worker_count"])) as executor:
        futures = [executor.submit(_generate_scene, task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    records.sort(key=lambda record: int(record["scene_index"]))
    report = {"scene_count": len(records), "records": records}
    (output_directory / "generation-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = [record for record in records if record["status"] != "PASS"]
    if failed:
        raise RuntimeError(f"{len(failed)} scenes failed connected long-horizon generation")

    corpus_config = load_corpus_config(config_path)
    manifest = create_manifest(output_directory, corpus_config)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest, args.manifest_output)
    episodes = {
        "episode_config": "navigation.json",
        "episodes": [record["episode"] for record in records],
    }
    args.episodes_output.parent.mkdir(parents=True, exist_ok=True)
    args.episodes_output.write_text(
        json.dumps(episodes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
