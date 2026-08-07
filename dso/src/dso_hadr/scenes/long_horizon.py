"""Deterministic long-horizon episode selection on a physical navmesh."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dso_hadr.graph.model import SceneGraphTask
from dso_hadr.planner.motion.astar import astar_search
from dso_hadr.planner.symbolic.dijkstra import dijkstra_search
from dso_hadr.scenes.traversability import select_region_point
from dso_hadr.types.navigation import Point3


@dataclass(frozen=True)
class Episode:
    """The maximum-geodesic semantic room pair for one scene."""

    start_room_id: str
    goal_room_id: str
    start_position: Point3
    goal_position: Point3
    geodesic_distance: float
    crosses_floors: bool


def select_episode(
    task: SceneGraphTask,
    *,
    navigable_tolerance: float,
    minimum_geodesic_distance: float,
) -> Episode:
    """Select the exact farthest room-representative pair under one corpus policy.

    Multi-floor houses consider every cross-floor room pair and orient the
    episode upward. Single-floor houses consider every distinct room pair.
    Every candidate is evaluated by the same A* implementation used by the
    episode runner on the exported physical triangulation.
    """

    if minimum_geodesic_distance <= 0.0:
        raise ValueError("minimum geodesic distance must be positive")
    regions = tuple(sorted(task.graph.regions, key=lambda region: region.id))
    if len(regions) < 2:
        raise ValueError("long-horizon navigation requires at least two rooms")
    floor_index = {floor.id: floor.level_index for floor in task.graph.floors}
    multi_floor = len(task.graph.floors) > 1
    points = {
        region.id: select_region_point(
            task,
            region.id,
            region.navigation_pose.position,
            navigable_tolerance,
        )
        for region in regions
    }

    candidates: list[Episode] = []
    for index, first in enumerate(regions):
        for second in regions[index + 1 :]:
            crosses_floors = first.floor_id != second.floor_id
            if multi_floor and not crosses_floors:
                continue
            if crosses_floors:
                ordered = sorted(
                    (first, second),
                    key=lambda region: (floor_index[region.floor_id], region.id),
                )
                start, goal = ordered[0], ordered[1]
            else:
                start, goal = first, second
            dijkstra_search(task.graph, start.id, goal.id)
            path = astar_search(
                task.graph.traversability_map,
                points[start.id],
                points[goal.id],
            )
            candidates.append(
                Episode(
                    start_room_id=start.id,
                    goal_room_id=goal.id,
                    start_position=points[start.id],
                    goal_position=points[goal.id],
                    geodesic_distance=path.geodesic_distance,
                    crosses_floors=crosses_floors,
                )
            )
    if not candidates:
        requirement = "cross-floor room pair" if multi_floor else "distinct room pair"
        raise ValueError(f"scene has no {requirement}")
    selected = min(
        candidates,
        key=lambda candidate: (
            -candidate.geodesic_distance,
            candidate.start_room_id,
            candidate.goal_room_id,
        ),
    )
    if selected.geodesic_distance + 1e-9 < minimum_geodesic_distance:
        raise ValueError(
            "scene maximum room-pair geodesic is too short: "
            f"{selected.geodesic_distance:.6f} m < {minimum_geodesic_distance:.6f} m"
        )
    if not math.isfinite(selected.geodesic_distance):
        raise ValueError("scene maximum room-pair geodesic is not finite")
    return selected


__all__ = ["Episode", "select_episode"]
