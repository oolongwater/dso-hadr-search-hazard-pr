"""Ground symbolic region routes into simulator metric paths."""

from __future__ import annotations

import math

from dso_hadr.graph.model import SceneGraphTask, SymbolicPlan
from dso_hadr.types.navigation import Point3


def _select_region_point(
    task: SceneGraphTask,
    region_id: str,
    target: Point3,
    navigable_tolerance: float,
) -> Point3:
    region = next(region for region in task.graph.regions if region.id == region_id)
    grid = next(grid for grid in task.floor_grids if grid.floor_id == region.floor_id)
    candidates: list[Point3] = []
    for point in task.graph.traversability_map.nodes:
        if abs(point[1] - grid.floor_height) > navigable_tolerance:
            continue
        row = math.floor((point[2] - grid.origin_xz[1]) / grid.meters_per_pixel + 0.5 + 1e-9)
        column = math.floor((point[0] - grid.origin_xz[0]) / grid.meters_per_pixel + 0.5 + 1e-9)
        if row < 0 or row >= grid.traversable.shape[0]:
            continue
        if column < 0 or column >= grid.traversable.shape[1]:
            continue
        if grid.semantic_regions[row, column] == region.semantic_region_value:
            candidates.append(point)
    return min(
        candidates,
        key=lambda point: (
            math.dist(point, target),
            point,
        ),
    )


def select_goal_point(
    task: SceneGraphTask,
    target: Point3,
    navigable_tolerance: float,
) -> Point3:
    """Select the traversability node nearest the goal region representative."""

    region = next(region for region in task.graph.regions if region.id == task.goal_region_id)
    return _select_region_point(
        task,
        region.id,
        target,
        navigable_tolerance,
    )


def ground_symbolic_subgoals(
    task: SceneGraphTask,
    plan: SymbolicPlan,
    goal_point: Point3,
    navigable_tolerance: float,
) -> tuple[Point3, ...]:
    """Map symbolic subgoals to nodes in the scene traversability map."""

    targets: list[Point3] = []
    for index, subgoal in enumerate(plan.subgoals[1:], start=1):
        targets.append(
            goal_point
            if index == len(plan.subgoals) - 1
            else _select_region_point(
                task,
                subgoal.region_id,
                subgoal.target_pose.position,
                navigable_tolerance,
            )
        )
    return tuple(targets) if targets else (goal_point,)


__all__ = ["ground_symbolic_subgoals", "select_goal_point"]
