"""Ground symbolic region routes into simulator metric paths."""

from __future__ import annotations

from dso_hadr.graph.model import SceneGraphTask, SymbolicPlan
from dso_hadr.scenes.traversability import select_region_traversability_point
from dso_hadr.types.navigation import Point3


def select_goal_point(
    task: SceneGraphTask,
    target: Point3,
    navigable_tolerance: float,
) -> Point3:
    """Select the traversability node nearest the goal region representative."""

    return select_region_traversability_point(
        task,
        task.goal_region_id,
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
            else select_region_traversability_point(
                task,
                subgoal.region_id,
                subgoal.target_pose.position,
                navigable_tolerance,
            )
        )
    return tuple(targets) if targets else (goal_point,)


__all__ = ["ground_symbolic_subgoals", "select_goal_point"]
