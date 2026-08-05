"""Discrete waypoint following over a navigation backend."""

from __future__ import annotations

import math
from collections.abc import Callable

from dso_hadr.simulator.navigation_backend import NavigationBackend
from dso_hadr.types.navigation import (
    FollowerResult,
    NavigationAction,
    Observation,
    Point3,
    Pose,
)

StepCallback = Callable[[int, Observation, NavigationAction], None]


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class WaypointFollower:
    """Turn and move forward until the final waypoint is reached."""

    def __init__(
        self,
        *,
        waypoint_tolerance: float,
        heading_tolerance_degrees: float,
        rotation_step_degrees: float,
    ) -> None:
        self.waypoint_tolerance = waypoint_tolerance
        self.heading_tolerance = math.radians(heading_tolerance_degrees)
        self.rotation_step = math.radians(rotation_step_degrees)

    def follow(
        self,
        backend: NavigationBackend,
        waypoints: tuple[Point3, ...],
        *,
        success_distance: float,
        max_steps: int,
        on_step: StepCallback | None = None,
    ) -> FollowerResult:
        target_index = 0
        pose = backend.get_agent_pose()
        trajectory: list[Pose] = [pose]
        actions: list[str] = []
        collisions = 0
        traveled_distance = 0.0
        stop_called = False

        while len(actions) < max_steps:
            move_distance = 0.0
            rotation_angle = 0.0
            if math.dist(pose.position, waypoints[-1]) < success_distance:
                action = NavigationAction.STOP
                stop_called = True
            else:
                while (
                    target_index < len(waypoints) - 1
                    and math.hypot(
                        pose.x - waypoints[target_index][0],
                        pose.z - waypoints[target_index][2],
                    )
                    <= self.waypoint_tolerance
                ):
                    target_index += 1
                target = waypoints[target_index]
                desired_yaw = math.atan2(-(target[0] - pose.x), -(target[2] - pose.z))
                heading_error = _wrap_angle(desired_yaw - pose.yaw)
                if abs(heading_error) <= self.heading_tolerance:
                    action = NavigationAction.MOVE_FORWARD
                    move_distance = math.hypot(
                        target[0] - pose.x,
                        target[2] - pose.z,
                    )
                elif heading_error > 0.0:
                    action = NavigationAction.TURN_LEFT
                    rotation_angle = min(heading_error, self.rotation_step)
                else:
                    action = NavigationAction.TURN_RIGHT
                    rotation_angle = max(heading_error, -self.rotation_step)

            observation = (
                backend.move_forward(move_distance, waypoints[target_index][1])
                if action is NavigationAction.MOVE_FORWARD
                else backend.rotate(rotation_angle)
                if action in (NavigationAction.TURN_LEFT, NavigationAction.TURN_RIGHT)
                else backend.step(action)
            )
            traveled_distance += math.dist(pose.position, observation.pose.position)
            collisions += int(observation.collision)
            pose = observation.pose
            actions.append(action.value)
            trajectory.append(pose)
            if on_step is not None:
                on_step(len(actions), observation, action)
            if stop_called or observation.collision:
                break

        final_distance = math.dist(pose.position, waypoints[-1])
        success = stop_called and final_distance < success_distance
        return FollowerResult(
            success=success,
            termination_reason=(
                "stop_within_success_distance"
                if success
                else "collision"
                if collisions
                else "max_steps"
            ),
            stop_called=stop_called,
            steps=len(actions),
            collisions=collisions,
            traveled_distance=traveled_distance,
            final_distance=final_distance,
            final_pose=pose,
            trajectory=tuple(trajectory),
            actions=tuple(actions),
        )


__all__ = ["WaypointFollower"]
