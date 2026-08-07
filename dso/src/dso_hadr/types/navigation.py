"""Simulator-neutral navigation data structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

Point3 = tuple[float, float, float]


def as_point3(value: Point3 | list[float]) -> Point3:
    """Normalize a three-value point."""

    if len(value) != 3:
        raise ValueError("a 3D point must contain exactly three values")
    return float(value[0]), float(value[1]), float(value[2])


@dataclass(frozen=True)
class Pose:
    """Agent position in metres and yaw in radians."""

    x: float
    y: float
    z: float
    yaw: float

    @classmethod
    def from_value(cls, value: Pose | tuple[float, float, float, float]) -> Pose:
        if isinstance(value, Pose):
            return value
        return cls(*(float(component) for component in value))

    @property
    def position(self) -> Point3:
        return self.x, self.y, self.z

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.yaw]


@dataclass(frozen=True)
class NavMesh:
    """Agent-specific runtime navmesh exported by the simulator."""

    agent_type_id: int
    agent_radius: float
    movement_radius: float
    vertices: tuple[Point3, ...]
    triangles: tuple[tuple[int, int, int], ...]
    areas: tuple[int, ...]
    adjacency: tuple[tuple[int, int], ...]


class NavigationAction(str, Enum):
    """Discrete actions supported by the navigation controller."""

    STOP = "stop"
    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


@dataclass(frozen=True)
class Observation:
    """One aligned RGB-D observation and agent state."""

    rgb: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    depth: np.ndarray[tuple[int, ...], np.dtype[np.generic]]
    pose: Pose
    collision: bool


@dataclass(frozen=True)
class ShortestPath:
    """Ordered metric path returned by a simulator backend."""

    points: tuple[Point3, ...]
    geodesic_distance: float


@dataclass(frozen=True)
class FollowerResult:
    """Result of one waypoint-following attempt."""

    success: bool
    termination_reason: str
    stop_called: bool
    steps: int
    collisions: int
    traveled_distance: float
    final_distance: float
    final_pose: Pose
    trajectory: tuple[Pose, ...]
    actions: tuple[str, ...]


__all__ = [
    "FollowerResult",
    "NavMesh",
    "NavigationAction",
    "Observation",
    "Point3",
    "Pose",
    "ShortestPath",
    "as_point3",
]
