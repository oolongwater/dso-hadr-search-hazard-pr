"""AI2-THOR implementation of the local navigation backend."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ai2thor.server import Event  # type: ignore[import-not-found]

from dso_hadr.simulator.navigation_backend import NavigationBackend
from dso_hadr.types.navigation import (
    NavigationAction,
    Observation,
    Point3,
    Pose,
    ShortestPath,
    as_point3,
)
from dso_hadr.simulator.ai2thor import ProcTHORSimulator
from dso_hadr.simulator.control import move_ahead, rotate_left, rotate_right
from dso_hadr.utils.coordinates import (
    navigation_yaw_to_native,
    native_yaw_to_navigation,
)


@dataclass(frozen=True)
class AI2THORNavigationConfig:
    move_magnitude: float
    rotation_degrees: float
    reachable_grid_size: float
    path_allowed_error: float
    navigable_tolerance: float


def _point(document: dict[str, float]) -> Point3:
    return float(document["x"]), float(document["y"]), float(document["z"])


def _point_document(point: Point3) -> dict[str, float]:
    return {"x": point[0], "y": point[1], "z": point[2]}


def _path_length(points: tuple[Point3, ...]) -> float:
    return sum(math.dist(source, target) for source, target in zip(points, points[1:]))


class AI2THORNavigationBackend(NavigationBackend):
    """Translate shared discrete navigation operations to AI2-THOR actions."""

    def __init__(
        self,
        simulator: ProcTHORSimulator,
        scene_paths: dict[str, Path],
        config: AI2THORNavigationConfig,
    ) -> None:
        self.simulator = simulator
        self.scene_paths = dict(scene_paths)
        self.config = config
        self._scene_id: str | None = None
        self._event: Event | None = None
        self._floor_heights: tuple[float, ...] = ()
        self._stored_horizon = 0.0
        self._stored_standing = True
        self._agent_y_offset = 0.0
        self._reachable_points: tuple[Point3, ...] = ()
        self._random = random.Random()
        self._closed = False

    @property
    def navmesh_samples(self) -> tuple[Point3, ...]:
        return self._reachable_points

    def _load_scene_records(self, scene_path: Path) -> None:
        house = json.loads(scene_path.read_text(encoding="utf-8"))
        metadata = house["metadata"]
        stored_agent = metadata["agent"]
        self._stored_horizon = float(stored_agent["horizon"])
        self._stored_standing = bool(stored_agent["standing"])

        if metadata["schema"] == "2.0.0":
            self._floor_heights = tuple(float(floor["baseY"]) for floor in house["floors"])
        else:
            room = house["rooms"][0]
            self._floor_heights = (float(room["floorPolygon"][0]["y"]),)

    def _current_event(self) -> Event:
        if self._event is None:
            raise RuntimeError("AI2-THOR navigation backend has not been reset")
        return self._event

    def _pose_from_event(self, event: Event) -> Pose:
        agent = event.metadata["agent"]
        position = agent["position"]
        return Pose(
            x=float(position["x"]),
            y=float(position["y"]) - self._agent_y_offset,
            z=float(position["z"]),
            yaw=native_yaw_to_navigation(float(agent["rotation"]["y"])),
        )

    def _observation(self, event: Event, *, collision: bool) -> Observation:
        return Observation(
            rgb=np.ascontiguousarray(event.frame, dtype=np.uint8).copy(),
            depth=np.ascontiguousarray(event.depth_frame, dtype=np.float32).copy(),
            pose=self._pose_from_event(event),
            collision=collision,
        )

    def reset(
        self,
        scene_id: str,
        start_pose: Pose | None,
        seed: int | None = None,
    ) -> Observation:
        scene_path = self.scene_paths[scene_id].expanduser().resolve(strict=True)
        self._load_scene_records(scene_path)
        self._scene_id = scene_id
        self._random.seed(seed)

        loaded = self.simulator.load_scene(scene_path)
        loaded_y = float(loaded.metadata["agent"]["position"]["y"])
        nearest_floor_y = min(
            self._floor_heights,
            key=lambda height: abs(height - loaded_y),
        )
        self._agent_y_offset = loaded_y - nearest_floor_y

        reachable = self.simulator.step(
            {
                "action": "GetReachablePositions",
                "gridSize": self.config.reachable_grid_size,
            }
        )
        self._reachable_points = tuple(
            (
                float(point["x"]),
                float(point["y"]) - self._agent_y_offset,
                float(point["z"]),
            )
            for point in reachable.metadata["actionReturn"]
        )

        if start_pose is None:
            pose = Pose(*self.sample_navigable_point(), 0.0)
        else:
            requested_pose = Pose.from_value(start_pose)
            start_point = min(
                self._reachable_points,
                key=lambda point: math.dist(requested_pose.position, point),
            )
            pose = Pose(*start_point, requested_pose.yaw)

        self._event = self.simulator.step(
            {
                "action": "TeleportFull",
                "position": _point_document((pose.x, pose.y + self._agent_y_offset, pose.z)),
                "rotation": {
                    "x": 0.0,
                    "y": navigation_yaw_to_native(pose.yaw),
                    "z": 0.0,
                },
                "horizon": self._stored_horizon,
                "standing": self._stored_standing,
            }
        )
        if not self._event:
            raise RuntimeError(self._event.metadata["errorMessage"])
        return self._observation(self._event, collision=False)

    def step(self, action: NavigationAction) -> Observation:
        if action is NavigationAction.MOVE_FORWARD:
            return self.move_forward(self.config.move_magnitude)
        native_actions: dict[NavigationAction, dict[str, object]] = {
            NavigationAction.TURN_LEFT: rotate_right(self.config.rotation_degrees),
            NavigationAction.TURN_RIGHT: rotate_left(self.config.rotation_degrees),
            NavigationAction.STOP: {"action": "Pass"},
        }
        self._event = self.simulator.step(native_actions[action])
        return self._observation(self._event, collision=False)

    def move_forward(self, distance: float, target_y: float | None = None) -> Observation:
        native_target_y = None if target_y is None else target_y + self._agent_y_offset
        self._event = self.simulator.step(move_ahead(distance, native_target_y))
        collision = not self._event.metadata["lastActionSuccess"]
        return self._observation(self._event, collision=collision)

    def rotate(self, angle_radians: float) -> Observation:
        angle_degrees = math.degrees(angle_radians)
        action = rotate_right(angle_degrees) if angle_degrees > 0.0 else rotate_left(-angle_degrees)
        self._event = self.simulator.step(action)
        return self._observation(self._event, collision=False)

    def get_observation(self) -> Observation:
        self._event = self.simulator.step({"action": "Pass"})
        return self._observation(self._event, collision=False)

    def sample_navigable_point(self) -> Point3:
        return self._random.choice(self._reachable_points)

    def get_navmesh_path(
        self,
        start: Point3,
        goal: Point3,
        max_path_length: float,
    ) -> ShortestPath | None:
        navmesh_start = as_point3(start)
        navmesh_goal = as_point3(goal)
        event = self.simulator.step(
            {
                "action": "GetShortestPathToPoint",
                "position": _point_document(navmesh_start),
                "target": _point_document(navmesh_goal),
                "allowedError": self.config.path_allowed_error,
                "sampleFromNavmesh": False,
            }
        )
        if not event:
            return None
        points = [navmesh_start]
        for corner in event.metadata["actionReturn"]["corners"]:
            point = _point(corner)
            if point != points[-1]:
                points.append(point)
        if navmesh_goal != points[-1]:
            points.append(navmesh_goal)
        transition = tuple(points)

        coordinate_groups: list[list[Point3]] = [[transition[0]]]
        for point in transition[1:]:
            previous = coordinate_groups[-1][-1]
            if math.isclose(point[0], previous[0]) and math.isclose(point[2], previous[2]):
                coordinate_groups[-1].append(point)
            else:
                coordinate_groups.append([point])
        transition = (
            coordinate_groups[0][0],
            *(group[-1] for group in coordinate_groups[1:]),
        )
        if len(transition) == 1 and navmesh_start != navmesh_goal:
            return None
        length = _path_length(transition)
        if length > max_path_length:
            return None

        saved_event = self._current_event()
        saved_agent = saved_event.metadata["agent"]
        traversable = True
        for source, target in zip(transition, transition[1:]):
            delta_x = target[0] - source[0]
            delta_z = target[2] - source[2]
            distance = math.hypot(delta_x, delta_z)
            if math.isclose(distance, 0.0):
                return None
            yaw = math.atan2(-delta_x, -delta_z)
            positioned = self.simulator.step(
                {
                    "action": "TeleportFull",
                    "position": _point_document(
                        (source[0], source[1] + self._agent_y_offset, source[2])
                    ),
                    "rotation": {
                        "x": 0.0,
                        "y": navigation_yaw_to_native(yaw),
                        "z": 0.0,
                    },
                    "horizon": self._stored_horizon,
                    "standing": self._stored_standing,
                    "forceAction": True,
                }
            )
            if not positioned:
                traversable = False
                break
            moved = self.simulator.step(move_ahead(distance, target[1] + self._agent_y_offset))
            if not moved:
                traversable = False
                break

        restored = self.simulator.step(
            {
                "action": "TeleportFull",
                "position": dict(saved_agent["position"]),
                "rotation": dict(saved_agent["rotation"]),
                "horizon": self._stored_horizon,
                "standing": self._stored_standing,
            }
        )
        if not restored:
            raise RuntimeError(restored.metadata["errorMessage"])
        self._event = restored
        if not traversable:
            return None

        return ShortestPath(points=transition, geodesic_distance=length)

    def get_agent_pose(self) -> Pose:
        return self._pose_from_event(self._current_event())

    def get_scene_metadata(self) -> dict[str, object]:
        return {
            "scene_id": self._scene_id,
            "reachable_position_count": len(self._reachable_points),
            "agent_y_offset": self._agent_y_offset,
        }

    def close(self) -> None:
        if not self._closed:
            self.simulator.close()
            self._closed = True


__all__ = ["AI2THORNavigationBackend", "AI2THORNavigationConfig"]
