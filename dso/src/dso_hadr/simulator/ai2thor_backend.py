"""AI2-THOR implementation of the local navigation backend."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ai2thor.server import Event  # type: ignore[import-not-found]

from dso_hadr.simulator.ai2thor import ProcTHORSimulator
from dso_hadr.simulator.control import move_ahead, rotate_left, rotate_right
from dso_hadr.simulator.navigation_backend import NavigationBackend
from dso_hadr.types.navigation import (
    NavigationAction,
    NavMesh,
    Observation,
    Point3,
    Pose,
)
from dso_hadr.utils.coordinates import (
    native_yaw_to_navigation,
    navigation_yaw_to_native,
)


@dataclass(frozen=True)
class AI2THORNavigationConfig:
    move_magnitude: float
    rotation_degrees: float


def _point(document: dict[str, float]) -> Point3:
    return float(document["x"]), float(document["y"]), float(document["z"])


def _point_document(point: Point3) -> dict[str, float]:
    return {"x": point[0], "y": point[1], "z": point[2]}


def _triangle_centroid(navmesh: NavMesh, triangle_id: int) -> Point3:
    triangle = navmesh.triangles[triangle_id]
    vertices = tuple(navmesh.vertices[index] for index in triangle)
    return (
        sum(point[0] for point in vertices) / 3.0,
        sum(point[1] for point in vertices) / 3.0,
        sum(point[2] for point in vertices) / 3.0,
    )


def _nearest_triangle_centroid(navmesh: NavMesh, target: Point3) -> Point3:
    return min(
        (_triangle_centroid(navmesh, index) for index in range(len(navmesh.triangles))),
        key=lambda point: (math.dist(target, point), point),
    )


def parse_navmesh(document: dict[str, object]) -> NavMesh:
    """Parse and validate one physical Unity navmesh export."""

    raw_vertices = document["vertices"]
    raw_indices = document["indices"]
    raw_areas = document["areas"]
    raw_adjacency = document["adjacency"]
    if not isinstance(raw_vertices, list):
        raise TypeError("runtime navmesh vertices must be a list")
    if not isinstance(raw_indices, list) or not isinstance(raw_areas, list):
        raise TypeError("runtime navmesh indices and areas must be lists")
    if not isinstance(raw_adjacency, list):
        raise TypeError("runtime navmesh adjacency must be a list")
    vertices = tuple(_point(point) for point in raw_vertices)
    indices = tuple(int(index) for index in raw_indices)
    areas = tuple(int(area) for area in raw_areas)
    adjacency_indices = tuple(int(index) for index in raw_adjacency)
    if len(indices) % 3 != 0:
        raise ValueError("runtime navmesh returned an incomplete triangle index list")
    triangles = tuple(
        (indices[offset], indices[offset + 1], indices[offset + 2])
        for offset in range(0, len(indices), 3)
    )
    if not vertices or not triangles:
        raise ValueError("runtime navmesh triangulation is empty")
    if len(areas) != len(triangles):
        raise ValueError("runtime navmesh area count does not match its triangles")
    if any(index < 0 or index >= len(vertices) for triangle in triangles for index in triangle):
        raise ValueError("runtime navmesh triangle references an invalid vertex")
    if len(adjacency_indices) % 2 != 0:
        raise ValueError("runtime navmesh returned an incomplete adjacency index list")
    adjacency = tuple(
        (adjacency_indices[offset], adjacency_indices[offset + 1])
        for offset in range(0, len(adjacency_indices), 2)
    )
    if any(
        node_a < 0
        or node_a >= len(triangles)
        or node_b < 0
        or node_b >= len(triangles)
        or node_a == node_b
        for node_a, node_b in adjacency
    ):
        raise ValueError("runtime navmesh adjacency references an invalid triangle")
    agent_type_id = document["agentTypeId"]
    if isinstance(agent_type_id, bool) or not isinstance(agent_type_id, int):
        raise TypeError("runtime navmesh agentTypeId must be an integer")
    agent_radius = document["agentRadius"]
    movement_radius = document["movementRadius"]
    if (
        isinstance(agent_radius, bool)
        or not isinstance(agent_radius, (int, float))
        or float(agent_radius) <= 0.0
    ):
        raise TypeError("runtime navmesh agentRadius must be a positive number")
    if (
        isinstance(movement_radius, bool)
        or not isinstance(movement_radius, (int, float))
        or float(movement_radius) <= 0.0
    ):
        raise TypeError("runtime navmesh movementRadius must be a positive number")
    if float(agent_radius) + 1e-6 < float(movement_radius):
        raise ValueError(
            "runtime navmesh clearance is smaller than the movement collision "
            f"footprint: {float(agent_radius)} < {float(movement_radius)}"
        )
    return NavMesh(
        agent_type_id=agent_type_id,
        agent_radius=float(agent_radius),
        movement_radius=float(movement_radius),
        vertices=vertices,
        triangles=triangles,
        areas=areas,
        adjacency=adjacency,
    )


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
        self._navmesh: NavMesh | None = None
        self._random = random.Random()
        self._closed = False

    @property
    def navmesh(self) -> NavMesh:
        if self._navmesh is None:
            raise RuntimeError("AI2-THOR navigation backend has not been reset")
        return self._navmesh

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

        triangulated = self.simulator.step(
            {
                "action": "DSOHADRGetNavMeshTriangulation",
                "renderImage": False,
            }
        )
        if not triangulated:
            raise RuntimeError(triangulated.metadata["errorMessage"])
        document = triangulated.metadata["actionReturn"]
        if not isinstance(document, dict):
            raise TypeError("runtime navmesh action return must be a JSON object")
        self._navmesh = parse_navmesh(document)

        if start_pose is None:
            triangle_id = self._random.randrange(len(self.navmesh.triangles))
            start_point = _triangle_centroid(self.navmesh, triangle_id)
            pose = Pose(*start_point, 0.0)
        else:
            requested_pose = Pose.from_value(start_pose)
            start_point = _nearest_triangle_centroid(self.navmesh, requested_pose.position)
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

    def get_agent_pose(self) -> Pose:
        return self._pose_from_event(self._current_event())

    def get_scene_metadata(self) -> dict[str, object]:
        return {
            "scene_id": self._scene_id,
            "navmesh_agent_type_id": self.navmesh.agent_type_id,
            "navmesh_agent_radius": self.navmesh.agent_radius,
            "movement_collision_radius": self.navmesh.movement_radius,
            "navmesh_vertex_count": len(self.navmesh.vertices),
            "navmesh_triangle_count": len(self.navmesh.triangles),
            "navmesh_adjacency_count": len(self.navmesh.adjacency),
            "agent_y_offset": self._agent_y_offset,
        }

    def close(self) -> None:
        if not self._closed:
            self.simulator.close()
            self._closed = True


__all__ = [
    "AI2THORNavigationBackend",
    "AI2THORNavigationConfig",
    "parse_navmesh",
]
