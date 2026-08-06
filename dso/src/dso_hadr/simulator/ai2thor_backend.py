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


def _polygon_center(polygon: list[dict[str, float]]) -> Point3:
    if not polygon:
        raise ValueError("a connector landing polygon cannot be empty")
    return (
        sum(float(point["x"]) for point in polygon) / len(polygon),
        sum(float(point["y"]) for point in polygon) / len(polygon),
        sum(float(point["z"]) for point in polygon) / len(polygon),
    )


def _point_in_polygon(point: tuple[float, float], polygon: list[dict[str, float]]) -> bool:
    """Return whether an XZ point lies inside or on a scene polygon."""

    if len(polygon) < 3:
        return False
    point_x, point_z = point
    inside = False
    for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        start_x, start_z = float(start["x"]), float(start["z"])
        end_x, end_z = float(end["x"]), float(end["z"])
        cross = (point_x - start_x) * (end_z - start_z) - (point_z - start_z) * (end_x - start_x)
        if (
            abs(cross) <= 1e-8
            and min(start_x, end_x) - 1e-8 <= point_x <= max(start_x, end_x) + 1e-8
            and min(start_z, end_z) - 1e-8 <= point_z <= max(start_z, end_z) + 1e-8
        ):
            return True
        if (start_z > point_z) != (end_z > point_z):
            crossing_x = start_x + (point_z - start_z) * (end_x - start_x) / (end_z - start_z)
            if point_x < crossing_x:
                inside = not inside
    return inside


def _door_path(
    door: dict[str, object],
    walls_by_id: dict[str, dict[str, object]],
) -> tuple[Point3, Point3]:
    """Derive the two room-side doorway anchors from schema geometry."""

    wall = walls_by_id[str(door["wall0"])]
    raw_wall_polygon = wall["polygon"]
    raw_hole_polygon = door["holePolygon"]
    if not isinstance(raw_wall_polygon, list) or not isinstance(raw_hole_polygon, list):
        raise TypeError("door and wall polygons must be lists")
    wall_start = _point(raw_wall_polygon[0])
    wall_end = _point(raw_wall_polygon[1])
    direction_x = wall_end[0] - wall_start[0]
    direction_z = wall_end[2] - wall_start[2]
    wall_length = math.hypot(direction_x, direction_z)
    if wall_length <= 0.0:
        raise ValueError(f"door {door['id']!r} belongs to a zero-length wall")
    hole = tuple(_point(point) for point in raw_hole_polygon)
    local_center = (min(point[0] for point in hole) + max(point[0] for point in hole)) / 2.0
    center_x = wall_start[0] + direction_x * local_center / wall_length
    center_z = wall_start[2] + direction_z * local_center / wall_length
    normal_x, normal_z = direction_z / wall_length, -direction_x / wall_length
    return (
        (center_x + normal_x * 0.35, wall_start[1], center_z + normal_z * 0.35),
        (center_x - normal_x * 0.35, wall_start[1], center_z - normal_z * 0.35),
    )


def _vertical_connector_paths(
    connector: dict[str, object],
    rooms_by_id: dict[str, dict[str, object]],
) -> tuple[tuple[Point3, ...], ...]:
    """Derive continuous room-to-room stair paths from schema-2 geometry."""

    position = connector["position"]
    rotation = connector["rotation"]
    asset_contract = connector["assetContract"]
    landing_records = connector["landingPolygons"]
    if not isinstance(position, dict) or not isinstance(rotation, dict):
        raise TypeError("vertical connector position and rotation must be records")
    if not isinstance(asset_contract, dict) or not isinstance(landing_records, list):
        raise TypeError("vertical connector geometry is incomplete")

    base = _point(position)
    yaw = math.radians(float(rotation["y"]))
    forward = math.sin(yaw), math.cos(yaw)
    right = math.cos(yaw), -math.sin(yaw)
    flight_run = float(asset_contract["flightRun"])
    rise = float(asset_contract["rise"])
    landing_depth = (float(asset_contract["reservedLength"]) - flight_run) / 2.0
    half_width = float(asset_contract["reservedWidth"]) / 2.0
    room_anchor_offset = 0.3

    landings_by_floor: dict[str, dict[str, object]] = {}
    for raw_landing in landing_records:
        if not isinstance(raw_landing, dict):
            raise TypeError("vertical connector landing must be a record")
        landings_by_floor[str(raw_landing["floorId"])] = raw_landing

    def landing_and_room_anchors(*, upper: bool) -> tuple[Point3, tuple[Point3, ...]]:
        floor_key = "upperFloorId" if upper else "lowerFloorId"
        room_key = "upperRoomId" if upper else "lowerRoomId"
        floor_id = str(connector[floor_key])
        room_id = str(connector[room_key])
        raw_landing_polygon = landings_by_floor[floor_id]["polygon"]
        raw_room_polygon = rooms_by_id[room_id]["floorPolygon"]
        if not isinstance(raw_landing_polygon, list) or not isinstance(raw_room_polygon, list):
            raise TypeError("vertical connector landing and room polygons must be lists")
        center = _polygon_center(raw_landing_polygon)
        floor_direction = 1.0 if upper else -1.0
        candidates = (
            (
                center[0]
                + forward[0] * floor_direction * (landing_depth / 2.0 + room_anchor_offset),
                center[2]
                + forward[1] * floor_direction * (landing_depth / 2.0 + room_anchor_offset),
            ),
            (
                center[0] - right[0] * (half_width + room_anchor_offset),
                center[2] - right[1] * (half_width + room_anchor_offset),
            ),
            (
                center[0] + right[0] * (half_width + room_anchor_offset),
                center[2] + right[1] * (half_width + room_anchor_offset),
            ),
        )
        anchor = next(
            (
                (candidate[0], center[1], candidate[1])
                for candidate in candidates
                if _point_in_polygon(candidate, raw_room_polygon)
            ),
            None,
        )
        if anchor is None:
            raise ValueError(
                f"vertical connector {connector['id']!r} has no room-side landing egress"
            )
        return center, (anchor,)

    lower_landing, lower_rooms = landing_and_room_anchors(upper=False)
    upper_landing, upper_rooms = landing_and_room_anchors(upper=True)
    half_run = flight_run / 2.0
    lower_ramp = (
        base[0] - forward[0] * half_run,
        base[1],
        base[2] - forward[1] * half_run,
    )
    upper_ramp = (
        base[0] + forward[0] * half_run,
        base[1] + rise,
        base[2] + forward[1] * half_run,
    )
    lower_paths = tuple((lower_room, lower_landing, lower_ramp) for lower_room in lower_rooms)
    upper_paths = tuple((upper_ramp, upper_landing, upper_room) for upper_room in upper_rooms)
    vertical_path = ((lower_ramp, upper_ramp),)
    return lower_paths + vertical_path + upper_paths


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
        self._scene_links: tuple[tuple[Point3, ...], ...] = ()
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

        walls_by_id = {str(wall["id"]): wall for wall in house.get("walls", ())}
        door_paths = tuple(
            _door_path(door, walls_by_id)
            for door in house.get("doors", ())
            if door.get("room1") and door.get("room0") != door.get("room1")
        )
        if metadata["schema"] == "2.0.0":
            self._floor_heights = tuple(float(floor["baseY"]) for floor in house["floors"])
            rooms_by_id = {str(room["id"]): room for room in house["rooms"]}
            connector_paths = tuple(
                path
                for connector in house.get("verticalConnectors", ())
                for path in _vertical_connector_paths(connector, rooms_by_id)
            )
        else:
            room = house["rooms"][0]
            self._floor_heights = (float(room["floorPolygon"][0]["y"]),)
            connector_paths = ()
        self._scene_links = door_paths + connector_paths

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
        vertices = tuple(_point(point) for point in document["vertices"])
        indices = tuple(int(index) for index in document["indices"])
        areas = tuple(int(area) for area in document["areas"])
        adjacency_indices = tuple(int(index) for index in document["adjacency"])
        if len(indices) % 3 != 0:
            raise ValueError("runtime navmesh returned an incomplete triangle index list")
        triangles = tuple(
            (
                indices[offset],
                indices[offset + 1],
                indices[offset + 2],
            )
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
            (
                adjacency_indices[offset],
                adjacency_indices[offset + 1],
            )
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
        self._navmesh = NavMesh(
            agent_type_id=int(document["agentTypeId"]),
            vertices=vertices,
            triangles=triangles,
            areas=areas,
            adjacency=adjacency,
            links=self._scene_links,
        )

        if start_pose is None:
            triangle_id = self._random.randrange(len(triangles))
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
            "navmesh_vertex_count": len(self.navmesh.vertices),
            "navmesh_triangle_count": len(self.navmesh.triangles),
            "navmesh_adjacency_count": len(self.navmesh.adjacency),
            "navmesh_link_count": len(self.navmesh.links),
            "agent_y_offset": self._agent_y_offset,
        }

    def close(self) -> None:
        if not self._closed:
            self.simulator.close()
            self._closed = True


__all__ = ["AI2THORNavigationBackend", "AI2THORNavigationConfig"]
