from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dso_hadr.controller.waypoint_follower import WaypointFollower
from dso_hadr.scenes.scene_graph import scene_agent_pose, scene_navigation_map
from dso_hadr.simulator.ai2thor_backend import (
    AI2THORNavigationBackend,
    AI2THORNavigationConfig,
    parse_navmesh,
)
from dso_hadr.types.navigation import NavigationAction, Observation, Pose
from dso_hadr.utils.coordinates import (
    native_yaw_to_navigation,
    navigation_yaw_to_native,
)


class FakeEvent:
    def __init__(
        self,
        rotation: float = 180.0,
        position: dict[str, float] | None = None,
    ) -> None:
        self.frame = np.zeros((4, 6, 3), dtype=np.uint8)
        self.depth_frame = np.ones((4, 6), dtype=np.float32)
        self.metadata = {
            "lastActionSuccess": True,
            "errorMessage": "",
            "agent": {
                "position": dict(position or {"x": 1.0, "y": 0.9, "z": 2.0}),
                "rotation": {"x": 0.0, "y": rotation, "z": 0.0},
            },
        }

    def __bool__(self) -> bool:
        return bool(self.metadata["lastActionSuccess"])


class FakeSimulator:
    def __init__(self) -> None:
        self.actions: list[dict[str, object]] = []
        self.loaded_paths: list[Path] = []
        self.position = {"x": 1.0, "y": 0.9, "z": 2.0}
        self.rotation = 180.0

    def load_scene(self, scene_path: Path) -> FakeEvent:
        self.loaded_paths.append(scene_path)
        return FakeEvent(self.rotation, self.position)

    def step(self, action: dict[str, object]) -> FakeEvent:
        self.actions.append(action)
        action_name = action["action"]
        if action_name == "TeleportFull":
            position = action["position"]
            rotation = action["rotation"]
            assert isinstance(position, dict)
            assert isinstance(rotation, dict)
            self.position = {
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position["z"]),
            }
            self.rotation = float(rotation["y"])
        elif action_name == "MoveAhead":
            magnitude = float(action["moveMagnitude"])
            radians = math.radians(self.rotation)
            self.position["x"] += math.sin(radians) * magnitude
            self.position["z"] += math.cos(radians) * magnitude
            if "targetY" in action:
                self.position["y"] = float(action["targetY"])

        event = FakeEvent(self.rotation, self.position)
        if action_name == "DSOHADRGetNavMeshTriangulation":
            event.metadata["actionReturn"] = {
                "agentTypeId": 0,
                "agentRadius": 0.28,
                "movementRadius": 0.28,
                "vertices": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 1.0, "y": 0.0, "z": 0.0},
                    {"x": 0.0, "y": 0.0, "z": 1.0},
                ],
                "indices": [0, 1, 2],
                "areas": [0],
                "adjacency": [],
            }
        return event

    def close(self) -> None:
        pass


def _backend(simulator: FakeSimulator) -> AI2THORNavigationBackend:
    backend = AI2THORNavigationBackend(
        simulator,
        {"scene": Path("scene.json")},
        AI2THORNavigationConfig(
            move_magnitude=0.25,
            rotation_degrees=15.0,
        ),
    )
    backend._event = FakeEvent()
    return backend


def test_ai2thor_yaw_round_trip() -> None:
    assert native_yaw_to_navigation(180.0) == 0.0
    assert native_yaw_to_navigation(270.0) == math.pi / 2.0
    assert navigation_yaw_to_native(math.pi / 2.0) == 270.0


def test_scene_agent_pose_uses_the_requested_room_center(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema": "1.0.0",
                    "agent": {
                        "position": {"x": 99.0, "y": 0.9, "z": 99.0},
                        "rotation": {"x": 0.0, "y": 180.0, "z": 0.0},
                    },
                },
                "rooms": [
                    {
                        "id": "room|2",
                        "floorPolygon": [
                            {"x": 4.0, "y": 3.0, "z": 6.0},
                            {"x": 8.0, "y": 3.0, "z": 6.0},
                            {"x": 8.0, "y": 3.0, "z": 10.0},
                            {"x": 4.0, "y": 3.0, "z": 10.0},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    pose = scene_agent_pose(scene_path, "room|2")

    assert pose == Pose(6.0, 3.0, 8.0, 0.0)


def test_navigation_turns_are_translated_to_matching_ai2thor_turns() -> None:
    simulator = FakeSimulator()
    backend = _backend(simulator)

    backend.step(NavigationAction.TURN_LEFT)
    backend.step(NavigationAction.TURN_RIGHT)

    assert simulator.actions == [
        {"action": "RotateRight", "degrees": 15.0},
        {"action": "RotateLeft", "degrees": 15.0},
    ]


def test_navigation_backend_can_apply_exact_planner_turns() -> None:
    simulator = FakeSimulator()
    backend = _backend(simulator)

    backend.rotate(math.radians(11.25))
    backend.rotate(math.radians(-7.5))

    assert simulator.actions[0] == {"action": "RotateRight", "degrees": 11.25}
    assert simulator.actions[1]["action"] == "RotateLeft"
    assert math.isclose(float(simulator.actions[1]["degrees"]), 7.5)


def test_backend_reset_exports_runtime_triangulation_without_sampling(
    tmp_path: Path,
) -> None:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "schema": "1.0.0",
                    "agent": {"horizon": 0.0, "standing": True},
                },
                "rooms": [
                    {
                        "floorPolygon": [
                            {"x": 0.0, "y": 0.0, "z": 0.0},
                            {"x": 1.0, "y": 0.0, "z": 0.0},
                            {"x": 0.0, "y": 0.0, "z": 1.0},
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    simulator = FakeSimulator()
    backend = AI2THORNavigationBackend(
        simulator,
        {"scene": scene_path},
        AI2THORNavigationConfig(move_magnitude=0.25, rotation_degrees=15.0),
    )

    observation = backend.reset("scene", Pose(0.2, 0.0, 0.2, 0.0), seed=0)

    assert simulator.loaded_paths == [scene_path.resolve()]
    assert backend.navmesh.agent_type_id == 0
    assert backend.navmesh.agent_radius == 0.28
    assert backend.navmesh.movement_radius == 0.28
    assert backend.navmesh.vertices == (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert backend.navmesh.triangles == ((0, 1, 2),)
    assert backend.navmesh.areas == (0,)
    assert backend.navmesh.adjacency == ()
    assert observation.pose.position == (1.0 / 3.0, 0.0, 1.0 / 3.0)
    assert simulator.actions[0] == {
        "action": "DSOHADRGetNavMeshTriangulation",
        "renderImage": False,
    }
    action_names = {str(action["action"]) for action in simulator.actions}
    assert "GetReachablePositions" not in action_names
    assert "GetShortestPathToPoint" not in action_names
    assert backend.get_scene_metadata()["navmesh_triangle_count"] == 1


def test_navmesh_export_rejects_clearance_smaller_than_the_movement_capsule() -> None:
    document = {
        "agentTypeId": 0,
        "agentRadius": 0.2,
        "movementRadius": 0.28,
        "vertices": [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 1.0, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": 1.0},
        ],
        "indices": [0, 1, 2],
        "areas": [0],
        "adjacency": [],
    }

    try:
        parse_navmesh(document)
    except ValueError as error:
        assert "0.2 < 0.28" in str(error)
    else:
        raise AssertionError("undersized navmesh clearance was accepted")


def test_waypoint_follower_skips_height_only_intermediate_points() -> None:
    class Backend:
        def __init__(self) -> None:
            self.pose = Pose(0.0, 0.0, 0.0, -math.pi / 2.0)
            self.move_targets: list[tuple[float, float | None]] = []

        def observation(self) -> Observation:
            return Observation(
                rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                depth=np.zeros((1, 1), dtype=np.float32),
                pose=self.pose,
                collision=False,
            )

        def get_agent_pose(self) -> Pose:
            return self.pose

        def move_forward(self, distance: float, target_y: float | None = None) -> Observation:
            self.move_targets.append((distance, target_y))
            height = self.pose.y if target_y is None else target_y
            self.pose = Pose(1.0, height, 0.0, -math.pi / 2.0)
            return self.observation()

        def rotate(self, _angle_radians: float) -> Observation:
            raise AssertionError("height-only waypoint should not require rotation")

        def step(self, action: NavigationAction) -> Observation:
            assert action is NavigationAction.STOP
            return self.observation()

    backend = Backend()
    result = WaypointFollower(
        waypoint_tolerance=0.12,
        heading_tolerance_degrees=0.1,
        rotation_step_degrees=15.0,
    ).follow(
        backend,
        ((0.0, 0.0, 0.0), (0.0, 0.2, 0.0), (1.0, 0.2, 0.0)),
        success_distance=0.12,
        max_steps=4,
    )

    assert result.success
    assert result.final_distance == 0.0
    assert backend.move_targets == [(1.0, 0.2)]


def test_waypoint_follower_does_not_accept_the_wrong_floor() -> None:
    class Backend:
        def get_agent_pose(self) -> Pose:
            return Pose(0.0, 0.0, 0.0, 0.0)

    result = WaypointFollower(
        waypoint_tolerance=0.001,
        heading_tolerance_degrees=0.1,
        rotation_step_degrees=15.0,
    ).follow(
        Backend(),
        ((0.0, 3.0, 0.0),),
        success_distance=0.12,
        max_steps=0,
    )

    assert not result.success
    assert result.final_distance == 3.0


def test_waypoint_follower_splits_a_right_angle_into_configured_steps() -> None:
    class Backend:
        def __init__(self) -> None:
            self.pose = Pose(0.0, 0.0, 0.0, 0.0)
            self.rotations: list[float] = []

        def observation(self) -> Observation:
            return Observation(
                rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                depth=np.zeros((1, 1), dtype=np.float32),
                pose=self.pose,
                collision=False,
            )

        def get_agent_pose(self) -> Pose:
            return self.pose

        def rotate(self, angle_radians: float) -> Observation:
            self.rotations.append(angle_radians)
            self.pose = Pose(self.pose.x, self.pose.y, self.pose.z, self.pose.yaw + angle_radians)
            return self.observation()

        def move_forward(self, _distance: float, target_y: float | None = None) -> Observation:
            self.pose = Pose(1.0, self.pose.y if target_y is None else target_y, 0.0, self.pose.yaw)
            return self.observation()

        def step(self, action: NavigationAction) -> Observation:
            assert action is NavigationAction.STOP
            return self.observation()

    backend = Backend()
    result = WaypointFollower(
        waypoint_tolerance=0.001,
        heading_tolerance_degrees=0.1,
        rotation_step_degrees=15.0,
    ).follow(
        backend,
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        success_distance=0.12,
        max_steps=10,
    )

    assert result.success
    assert result.actions[:6] == ("turn_right",) * 6
    assert result.actions[6:] == ("move_forward", "stop")
    assert len(backend.rotations) == 6
    assert all(math.isclose(angle, -math.pi / 12.0) for angle in backend.rotations)


def test_scene_navigation_map_uses_physical_floor_surfaces(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    scene_path.write_text(
        json.dumps(
            {
                "floors": [
                    {
                        "floorSurfaces": [
                            {
                                "id": "surface|1",
                                "floorId": "floor|1",
                                "roomId": "room|2",
                                "polygon": [
                                    {"x": 1.0, "y": 3.0, "z": 2.0},
                                    {"x": 2.0, "y": 3.0, "z": 2.0},
                                    {"x": 2.0, "y": 3.0, "z": 4.0},
                                    {"x": 1.0, "y": 3.0, "z": 4.0},
                                ],
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    navigation_map = scene_navigation_map(scene_path)

    assert navigation_map == {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "kind": "logical_walkable",
                    "floorId": "floor|1",
                    "roomId": "room|2",
                    "surfaceId": "surface|1",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [1.0, 2.0],
                            [2.0, 2.0],
                            [2.0, 4.0],
                            [1.0, 4.0],
                            [1.0, 2.0],
                        ]
                    ],
                },
            }
        ],
    }
