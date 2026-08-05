from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from dso_hadr.controller.waypoint_follower import WaypointFollower
from dso_hadr.utils.coordinates import (
    navigation_yaw_to_native,
    native_yaw_to_navigation,
)
from dso_hadr.types.navigation import NavigationAction, Observation, Pose
from dso_hadr.scenes.scene_graph import scene_navigation_map
from dso_hadr.simulator.ai2thor_backend import (
    AI2THORNavigationBackend,
    AI2THORNavigationConfig,
)


class FakeEvent:
    def __init__(self, rotation: float = 180.0) -> None:
        self.frame = np.zeros((4, 6, 3), dtype=np.uint8)
        self.depth_frame = np.ones((4, 6), dtype=np.float32)
        self.metadata = {
            "lastActionSuccess": True,
            "errorMessage": "",
            "agent": {
                "position": {"x": 1.0, "y": 0.9, "z": 2.0},
                "rotation": {"x": 0.0, "y": rotation, "z": 0.0},
            },
        }

    def __bool__(self) -> bool:
        return bool(self.metadata["lastActionSuccess"])


class FakeSimulator:
    def __init__(self, *, fail_move: bool = False) -> None:
        self.actions: list[dict[str, object]] = []
        self.fail_move = fail_move

    def step(self, action: dict[str, object]) -> FakeEvent:
        self.actions.append(action)
        event = FakeEvent()
        if action["action"] == "GetShortestPathToPoint":
            event.metadata["actionReturn"] = {
                "corners": [
                    {"x": 1.0, "y": 0.0, "z": 0.0},
                ]
            }
        elif action["action"] == "MoveAhead" and self.fail_move:
            event.metadata["lastActionSuccess"] = False
            event.metadata["errorMessage"] = "blocked"
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
            reachable_grid_size=0.25,
            path_allowed_error=0.2,
            navigable_tolerance=0.18,
        ),
    )
    backend._event = FakeEvent()
    return backend


def test_ai2thor_yaw_round_trip() -> None:
    assert native_yaw_to_navigation(180.0) == 0.0
    assert native_yaw_to_navigation(270.0) == math.pi / 2.0
    assert navigation_yaw_to_native(math.pi / 2.0) == 270.0


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


def test_navmesh_path_query_includes_requested_endpoints() -> None:
    simulator = FakeSimulator()
    backend = _backend(simulator)

    path = backend.get_navmesh_path(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        max_path_length=2.0,
    )

    assert path is not None
    assert path.points == ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    assert path.geodesic_distance == 2.0
    assert any(
        action["action"] == "TeleportFull" and action.get("forceAction") is True
        for action in simulator.actions
    )


def test_navmesh_path_query_rejects_a_physically_blocked_edge() -> None:
    backend = _backend(FakeSimulator(fail_move=True))

    path = backend.get_navmesh_path(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        max_path_length=2.0,
    )

    assert path is None
    assert backend.get_agent_pose() == Pose(1.0, 0.9, 2.0, 0.0)


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
