import pytest

from dso_hadr.graph.model import (
    FloorNode,
    GraphEvidence,
    RegionKind,
    RegionNode,
    SceneGraph,
    SceneGraphTask,
    TraversabilityMap,
    TraversabilitySource,
)
from dso_hadr.scenes import long_horizon
from dso_hadr.types.navigation import Pose, ShortestPath


def _task(region_floors: tuple[str, ...]) -> SceneGraphTask:
    floor_ids = tuple(dict.fromkeys(region_floors))
    floors = tuple(
        FloorNode(
            id=floor_id,
            label=floor_id,
            scene_model="scene.json",
            level_index=index,
            evidence=GraphEvidence.DATASET_SEMANTICS,
        )
        for index, floor_id in enumerate(floor_ids)
    )
    regions = tuple(
        RegionNode(
            id=f"room|{index}",
            label=f"room|{index}",
            category="room",
            kind=RegionKind.ROOM,
            floor_id=floor_id,
            navigation_pose=Pose(
                float(index),
                float(floors.index(next(f for f in floors if f.id == floor_id))),
                0.0,
                0.0,
            ),
            bounds_xz=(float(index), 0.0, float(index + 1), 1.0),
            semantic_region_value=index + 1,
            evidence=GraphEvidence.DATASET_SEMANTICS,
        )
        for index, floor_id in enumerate(region_floors)
    )
    return SceneGraphTask(
        graph=SceneGraph(
            scene_id="scene",
            floors=floors,
            regions=regions,
            containment_edges=(),
            connectivity_edges=(),
            traversability_map=TraversabilityMap(
                source=TraversabilitySource.AI2THOR_NAVMESH_GROUND_TRUTH,
                nodes=(),
                edges=(),
            ),
        ),
        floor_grids=(),
        start_region_id=regions[0].id,
        goal_region_id=regions[-1].id,
    )


def _install_exact_pair_distances(
    monkeypatch: pytest.MonkeyPatch,
    distances: dict[tuple[int, int], float],
) -> list[tuple[int, int]]:
    evaluated: list[tuple[int, int]] = []
    monkeypatch.setattr(
        long_horizon,
        "select_region_point",
        lambda _task, _region_id, target, _tolerance: target,
    )
    monkeypatch.setattr(long_horizon, "dijkstra_search", lambda *_args: None)

    def fake_astar(
        _map: TraversabilityMap, start: tuple[float, float, float], goal: tuple[float, float, float]
    ) -> ShortestPath:
        pair = tuple(sorted((int(start[0]), int(goal[0]))))
        evaluated.append(pair)
        return ShortestPath(points=(start, goal), geodesic_distance=distances[pair])

    monkeypatch.setattr(long_horizon, "astar_search", fake_astar)
    return evaluated


def test_select_episode_chooses_the_maximum_of_every_single_floor_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(("floor|0", "floor|0", "floor|0"))
    evaluated = _install_exact_pair_distances(
        monkeypatch,
        {(0, 1): 12.0, (0, 2): 20.0, (1, 2): 14.0},
    )

    episode = long_horizon.select_episode(
        task,
        navigable_tolerance=0.18,
        minimum_geodesic_distance=10.0,
    )

    assert sorted(evaluated) == [(0, 1), (0, 2), (1, 2)]
    assert episode.start_room_id == "room|0"
    assert episode.goal_room_id == "room|2"
    assert episode.geodesic_distance == 20.0
    assert not episode.crosses_floors


def test_select_episode_multifloor_considers_only_cross_floor_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(("floor|0", "floor|0", "floor|1", "floor|1"))
    evaluated = _install_exact_pair_distances(
        monkeypatch,
        {(0, 2): 11.0, (0, 3): 15.0, (1, 2): 16.0, (1, 3): 12.0},
    )

    episode = long_horizon.select_episode(
        task,
        navigable_tolerance=0.18,
        minimum_geodesic_distance=10.0,
    )

    assert sorted(evaluated) == [(0, 2), (0, 3), (1, 2), (1, 3)]
    assert episode.start_room_id == "room|1"
    assert episode.goal_room_id == "room|2"
    assert episode.geodesic_distance == 16.0
    assert episode.crosses_floors
