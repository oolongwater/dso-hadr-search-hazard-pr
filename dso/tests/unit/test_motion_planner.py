import inspect
import math
from itertools import pairwise

import pytest

from dso_hadr.graph.model import (
    TraversabilityEdge,
    TraversabilityMap,
    TraversabilitySource,
)
from dso_hadr.planner.motion.astar import astar_search, remove_immediate_backtracks
from dso_hadr.scenes.traversability import (
    extract_ground_truth_traversability,
    ground_truth_traversability_nodes,
)
from dso_hadr.types.navigation import Point3, ShortestPath


def _edge(
    node_a: int,
    node_b: int,
    points: tuple[Point3, ...],
) -> TraversabilityEdge:
    return TraversabilityEdge(
        node_a=node_a,
        node_b=node_b,
        path=points,
        cost=sum(math.dist(start, goal) for start, goal in pairwise(points)),
    )


def _map(
    points: tuple[Point3, ...],
    edges: tuple[TraversabilityEdge, ...],
) -> TraversabilityMap:
    return TraversabilityMap(
        source=TraversabilitySource.AI2THOR_NAVMESH_GROUND_TRUTH,
        nodes=points,
        edges=edges,
    )


def _straight_navmesh_path(
    start: Point3,
    goal: Point3,
    max_path_length: float,
) -> ShortestPath | None:
    distance = math.dist(start, goal)
    return (
        ShortestPath(points=(start, goal), geodesic_distance=distance)
        if distance <= max_path_length
        else None
    )


def _extract(
    points: tuple[Point3, ...],
    start: Point3,
    goal: Point3,
    query,
    *,
    max_vertical_step: float = 0.3,
    max_edge_length_ratio: float = 1.6,
    max_link_distance: float = 0.1,
    max_transition_slope_ratio: float = 0.8,
) -> TraversabilityMap:
    return extract_ground_truth_traversability(
        ground_truth_traversability_nodes(points),
        (start, goal),
        grid_size=1.0,
        max_vertical_step=max_vertical_step,
        max_edge_length_ratio=max_edge_length_ratio,
        max_link_path_length_ratio=2.0,
        max_link_distance=max_link_distance,
        max_transition_slope_ratio=max_transition_slope_ratio,
        transition_height_tolerance=0.05,
        link_candidate_count=2,
        query_navmesh_path=query,
    )


def test_astar_accepts_only_a_traversability_map_and_points() -> None:
    assert tuple(inspect.signature(astar_search).parameters) == (
        "traversability_map",
        "start",
        "goal",
    )


def test_astar_selects_the_lowest_cost_map_route() -> None:
    points: tuple[Point3, ...] = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    )
    traversability_map = _map(
        points,
        (
            _edge(0, 1, points[:2]),
            _edge(1, 2, points[1:]),
            TraversabilityEdge(0, 2, (points[0], points[2]), 3.0),
        ),
    )

    path = astar_search(traversability_map, points[0], points[2])

    assert path.points == points
    assert path.geodesic_distance == 2.0


def test_ground_truth_extraction_rejects_an_invalid_edge_before_astar() -> None:
    points: tuple[Point3, ...] = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.0, 0.0, 1.0),
    )
    blocked = {
        (points[1], points[2]),
        (points[2], points[1]),
    }

    def query(
        start: Point3,
        goal: Point3,
        max_path_length: float,
    ) -> ShortestPath | None:
        return (
            None
            if (start, goal) in blocked
            else _straight_navmesh_path(start, goal, max_path_length)
        )

    traversability_map = _extract(points, points[0], points[2], query)
    path = astar_search(traversability_map, points[0], points[2])

    assert path.points[0] == points[0]
    assert path.points[-1] == points[2]
    assert all(edge not in blocked for edge in zip(path.points, path.points[1:]))


def test_ground_truth_extraction_discovers_a_bridge_after_validation_splits_a_component() -> None:
    points: tuple[Point3, ...] = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    )
    queries: list[tuple[Point3, Point3]] = []

    def query(
        start: Point3,
        goal: Point3,
        max_path_length: float,
    ) -> ShortestPath | None:
        queries.append((start, goal))
        return (
            None
            if {start, goal} == {points[1], points[2]}
            else _straight_navmesh_path(start, goal, max_path_length)
        )

    traversability_map = _extract(
        points,
        points[0],
        points[2],
        query,
        max_edge_length_ratio=1.1,
        max_link_distance=2.1,
    )

    assert (points[0], points[2]) in queries
    assert astar_search(traversability_map, points[0], points[2]).points == (
        points[0],
        points[2],
    )


def test_astar_keeps_levels_distinct_and_follows_slope_edges() -> None:
    points: tuple[Point3, ...] = (
        (0.0, 0.0, 0.0),
        (1.0, 0.2, 0.0),
        (2.0, 0.4, 0.0),
        (0.0, 3.0, 0.0),
        (1.0, 3.0, 0.0),
        (2.0, 3.0, 0.0),
    )
    traversability_map = _map(
        points,
        (
            _edge(0, 1, points[0:2]),
            _edge(1, 2, points[1:3]),
            _edge(3, 4, points[3:5]),
            _edge(4, 5, points[4:6]),
        ),
    )

    path = astar_search(traversability_map, points[0], points[2])

    assert path.points == points[:3]
    assert math.isclose(path.geodesic_distance, 2.0 * math.sqrt(1.04))


def test_ground_truth_extraction_connects_a_two_cell_sample_gap() -> None:
    points: tuple[Point3, ...] = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))

    traversability_map = _extract(
        points,
        points[0],
        points[1],
        _straight_navmesh_path,
        max_edge_length_ratio=2.1,
    )

    assert astar_search(traversability_map, points[0], points[1]).points == points


def test_astar_expands_ground_truth_edge_geometry() -> None:
    points: tuple[Point3, ...] = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 1.0, 0.0),
        (3.0, 1.0, 0.0),
    )

    def query(
        start: Point3,
        goal: Point3,
        max_path_length: float,
    ) -> ShortestPath | None:
        if start == points[1] and goal == points[2]:
            return ShortestPath(
                points=(start, (1.5, 0.5, 0.0), goal),
                geodesic_distance=math.sqrt(2.0),
            )
        return _straight_navmesh_path(start, goal, max_path_length)

    traversability_map = _extract(
        points,
        points[0],
        points[-1],
        query,
        max_link_distance=1.5,
        max_transition_slope_ratio=1.1,
    )

    assert astar_search(traversability_map, points[0], points[-1]).points == (
        points[0],
        points[1],
        (1.5, 0.5, 0.0),
        points[2],
        points[3],
    )


def test_ground_truth_extraction_rejects_vertical_transition_geometry() -> None:
    points: tuple[Point3, ...] = (
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
    )

    def vertical_path(
        start: Point3,
        goal: Point3,
        _max_path_length: float,
    ) -> ShortestPath:
        return ShortestPath(
            points=(start, (0.0, 1.0, 0.0), goal),
            geodesic_distance=2.0,
        )

    with pytest.raises(ValueError, match="ground-truth traversability extraction"):
        _extract(
            points,
            points[0],
            points[1],
            vertical_path,
            max_link_distance=1.5,
            max_transition_slope_ratio=1.1,
        )


def test_astar_rejects_a_disconnected_map() -> None:
    points: tuple[Point3, ...] = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="A\\* could not connect"):
        astar_search(_map(points, ()), points[0], points[1])


def test_astar_removes_an_exact_immediate_backtrack() -> None:
    points: tuple[Point3, ...] = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    traversability_map = _map(
        points,
        (
            TraversabilityEdge(
                node_a=0,
                node_b=1,
                path=(points[0], (1.0, 0.0, 0.0), points[0], points[1]),
                cost=4.0,
            ),
        ),
    )

    path = astar_search(traversability_map, points[0], points[1])

    assert path.points == points
    assert path.geodesic_distance == 2.0


def test_remove_immediate_backtracks_across_segment_boundaries() -> None:
    first = (0.0, 0.0, 0.0)
    detour = (0.0, 0.0, 0.01)
    target = (1.0, 0.0, 0.0)

    assert remove_immediate_backtracks((first, detour, first, target)) == (
        first,
        target,
    )
