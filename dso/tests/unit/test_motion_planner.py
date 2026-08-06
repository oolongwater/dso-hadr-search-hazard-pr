import inspect
import math
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from dso_hadr.graph.model import (
    TraversabilityEdge,
    TraversabilityMap,
    TraversabilitySource,
)
from dso_hadr.planner.motion.astar import astar_search, remove_immediate_backtracks
from dso_hadr.scenes.traversability import (
    direct_navmesh_traversability,
    select_region_traversability_point,
)
from dso_hadr.types.navigation import NavMesh, Point3


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


def test_region_selection_ignores_a_nearer_isolated_navmesh_island() -> None:
    points: tuple[Point3, ...] = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
    )
    traversability_map = _map(
        points,
        (
            _edge(0, 1, points[0:2]),
            _edge(1, 2, points[1:3]),
        ),
    )
    task = SimpleNamespace(
        graph=SimpleNamespace(
            regions=(
                SimpleNamespace(
                    id="room|2",
                    floor_id="floor|0",
                    semantic_region_value=1,
                ),
            ),
            traversability_map=traversability_map,
        ),
        floor_grids=(
            SimpleNamespace(
                floor_id="floor|0",
                floor_height=0.0,
                origin_xz=(0.0, 0.0),
                meters_per_pixel=1.0,
                traversable=np.ones((1, 3), dtype=np.bool_),
                semantic_regions=np.ones((1, 3), dtype=np.int32),
            ),
        ),
    )

    selected = select_region_traversability_point(
        task,
        "room|2",
        points[3],
        navigable_tolerance=0.18,
    )

    assert selected == points[0]


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


def test_direct_navmesh_merges_duplicate_vertices_and_connects_shared_edges() -> None:
    navmesh = NavMesh(
        agent_type_id=0,
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.00000001, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 0.0, 1.00000001),
        ),
        triangles=((0, 1, 2), (3, 4, 5)),
        areas=(0, 0),
        adjacency=((0, 1),),
        links=(),
    )

    traversability_map = direct_navmesh_traversability(navmesh, move_magnitude=0.2)
    path = astar_search(
        traversability_map,
        traversability_map.nodes[0],
        traversability_map.nodes[1],
    )

    assert len(traversability_map.nodes) == 2
    assert len(traversability_map.edges) == 1
    assert path.points[0] == traversability_map.nodes[0]
    assert path.points[-1] == traversability_map.nodes[1]
    assert (0.5, 0.0, 0.5) in path.points
    assert all(math.dist(a, b) <= 0.2 + 1e-12 for a, b in pairwise(path.points))


def test_direct_navmesh_does_not_infer_an_unexported_shared_edge() -> None:
    navmesh = NavMesh(
        agent_type_id=0,
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ),
        triangles=((0, 1, 2), (3, 4, 5)),
        areas=(0, 0),
        adjacency=(),
        links=(),
    )

    traversability_map = direct_navmesh_traversability(navmesh, move_magnitude=0.25)

    assert traversability_map.edges == ()
    with pytest.raises(ValueError, match="A\\* could not connect"):
        astar_search(
            traversability_map,
            traversability_map.nodes[0],
            traversability_map.nodes[1],
        )


def test_direct_navmesh_uses_exported_runtime_links() -> None:
    navmesh = NavMesh(
        agent_type_id=0,
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 0.0, 1.0),
        ),
        triangles=((0, 1, 2), (3, 4, 5)),
        areas=(0, 0),
        adjacency=(),
        links=(((0.2, 0.0, 0.2), (2.2, 0.0, 0.2)),),
    )

    traversability_map = direct_navmesh_traversability(navmesh, move_magnitude=0.25)
    path = astar_search(
        traversability_map,
        traversability_map.nodes[0],
        traversability_map.nodes[1],
    )

    assert len(traversability_map.edges) == 3
    assert traversability_map.edges[-1].portal is None
    assert (0.2, 0.0, 0.2) in path.points
    assert (2.2, 0.0, 0.2) in path.points


def test_direct_navmesh_links_join_at_one_shared_endpoint_node() -> None:
    shared = (2.2, 0.0, 0.2)
    navmesh = NavMesh(
        agent_type_id=0,
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (2.0, 0.0, 1.0),
            (4.0, 0.0, 0.0),
            (5.0, 0.0, 0.0),
            (4.0, 0.0, 1.0),
        ),
        triangles=((0, 1, 2), (3, 4, 5), (6, 7, 8)),
        areas=(0, 0, 0),
        adjacency=(),
        links=(
            ((0.2, 0.0, 0.2), shared),
            (shared, (4.2, 0.0, 0.2)),
        ),
    )

    traversability_map = direct_navmesh_traversability(navmesh, move_magnitude=0.25)
    path = astar_search(
        traversability_map,
        traversability_map.nodes[0],
        traversability_map.nodes[2],
    )

    assert path.points.count(shared) == 1
    assert traversability_map.nodes[1] not in path.points


def test_direct_navmesh_preserves_scene_link_geometry() -> None:
    navmesh = NavMesh(
        agent_type_id=0,
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (4.0, 3.0, 0.0),
            (5.0, 3.0, 0.0),
            (4.0, 3.0, 1.0),
        ),
        triangles=((0, 1, 2), (3, 4, 5)),
        areas=(0, 0),
        adjacency=(),
        links=(
            (
                (0.2, 0.0, 0.2),
                (1.0, 0.0, 0.2),
                (4.0, 3.0, 0.2),
                (4.2, 3.0, 0.2),
            ),
        ),
    )

    traversability_map = direct_navmesh_traversability(navmesh, move_magnitude=0.25)
    path = astar_search(
        traversability_map,
        traversability_map.nodes[0],
        traversability_map.nodes[1],
    )

    assert (1.0, 0.0, 0.2) in path.points
    assert (4.0, 3.0, 0.2) in path.points
    assert all(math.dist(a, b) <= 0.25 + 1e-12 for a, b in pairwise(path.points))


def test_direct_navmesh_preserves_sloped_triangle_geometry() -> None:
    navmesh = NavMesh(
        agent_type_id=0,
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.2, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.2, 0.0),
            (1.0, 0.2, 1.0),
            (0.0, 0.0, 1.0),
        ),
        triangles=((0, 1, 2), (3, 4, 5)),
        areas=(0, 0),
        adjacency=((0, 1),),
        links=(),
    )

    traversability_map = direct_navmesh_traversability(navmesh, move_magnitude=0.1)
    path = astar_search(
        traversability_map,
        traversability_map.nodes[0],
        traversability_map.nodes[1],
    )

    assert path.points[0][1] < path.points[-1][1]
    assert all(math.dist(a, b) <= 0.1 + 1e-12 for a, b in pairwise(path.points))


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
