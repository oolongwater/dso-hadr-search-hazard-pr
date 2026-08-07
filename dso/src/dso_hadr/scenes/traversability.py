"""Build traversability directly from the simulator's runtime navmesh."""

from __future__ import annotations

import itertools
import math

from dso_hadr.graph.model import (
    SceneGraphTask,
    TraversabilityEdge,
    TraversabilityMap,
    TraversabilitySource,
)
from dso_hadr.types.navigation import NavMesh, Point3

_VERTEX_DIGITS = 5


def _path_length(points: tuple[Point3, ...]) -> float:
    return sum(math.dist(source, target) for source, target in itertools.pairwise(points))


def _dense_path(points: tuple[Point3, ...], max_segment_length: float) -> tuple[Point3, ...]:
    if max_segment_length <= 0.0:
        raise ValueError("max segment length must be positive")
    dense: list[Point3] = [points[0]]
    for source, target in itertools.pairwise(points):
        distance = math.dist(source, target)
        steps = max(1, math.ceil(distance / max_segment_length))
        for step in range(1, steps + 1):
            fraction = step / steps
            point = (
                source[0] + (target[0] - source[0]) * fraction,
                source[1] + (target[1] - source[1]) * fraction,
                source[2] + (target[2] - source[2]) * fraction,
            )
            if point != dense[-1]:
                dense.append(point)
    return tuple(dense)


def _canonical_navmesh(
    navmesh: NavMesh,
) -> tuple[tuple[Point3, ...], tuple[tuple[int, int, int], ...]]:
    vertices: list[Point3] = []
    vertex_by_position: dict[Point3, int] = {}
    raw_to_canonical: list[int] = []
    for raw_vertex in navmesh.vertices:
        position: Point3 = (
            round(raw_vertex[0], _VERTEX_DIGITS),
            round(raw_vertex[1], _VERTEX_DIGITS),
            round(raw_vertex[2], _VERTEX_DIGITS),
        )
        vertex = vertex_by_position.get(position)
        if vertex is None:
            vertex = len(vertices)
            vertex_by_position[position] = vertex
            vertices.append(position)
        raw_to_canonical.append(vertex)

    triangles: list[tuple[int, int, int]] = []
    for raw_triangle in navmesh.triangles:
        triangle = (
            raw_to_canonical[raw_triangle[0]],
            raw_to_canonical[raw_triangle[1]],
            raw_to_canonical[raw_triangle[2]],
        )
        if len(set(triangle)) != 3:
            raise ValueError("runtime navmesh contains a degenerate triangle")
        triangles.append(triangle)
    if not triangles:
        raise ValueError("runtime navmesh contains no triangles")
    return tuple(vertices), tuple(triangles)


def build_traversability(
    navmesh: NavMesh,
    *,
    move_magnitude: float,
) -> TraversabilityMap:
    """Convert one runtime triangulation into a dense triangle-adjacency graph."""

    vertices, triangles = _canonical_navmesh(navmesh)
    triangle_nodes = tuple(
        (
            sum(vertices[vertex][0] for vertex in triangle) / 3.0,
            sum(vertices[vertex][1] for vertex in triangle) / 3.0,
            sum(vertices[vertex][2] for vertex in triangle) / 3.0,
        )
        for triangle in triangles
    )

    edges: list[TraversabilityEdge] = []
    for raw_node_a, raw_node_b in navmesh.adjacency:
        node_a, node_b = min(raw_node_a, raw_node_b), max(raw_node_a, raw_node_b)
        if node_a < 0 or node_b >= len(triangles) or node_a == node_b:
            raise ValueError("runtime navmesh adjacency references an invalid triangle")
        shared_edge = tuple(sorted(set(triangles[node_a]) & set(triangles[node_b])))
        if len(shared_edge) != 2:
            raise ValueError("adjacent runtime navmesh triangles do not share one edge")
        first_vertex, second_vertex = (vertices[index] for index in shared_edge)
        midpoint = (
            (first_vertex[0] + second_vertex[0]) * 0.5,
            (first_vertex[1] + second_vertex[1]) * 0.5,
            (first_vertex[2] + second_vertex[2]) * 0.5,
        )
        path = _dense_path(
            (triangle_nodes[node_a], midpoint, triangle_nodes[node_b]),
            move_magnitude,
        )
        edges.append(
            TraversabilityEdge(
                node_a=node_a,
                node_b=node_b,
                path=path,
                cost=_path_length(path),
                portal=(first_vertex, second_vertex),
            )
        )

    traversability_map = TraversabilityMap(
        source=TraversabilitySource.AI2THOR_NAVMESH_GROUND_TRUTH,
        nodes=triangle_nodes,
        edges=tuple(edges),
    )
    _require_connected(traversability_map)
    return traversability_map


def _connected_components(
    traversability_map: TraversabilityMap,
) -> tuple[frozenset[int], ...]:
    """Return every exact connected component in the exported triangle graph."""

    adjacency: list[set[int]] = [set() for _node in traversability_map.nodes]
    for edge in traversability_map.edges:
        adjacency[edge.node_a].add(edge.node_b)
        adjacency[edge.node_b].add(edge.node_a)

    unseen = set(range(len(traversability_map.nodes)))
    components: list[frozenset[int]] = []
    while unseen:
        root = min(unseen)
        component = {root}
        frontier = [root]
        unseen.remove(root)
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current]:
                if neighbor not in unseen:
                    continue
                unseen.remove(neighbor)
                component.add(neighbor)
                frontier.append(neighbor)
        components.append(frozenset(component))
    return tuple(sorted(components, key=lambda component: (-len(component), min(component))))


def _require_connected(traversability_map: TraversabilityMap) -> None:
    """Reject a runtime navmesh unless every exported triangle is connected."""

    components = _connected_components(traversability_map)
    if len(components) != 1:
        sizes = [len(component) for component in components]
        raise ValueError(
            "runtime navmesh triangulation is disconnected: "
            f"{len(components)} components with triangle counts {sizes}"
        )


def _region_nodes(
    task: SceneGraphTask,
    region_id: str,
    navigable_tolerance: float,
) -> tuple[int, ...]:
    region = next(region for region in task.graph.regions if region.id == region_id)
    grid = next(grid for grid in task.floor_grids if grid.floor_id == region.floor_id)
    candidates: list[int] = []
    for node, point in enumerate(task.graph.traversability_map.nodes):
        if abs(point[1] - grid.floor_height) > navigable_tolerance:
            continue
        row = math.floor((point[2] - grid.origin_xz[1]) / grid.meters_per_pixel + 0.5 + 1e-9)
        column = math.floor((point[0] - grid.origin_xz[0]) / grid.meters_per_pixel + 0.5 + 1e-9)
        if row < 0 or row >= grid.traversable.shape[0]:
            continue
        if column < 0 or column >= grid.traversable.shape[1]:
            continue
        if grid.semantic_regions[row, column] == region.semantic_region_value:
            candidates.append(node)
    if not candidates:
        raise ValueError(f"runtime navmesh has no triangle centroid in region {region_id!r}")
    return tuple(candidates)


def select_region_point(
    task: SceneGraphTask,
    region_id: str,
    target: Point3,
    navigable_tolerance: float,
) -> Point3:
    """Select the navmesh point nearest a target within one semantic region."""

    candidates = _region_nodes(task, region_id, navigable_tolerance)
    node = min(
        candidates,
        key=lambda candidate: (
            math.dist(task.graph.traversability_map.nodes[candidate], target),
            candidate,
        ),
    )
    return task.graph.traversability_map.nodes[node]


__all__ = [
    "build_traversability",
    "select_region_point",
]
