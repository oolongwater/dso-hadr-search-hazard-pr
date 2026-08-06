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
_LINK_TRIANGLE_TOLERANCE = 0.6


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


def _subtract(first: Point3, second: Point3) -> Point3:
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def _dot(first: Point3, second: Point3) -> float:
    return sum(a * b for a, b in zip(first, second, strict=True))


def _scaled_add(origin: Point3, direction: Point3, scale: float) -> Point3:
    return (
        origin[0] + direction[0] * scale,
        origin[1] + direction[1] * scale,
        origin[2] + direction[2] * scale,
    )


def _closest_point_on_triangle(
    point: Point3,
    vertex_a: Point3,
    vertex_b: Point3,
    vertex_c: Point3,
) -> Point3:
    edge_ab = _subtract(vertex_b, vertex_a)
    edge_ac = _subtract(vertex_c, vertex_a)
    offset_a = _subtract(point, vertex_a)
    dot_ab_a = _dot(edge_ab, offset_a)
    dot_ac_a = _dot(edge_ac, offset_a)
    if dot_ab_a <= 0.0 and dot_ac_a <= 0.0:
        return vertex_a

    offset_b = _subtract(point, vertex_b)
    dot_ab_b = _dot(edge_ab, offset_b)
    dot_ac_b = _dot(edge_ac, offset_b)
    if dot_ab_b >= 0.0 and dot_ac_b <= dot_ab_b:
        return vertex_b

    vertex_c_coordinate = dot_ab_a * dot_ac_b - dot_ab_b * dot_ac_a
    if vertex_c_coordinate <= 0.0 and dot_ab_a >= 0.0 and dot_ab_b <= 0.0:
        return _scaled_add(vertex_a, edge_ab, dot_ab_a / (dot_ab_a - dot_ab_b))

    offset_c = _subtract(point, vertex_c)
    dot_ab_c = _dot(edge_ab, offset_c)
    dot_ac_c = _dot(edge_ac, offset_c)
    if dot_ac_c >= 0.0 and dot_ab_c <= dot_ac_c:
        return vertex_c

    vertex_b_coordinate = dot_ab_c * dot_ac_a - dot_ab_a * dot_ac_c
    if vertex_b_coordinate <= 0.0 and dot_ac_a >= 0.0 and dot_ac_c <= 0.0:
        return _scaled_add(vertex_a, edge_ac, dot_ac_a / (dot_ac_a - dot_ac_c))

    vertex_a_coordinate = dot_ab_b * dot_ac_c - dot_ab_c * dot_ac_b
    if vertex_a_coordinate <= 0.0 and dot_ac_b - dot_ab_b >= 0.0 and dot_ab_c - dot_ac_c >= 0.0:
        edge_bc = _subtract(vertex_c, vertex_b)
        scale = (dot_ac_b - dot_ab_b) / (dot_ac_b - dot_ab_b + dot_ab_c - dot_ac_c)
        return _scaled_add(vertex_b, edge_bc, scale)

    denominator = 1.0 / (vertex_a_coordinate + vertex_b_coordinate + vertex_c_coordinate)
    scale_ab = vertex_b_coordinate * denominator
    scale_ac = vertex_c_coordinate * denominator
    return _scaled_add(_scaled_add(vertex_a, edge_ab, scale_ab), edge_ac, scale_ac)


def _nearest_triangle(
    point: Point3,
    vertices: tuple[Point3, ...],
    triangles: tuple[tuple[int, int, int], ...],
    centroids: tuple[Point3, ...],
) -> tuple[int, float, Point3]:
    candidates = (
        (
            math.dist(point, closest_point),
            math.dist(point, centroids[node]),
            node,
            closest_point,
        )
        for node, triangle in enumerate(triangles)
        for closest_point in (
            _closest_point_on_triangle(
                point,
                vertices[triangle[0]],
                vertices[triangle[1]],
                vertices[triangle[2]],
            ),
        )
    )
    distance, _centroid_distance, node, closest_point = min(candidates)
    return node, distance, closest_point


def direct_navmesh_traversability(
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

    links: list[tuple[Point3, ...]] = []
    link_endpoints: dict[Point3, tuple[int, Point3]] = {}
    for raw_link in navmesh.links:
        if len(raw_link) < 2:
            raise ValueError("runtime navmesh link must contain at least two points")
        link: tuple[Point3, ...] = tuple(
            (
                round(point[0], _VERTEX_DIGITS),
                round(point[1], _VERTEX_DIGITS),
                round(point[2], _VERTEX_DIGITS),
            )
            for point in raw_link
        )
        start, end = link[0], link[-1]
        node_a, distance_a, closest_a = _nearest_triangle(
            start,
            vertices,
            triangles,
            triangle_nodes,
        )
        node_b, distance_b, closest_b = _nearest_triangle(
            end,
            vertices,
            triangles,
            triangle_nodes,
        )
        if max(distance_a, distance_b) > _LINK_TRIANGLE_TOLERANCE:
            raise ValueError(
                "runtime navmesh link endpoint is not on the exported triangulation: "
                f"start={start} distance={distance_a}, end={end} distance={distance_b}"
            )
        link_endpoints.setdefault(start, (node_a, closest_a))
        link_endpoints.setdefault(end, (node_b, closest_b))
        links.append(link)

    endpoint_nodes = {
        point: len(triangle_nodes) + index for index, point in enumerate(link_endpoints)
    }
    nodes = (*triangle_nodes, *endpoint_nodes)
    for point, endpoint_node in endpoint_nodes.items():
        triangle_node, closest_point = link_endpoints[point]
        attachment_points = (
            (triangle_nodes[triangle_node], point)
            if closest_point == point
            else (triangle_nodes[triangle_node], closest_point, point)
        )
        path = _dense_path(attachment_points, move_magnitude)
        edges.append(
            TraversabilityEdge(
                node_a=triangle_node,
                node_b=endpoint_node,
                path=path,
                cost=_path_length(path),
                portal=(point, point),
            )
        )

    for link in links:
        node_a = endpoint_nodes[link[0]]
        node_b = endpoint_nodes[link[-1]]
        if node_a == node_b:
            continue
        path = _dense_path(link, move_magnitude)
        if node_a > node_b:
            node_a, node_b = node_b, node_a
            path = tuple(reversed(path))
        edges.append(
            TraversabilityEdge(
                node_a=node_a,
                node_b=node_b,
                path=path,
                cost=_path_length(path),
            )
        )

    return TraversabilityMap(
        source=TraversabilitySource.AI2THOR_NAVMESH_GROUND_TRUTH,
        nodes=nodes,
        edges=tuple(edges),
    )


def _primary_component_nodes(traversability_map: TraversabilityMap) -> frozenset[int]:
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
    return max(components, key=lambda component: (len(component), -min(component)))


def _region_candidate_nodes(
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
    primary_nodes = _primary_component_nodes(task.graph.traversability_map)
    connected_candidates = tuple(node for node in candidates if node in primary_nodes)
    if not connected_candidates:
        raise ValueError(
            f"runtime navmesh primary component has no triangle centroid in region {region_id!r}"
        )
    return connected_candidates


def select_region_traversability_node(
    task: SceneGraphTask,
    region_id: str,
    target: Point3,
    navigable_tolerance: float,
) -> int:
    """Select the navmesh triangle centroid nearest a point within one region."""

    candidates = _region_candidate_nodes(task, region_id, navigable_tolerance)
    return min(
        candidates,
        key=lambda node: (
            math.dist(task.graph.traversability_map.nodes[node], target),
            node,
        ),
    )


def select_region_traversability_point(
    task: SceneGraphTask,
    region_id: str,
    target: Point3,
    navigable_tolerance: float,
) -> Point3:
    """Select the navmesh point nearest a target within one semantic region."""

    node = select_region_traversability_node(
        task,
        region_id,
        target,
        navigable_tolerance,
    )
    return task.graph.traversability_map.nodes[node]


__all__ = [
    "direct_navmesh_traversability",
    "select_region_traversability_node",
    "select_region_traversability_point",
]
