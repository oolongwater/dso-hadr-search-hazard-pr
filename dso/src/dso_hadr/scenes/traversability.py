"""Extract a ground-truth traversability map from simulator navmesh data."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from collections.abc import Callable

from dso_hadr.graph.model import (
    TraversabilityEdge,
    TraversabilityMap,
    TraversabilitySource,
)
from dso_hadr.types.navigation import Point3, ShortestPath, as_point3

NavmeshPathQuery = Callable[[Point3, Point3, float], ShortestPath | None]


def ground_truth_traversability_nodes(
    navmesh_samples: tuple[Point3, ...],
) -> TraversabilityMap:
    """Create a ground-truth map whose connectivity has not yet been extracted."""

    return TraversabilityMap(
        source=TraversabilitySource.AI2THOR_NAVMESH_GROUND_TRUTH,
        nodes=tuple(sorted(set(navmesh_samples))),
        edges=(),
    )


def extract_ground_truth_traversability(
    traversability_map: TraversabilityMap,
    anchors: tuple[Point3, ...],
    *,
    grid_size: float,
    max_vertical_step: float,
    max_edge_length_ratio: float,
    max_link_path_length_ratio: float,
    max_link_distance: float,
    max_transition_slope_ratio: float,
    transition_height_tolerance: float,
    link_candidate_count: int,
    query_navmesh_path: NavmeshPathQuery,
) -> TraversabilityMap:
    """Populate the map with ground-truth edges connecting ordered anchors."""

    nodes = traversability_map.nodes
    origin_x = min(point[0] for point in nodes)
    origin_z = min(point[2] for point in nodes)

    def grid_cell(point: Point3) -> tuple[int, int]:
        return (
            round((point[0] - origin_x) / grid_size),
            round((point[2] - origin_z) / grid_size),
        )

    nodes_by_cell: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(nodes):
        nodes_by_cell.setdefault(grid_cell(point), []).append(index)

    def local_neighbors(node: int) -> tuple[int, ...]:
        point = nodes[node]
        column, row = grid_cell(point)
        cell_radius = math.ceil(max_edge_length_ratio)
        max_distance = grid_size * max_edge_length_ratio
        neighbors: list[int] = []
        offsets = range(-cell_radius, cell_radius + 1)
        for column_offset, row_offset in itertools.product(offsets, repeat=2):
            if column_offset == 0 and row_offset == 0:
                continue
            for candidate in nodes_by_cell.get(
                (column + column_offset, row + row_offset),
                (),
            ):
                candidate_point = nodes[candidate]
                if (
                    math.hypot(
                        candidate_point[0] - point[0],
                        candidate_point[2] - point[2],
                    )
                    <= max_distance
                    and abs(candidate_point[1] - point[1]) <= max_vertical_step
                ):
                    neighbors.append(candidate)
        return tuple(neighbors)

    unvisited = set(range(len(nodes)))
    components: list[tuple[int, ...]] = []
    while unvisited:
        first = unvisited.pop()
        component = [first]
        queue = deque((first,))
        while queue:
            current = queue.popleft()
            for neighbor in local_neighbors(current):
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        components.append(tuple(component))

    link_neighbors: dict[int, set[int]] = {}
    for source_component, target_component in itertools.combinations(components, 2):
        candidates = heapq.nsmallest(
            link_candidate_count,
            (
                edge
                for edge in itertools.product(source_component, target_component)
                if math.dist(nodes[edge[0]], nodes[edge[1]]) <= max_link_distance
                and abs(nodes[edge[1]][1] - nodes[edge[0]][1])
                <= math.hypot(
                    nodes[edge[1]][0] - nodes[edge[0]][0],
                    nodes[edge[1]][2] - nodes[edge[0]][2],
                )
                * max_transition_slope_ratio
                + transition_height_tolerance
            ),
            key=lambda edge: math.dist(nodes[edge[0]], nodes[edge[1]]),
        )
        for source, target in candidates:
            link_neighbors.setdefault(source, set()).add(target)
            link_neighbors.setdefault(target, set()).add(source)

    accepted: dict[tuple[int, int], TraversabilityEdge] = {
        (edge.node_a, edge.node_b): edge for edge in traversability_map.edges
    }

    def edge_key(node_a: int, node_b: int) -> tuple[int, int]:
        return min(node_a, node_b), max(node_a, node_b)

    def neighbors(current: int) -> tuple[int, ...]:
        candidates = set(local_neighbors(current))
        candidates.update(link_neighbors.get(current, ()))
        return tuple(sorted(candidates))

    def nearest_node(point: Point3) -> int:
        target = as_point3(point)
        return min(
            range(len(nodes)),
            key=lambda node: (math.dist(target, nodes[node]), node),
        )

    def candidate_route(
        start_node: int,
        goal_node: int,
    ) -> tuple[tuple[int, ...] | None, set[int]]:
        frontier: list[tuple[float, float, int]] = [
            (math.dist(nodes[start_node], nodes[goal_node]), 0.0, start_node)
        ]
        came_from: dict[int, int] = {}
        cost_so_far = {start_node: 0.0}
        while frontier:
            _priority, current_cost, current = heapq.heappop(frontier)
            if current_cost != cost_so_far[current]:
                continue
            if current == goal_node:
                break
            for neighbor in neighbors(current):
                key = edge_key(current, neighbor)
                if key in rejected:
                    continue
                edge_cost = (
                    accepted[key].cost
                    if key in accepted
                    else math.dist(
                        nodes[current],
                        nodes[neighbor],
                    )
                )
                neighbor_cost = current_cost + edge_cost
                if neighbor_cost >= cost_so_far.get(neighbor, math.inf):
                    continue
                cost_so_far[neighbor] = neighbor_cost
                came_from[neighbor] = current
                heapq.heappush(
                    frontier,
                    (
                        neighbor_cost + math.dist(nodes[neighbor], nodes[goal_node]),
                        neighbor_cost,
                        neighbor,
                    ),
                )
        if goal_node != start_node and goal_node not in came_from:
            return None, set(cost_so_far)
        route = [goal_node]
        while route[-1] != start_node:
            route.append(came_from[route[-1]])
        return tuple(reversed(route)), set(cost_so_far)

    anchor_nodes = tuple(nearest_node(anchor) for anchor in anchors)
    for start_node, goal_node in zip(anchor_nodes, anchor_nodes[1:]):
        rejected: set[tuple[int, int]] = set()
        while True:
            route, reachable = candidate_route(start_node, goal_node)
            if route is None:
                bridge_candidates = heapq.nsmallest(
                    link_candidate_count,
                    (
                        (source, target)
                        for source in reachable
                        for target in range(len(nodes))
                        if target not in reachable
                        and edge_key(source, target) not in rejected
                        and math.dist(nodes[source], nodes[target]) <= max_link_distance
                        and abs(nodes[target][1] - nodes[source][1])
                        <= math.hypot(
                            nodes[target][0] - nodes[source][0],
                            nodes[target][2] - nodes[source][2],
                        )
                        * max_transition_slope_ratio
                        + transition_height_tolerance
                    ),
                    key=lambda edge: math.dist(nodes[edge[0]], nodes[edge[1]]),
                )
                if not bridge_candidates:
                    raise ValueError(
                        f"ground-truth traversability extraction could not connect "
                        f"{nodes[start_node]} to {nodes[goal_node]} after rejecting "
                        f"{len(rejected)} candidate edges"
                    )
                for source, target in bridge_candidates:
                    link_neighbors.setdefault(source, set()).add(target)
                    link_neighbors.setdefault(target, set()).add(source)
                continue
            unknown = [
                (source, target)
                for source, target in zip(route, route[1:])
                if edge_key(source, target) not in accepted
            ]
            if not unknown:
                break
            for source, target in unknown:
                source_point = nodes[source]
                target_point = nodes[target]
                path_length_ratio = (
                    max_link_path_length_ratio
                    if target in link_neighbors.get(source, ())
                    else max_edge_length_ratio
                )
                path = query_navmesh_path(
                    source_point,
                    target_point,
                    math.dist(source_point, target_point) * path_length_ratio,
                )
                if path is not None and any(
                    abs(point_b[1] - point_a[1])
                    > math.hypot(
                        point_b[0] - point_a[0],
                        point_b[2] - point_a[2],
                    )
                    * max_transition_slope_ratio
                    + transition_height_tolerance
                    for point_a, point_b in zip(path.points, path.points[1:])
                ):
                    path = None
                key = edge_key(source, target)
                if path is None:
                    rejected.add(key)
                    break
                accepted[key] = TraversabilityEdge(
                    node_a=key[0],
                    node_b=key[1],
                    path=(path.points if source == key[0] else tuple(reversed(path.points))),
                    cost=path.geodesic_distance,
                )

    return TraversabilityMap(
        source=traversability_map.source,
        nodes=nodes,
        edges=tuple(accepted[key] for key in sorted(accepted)),
    )


__all__ = [
    "extract_ground_truth_traversability",
    "ground_truth_traversability_nodes",
]
