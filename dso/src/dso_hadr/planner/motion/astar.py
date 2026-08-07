"""A* motion planning over a scene traversability map."""

from __future__ import annotations

import heapq
import math
from itertools import pairwise

from dso_hadr.graph.model import TraversabilityEdge, TraversabilityMap
from dso_hadr.types.navigation import Point3, ShortestPath, as_point3


def _nearest_node(target: Point3, traversability_map: TraversabilityMap) -> int:
    point = as_point3(target)
    return min(
        range(len(traversability_map.nodes)),
        key=lambda node: (math.dist(point, traversability_map.nodes[node]), node),
    )


def _area_xz(apex: Point3, first: Point3, second: Point3) -> float:
    return (first[0] - apex[0]) * (second[2] - apex[2]) - (first[2] - apex[2]) * (
        second[0] - apex[0]
    )


def _same_xz(first: Point3, second: Point3) -> bool:
    return math.isclose(first[0], second[0], abs_tol=1e-9) and math.isclose(
        first[2],
        second[2],
        abs_tol=1e-9,
    )


def _orient_portal(
    source: Point3,
    target: Point3,
    portal: tuple[Point3, Point3],
) -> tuple[Point3, Point3]:
    first, second = portal
    first_side = _area_xz(source, target, first)
    second_side = _area_xz(source, target, second)
    return (second, first) if first_side >= second_side else (first, second)


def _inset_portal(
    portal: tuple[Point3, Point3],
    *,
    margin: float,
) -> tuple[Point3, Point3]:
    first, second = portal
    length = math.dist(first, second)
    if math.isclose(length, 0.0):
        return portal
    fraction = min(margin / length, 0.25)
    return (
        (
            first[0] + (second[0] - first[0]) * fraction,
            first[1] + (second[1] - first[1]) * fraction,
            first[2] + (second[2] - first[2]) * fraction,
        ),
        (
            second[0] + (first[0] - second[0]) * fraction,
            second[1] + (first[1] - second[1]) * fraction,
            second[2] + (first[2] - second[2]) * fraction,
        ),
    )


def _string_pull(
    start: Point3,
    goal: Point3,
    portals: tuple[tuple[Point3, Point3], ...],
) -> tuple[Point3, ...]:
    ordered = ((start, start), *portals, (goal, goal))
    apex = start
    left = start
    right = start
    apex_index = 0
    left_index = 0
    right_index = 0
    points: list[Point3] = [start]
    index = 1
    while index < len(ordered):
        next_left, next_right = ordered[index]

        if _area_xz(apex, right, next_right) <= 0.0:
            if _same_xz(apex, right) or _area_xz(apex, left, next_right) > 0.0:
                right = next_right
                right_index = index
            else:
                if left != points[-1]:
                    points.append(left)
                apex = left
                apex_index = left_index
                left = apex
                right = apex
                left_index = apex_index
                right_index = apex_index
                index = apex_index + 1
                continue

        if _area_xz(apex, left, next_left) >= 0.0:
            if _same_xz(apex, left) or _area_xz(apex, right, next_left) < 0.0:
                left = next_left
                left_index = index
            else:
                if right != points[-1]:
                    points.append(right)
                apex = right
                apex_index = right_index
                left = apex
                right = apex
                left_index = apex_index
                right_index = apex_index
                index = apex_index + 1
                continue
        index += 1

    if goal != points[-1]:
        points.append(goal)
    return tuple(points)


def _densify(points: tuple[Point3, ...], max_segment_length: float) -> tuple[Point3, ...]:
    dense: list[Point3] = [points[0]]
    for source, target in pairwise(points):
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


def _edge_resolution(edge: TraversabilityEdge) -> float:
    lengths = tuple(math.dist(source, target) for source, target in pairwise(edge.path))
    return max(lengths, default=edge.cost)


def _expand_steps(
    steps: tuple[tuple[int, int, TraversabilityEdge], ...],
    nodes: tuple[Point3, ...],
) -> tuple[Point3, ...]:
    portals = tuple(
        _inset_portal(
            _orient_portal(nodes[source], nodes[target], edge.portal),
            margin=0.5 * _edge_resolution(edge),
        )
        for source, target, edge in steps
    )
    pulled = _string_pull(nodes[steps[0][0]], nodes[steps[-1][1]], portals)
    resolution = max(_edge_resolution(edge) for _source, _target, edge in steps)
    return _densify(pulled, resolution)


def astar_search(
    traversability_map: TraversabilityMap,
    start: Point3,
    goal: Point3,
) -> ShortestPath:
    """Find a path using only the scene representation."""

    start_node = _nearest_node(start, traversability_map)
    goal_node = _nearest_node(goal, traversability_map)
    if start_node == goal_node:
        point = traversability_map.nodes[start_node]
        return ShortestPath(points=(point,), geodesic_distance=0.0)

    adjacency: dict[int, list[TraversabilityEdge]] = {}
    for edge in traversability_map.edges:
        adjacency.setdefault(edge.node_a, []).append(edge)
        adjacency.setdefault(edge.node_b, []).append(edge)

    nodes = traversability_map.nodes
    frontier: list[tuple[float, float, int]] = [
        (math.dist(nodes[start_node], nodes[goal_node]), 0.0, start_node)
    ]
    came_from: dict[int, tuple[int, TraversabilityEdge]] = {}
    cost_so_far = {start_node: 0.0}
    while frontier:
        _priority, current_cost, current = heapq.heappop(frontier)
        if current_cost != cost_so_far[current]:
            continue
        if current == goal_node:
            break
        for edge in adjacency.get(current, ()):
            neighbor = edge.neighbor(current)
            neighbor_cost = current_cost + edge.cost
            if neighbor_cost >= cost_so_far.get(neighbor, math.inf):
                continue
            cost_so_far[neighbor] = neighbor_cost
            came_from[neighbor] = current, edge
            heapq.heappush(
                frontier,
                (
                    neighbor_cost + math.dist(nodes[neighbor], nodes[goal_node]),
                    neighbor_cost,
                    neighbor,
                ),
            )

    if goal_node not in came_from:
        raise ValueError(
            f"A* could not connect {nodes[start_node]} to {nodes[goal_node]} in the "
            f"scene traversability map"
        )

    reversed_steps: list[tuple[int, int, TraversabilityEdge]] = []
    current = goal_node
    while current != start_node:
        previous, edge = came_from[current]
        reversed_steps.append((previous, current, edge))
        current = previous

    steps = tuple(reversed(reversed_steps))
    points = _expand_steps(steps, nodes)
    distance = sum(math.dist(source, target) for source, target in pairwise(points))
    return ShortestPath(points=points, geodesic_distance=distance)


__all__ = ["astar_search"]
