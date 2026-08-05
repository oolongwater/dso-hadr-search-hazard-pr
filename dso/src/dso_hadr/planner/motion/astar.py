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


def remove_immediate_backtracks(points: tuple[Point3, ...]) -> tuple[Point3, ...]:
    """Remove exact A-B-A spikes without smoothing valid corners."""

    simplified: list[Point3] = []
    for point in points:
        if simplified and point == simplified[-1]:
            continue
        if len(simplified) >= 2 and point == simplified[-2]:
            simplified.pop()
        else:
            simplified.append(point)
    return tuple(simplified)


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
            neighbor, _path = edge.orient(current)
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

    points: list[Point3] = [nodes[start_node]]
    for source, _target, edge in reversed(reversed_steps):
        _neighbor, edge_path = edge.orient(source)
        for point in edge_path:
            if point != points[-1]:
                points.append(point)
    simplified = remove_immediate_backtracks(tuple(points))
    distance = sum(math.dist(source, target) for source, target in pairwise(simplified))
    return ShortestPath(points=simplified, geodesic_distance=distance)


__all__ = ["astar_search", "remove_immediate_backtracks"]
