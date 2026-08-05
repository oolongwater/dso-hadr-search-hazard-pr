"""Deterministic symbolic search over the local scene graph."""

from __future__ import annotations

import heapq

from dso_hadr.graph.model import (
    ConnectivityEdge,
    SceneGraph,
    SymbolicPlan,
    SymbolicSubgoal,
)

SearchState = tuple[float, tuple[str, ...], tuple[str, ...]]


def _neighbors(
    graph: SceneGraph,
) -> dict[str, tuple[tuple[str, ConnectivityEdge], ...]]:
    entries: dict[str, list[tuple[str, ConnectivityEdge]]] = {
        region.id: [] for region in graph.regions
    }
    for edge in graph.connectivity_edges:
        entries[edge.node_a].append((edge.node_b, edge))
        entries[edge.node_b].append((edge.node_a, edge))
    return {
        region_id: tuple(sorted(neighbors, key=lambda item: (item[0], item[1].id)))
        for region_id, neighbors in entries.items()
    }


def dijkstra_search(
    graph: SceneGraph,
    start_region_id: str,
    goal_region_id: str,
) -> SymbolicPlan:
    """Return the minimum-cost symbolic route between two regions."""

    regions = {region.id: region for region in graph.regions}
    edges = {edge.id: edge for edge in graph.connectivity_edges}
    neighbors = _neighbors(graph)
    frontier: list[SearchState] = [(0.0, (start_region_id,), ())]
    best: dict[str, SearchState] = {start_region_id: frontier[0]}

    while frontier:
        state = heapq.heappop(frontier)
        cost, region_ids, edge_ids = state
        current = region_ids[-1]
        if best[current] != state:
            continue
        if current == goal_region_id:
            break
        for destination, edge in neighbors[current]:
            candidate = (
                cost + edge.cost,
                (*region_ids, destination),
                (*edge_ids, edge.id),
            )
            if destination not in best or candidate < best[destination]:
                best[destination] = candidate
                heapq.heappush(frontier, candidate)

    total_cost, region_ids, edge_ids = best[goal_region_id]
    subgoals = [SymbolicSubgoal(start_region_id, regions[start_region_id].navigation_pose, None)]
    for index, region_id in enumerate(region_ids[1:], start=1):
        edge_id = edge_ids[index - 1]
        edge = edges[edge_id]
        _, _, destination_pose = edge.orient(region_ids[index - 1])
        subgoals.append(SymbolicSubgoal(region_id, destination_pose, edge_id))

    return SymbolicPlan(
        start_region_id=start_region_id,
        goal_region_id=goal_region_id,
        region_ids=region_ids,
        edge_ids=edge_ids,
        subgoals=tuple(subgoals),
        total_cost=total_cost,
    )


__all__ = ["dijkstra_search"]
