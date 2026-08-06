"""Local scene-graph and symbolic-planning data structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from dso_hadr.types.navigation import Point3, Pose


class GraphEvidence(str, Enum):
    DATASET_SEMANTICS = "dataset_semantics"


class RegionKind(str, Enum):
    ROOM = "room"
    HALLWAY = "hallway"


class ConnectivityKind(str, Enum):
    DOOR = "door"
    ADJACENT = "adjacent"
    CROSS_FLOOR = "cross_floor"


class TraversabilitySource(str, Enum):
    AI2THOR_NAVMESH_GROUND_TRUTH = "ai2thor_navmesh_ground_truth"


@dataclass(frozen=True)
class FloorNode:
    id: str
    label: str
    scene_model: str
    level_index: int
    evidence: GraphEvidence


@dataclass(frozen=True)
class RegionNode:
    id: str
    label: str
    category: str
    kind: RegionKind
    floor_id: str
    navigation_pose: Pose
    bounds_xz: tuple[float, float, float, float]
    semantic_region_value: int
    evidence: GraphEvidence


@dataclass(frozen=True)
class ContainsEdge:
    id: str
    floor_id: str
    region_id: str
    evidence: GraphEvidence


@dataclass(frozen=True)
class ConnectivityEdge:
    id: str
    label: str
    node_a: str
    node_b: str
    kind: ConnectivityKind
    pose_a: Pose
    pose_b: Pose
    cost: float
    evidence: GraphEvidence
    evidence_detail: str
    supporting_entity_id: str

    def orient(self, source_region_id: str) -> tuple[str, Pose, Pose]:
        if source_region_id == self.node_a:
            return self.node_b, self.pose_a, self.pose_b
        if source_region_id == self.node_b:
            return self.node_a, self.pose_b, self.pose_a
        raise ValueError(f"region {source_region_id!r} is not connected by {self.id!r}")


@dataclass(frozen=True)
class TraversabilityEdge:
    node_a: int
    node_b: int
    path: tuple[Point3, ...]
    cost: float
    portal: tuple[Point3, Point3] | None = None

    def orient(self, source_node: int) -> tuple[int, tuple[Point3, ...]]:
        if source_node == self.node_a:
            return self.node_b, self.path
        if source_node == self.node_b:
            return self.node_a, tuple(reversed(self.path))
        raise ValueError(f"node {source_node} is not connected by this edge")


@dataclass(frozen=True)
class TraversabilityMap:
    source: TraversabilitySource
    nodes: tuple[Point3, ...]
    edges: tuple[TraversabilityEdge, ...]


@dataclass(frozen=True)
class SceneGraph:
    scene_id: str
    floors: tuple[FloorNode, ...]
    regions: tuple[RegionNode, ...]
    containment_edges: tuple[ContainsEdge, ...]
    connectivity_edges: tuple[ConnectivityEdge, ...]
    traversability_map: TraversabilityMap


@dataclass(frozen=True)
class FloorGrid:
    floor_id: str
    traversable: np.ndarray[tuple[int, int], np.dtype[np.bool_]]
    semantic_regions: np.ndarray[tuple[int, int], np.dtype[np.int32]]
    meters_per_pixel: float
    origin_xz: tuple[float, float]
    floor_height: float

    def grid_to_world(self, cell: tuple[int, int]) -> tuple[float, float, float]:
        row, column = cell
        return (
            self.origin_xz[0] + column * self.meters_per_pixel,
            self.floor_height,
            self.origin_xz[1] + row * self.meters_per_pixel,
        )


@dataclass(frozen=True)
class SceneGraphTask:
    graph: SceneGraph
    floor_grids: tuple[FloorGrid, ...]
    start_region_id: str
    goal_region_id: str


@dataclass(frozen=True)
class SymbolicSubgoal:
    region_id: str
    target_pose: Pose
    incoming_edge_id: str | None


@dataclass(frozen=True)
class SymbolicPlan:
    start_region_id: str
    goal_region_id: str
    region_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    subgoals: tuple[SymbolicSubgoal, ...]
    total_cost: float


__all__ = [
    "ConnectivityEdge",
    "ConnectivityKind",
    "ContainsEdge",
    "FloorGrid",
    "FloorNode",
    "GraphEvidence",
    "RegionKind",
    "RegionNode",
    "SceneGraph",
    "SceneGraphTask",
    "SymbolicPlan",
    "SymbolicSubgoal",
    "TraversabilityEdge",
    "TraversabilityMap",
    "TraversabilitySource",
]
