"""
Canonical scene-graph schema for language-grounded indoor navigation (HADR / SAR).

Serves two workstreams from one definition:
  - Oracle-first: ground-truth graphs extracted from AI2-THOR / ProcTHOR layouts.
  - Online: partial graphs built from observations (``obs`` layer + ``last_updated``).

Serialization is LLM-parseable JSON via ``.model_dump()`` / ``.model_validate()``.

AI2-THOR emitter integration
----------------------------
``SceneGraph.from_ai2thor(house, objects, scene_id)`` builds a graph from:

  (a) a ProcTHOR-style house dict with ``rooms`` and ``doors`` arrays, and
  (b) AI2-THOR object metadata (``event.metadata["objects"]``).

Field mapping:

  - FLOOR: always one node ``floor_0``.
  - REGION  <- ``house["rooms"][i]``: ``semantic_type`` <- ``roomType``;
    ``floor`` <- ``"floor_0"``; ``bounds``/``centroid`` <- ``floorPolygon`` corners.
  - CONNECTOR <- ``house["doors"][i]``: ``connector_type`` <- ``"door"`` if openable
    else ``"doorway"``/``"opening"``; ``endpoints`` <- ``[room0, room1]``;
    ``world_pos``/``width`` <- ``holePolygon``; ``obs.passage_state`` <- ``"open"``
    if (``openness`` > 0 or not openable) else ``"closed"``; ``obs.clearance`` <- width.
  - OBJECT  <- ``objects[i]``: ``category`` <- ``objectType``;
    pose <- position + rotation (euler->quat); ``bbox_extent`` <- AABB size;
    ``movable`` <- (pickupable or moveable);
    ``support_parent`` <- ``parentReceptacles[0]`` if present else containing region id;
    ``obs.state`` <- ``"open"``/``"closed"`` from openable/``isOpen``.
  - EDGES: ``floor_0`` ``contains`` each region; region ``contains`` each object;
    for each door emit region0 ``connected_by`` region1 (``via`` = connector id);
    for each object with a parent receptacle emit ``support_parent`` ``supports`` object.

Region assignment for objects uses point-in-polygon against room ``floorPolygon``,
falling back to nearest room centroid.

No simulator dependency at import time — the emitter consumes already-extracted dicts.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Iterator, Literal

import networkx as nx
from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"


class Level(str, Enum):
    FLOOR = "floor"
    REGION = "region"
    CONNECTOR = "connector"
    OBJECT = "object"


class ConnectorType(str, Enum):
    DOOR = "door"
    DOORWAY = "doorway"
    STAIRCASE = "staircase"
    PASSAGE = "passage"
    OPENING = "opening"


class PassageState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    BLOCKED = "blocked"
    CONSTRAINED = "constrained"


class Directionality(str, Enum):
    BIDIRECTIONAL = "bidirectional"
    UP = "up"
    DOWN = "down"
    ONE_WAY = "one_way"


class HazardState(str, Enum):
    NONE = "none"
    SMOKE = "smoke"
    FIRE = "fire"
    DEBRIS = "debris"
    STRUCTURAL = "structural"


class EdgeType(str, Enum):
    CONTAINS = "contains"
    CONNECTED_BY = "connected_by"
    ADJACENT_TO = "adjacent_to"
    SUPPORTS = "supports"
    BLOCKS = "blocks"
    REACHABLE_FROM = "reachable_from"
    SAME_REGION_AS = "same_region_as"


# --- Observation base ---


class ObsBase(BaseModel):
    last_updated: int | None = None


class FloorObs(ObsBase):
    explored: bool = False
    accessible: bool = True
    hazard_state: HazardState = HazardState.NONE


class RegionObs(ObsBase):
    explored: bool = False
    visible: bool = False
    traversable: bool = True
    hazard_severity: float = 0.0
    target_likelihood: float | None = None


class ConnectorObs(ObsBase):
    passage_state: PassageState = PassageState.OPEN
    clearance: float = 0.0
    traversal_cost: float = 1.0
    detected: bool = False
    confidence: float = 0.0


class ObjectObs(ObsBase):
    detected: bool = False
    confidence: float = 0.0
    fallen: bool = False
    task_relevance: float | None = None
    state: str | None = None


# --- Ground-truth attribute models ---


class FloorGT(BaseModel):
    floor_index: int


class RegionGT(BaseModel):
    semantic_type: str
    floor: str
    bounds: list[list[float]]
    centroid: list[float]

    @field_validator("centroid")
    @classmethod
    def _centroid_len(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("centroid must have length 3")
        return v


class ConnectorGT(BaseModel):
    connector_type: ConnectorType
    endpoints: list[str]
    world_pos: list[float]
    width: float
    directionality: Directionality = Directionality.BIDIRECTIONAL

    @field_validator("endpoints")
    @classmethod
    def _two_endpoints(cls, v: list[str]) -> list[str]:
        if len(v) != 2:
            raise ValueError("endpoints must have exactly 2 node ids")
        return v


class ObjectGT(BaseModel):
    category: str
    region: str
    pose: list[float]
    bbox_extent: list[float]
    movable: bool
    support_parent: str | None = None

    @field_validator("pose")
    @classmethod
    def _pose_len(cls, v: list[float]) -> list[float]:
        if len(v) != 7:
            raise ValueError("pose must have length 7 [x,y,z,qx,qy,qz,qw]")
        return v

    @field_validator("bbox_extent")
    @classmethod
    def _bbox_len(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError("bbox_extent must have length 3")
        return v


# --- Node models ---


class FloorNode(BaseModel):
    id: str
    level: Literal[Level.FLOOR] = Level.FLOOR
    gt: FloorGT
    obs: FloorObs = Field(default_factory=FloorObs)


class RegionNode(BaseModel):
    id: str
    level: Literal[Level.REGION] = Level.REGION
    gt: RegionGT
    obs: RegionObs = Field(default_factory=RegionObs)


class ConnectorNode(BaseModel):
    id: str
    level: Literal[Level.CONNECTOR] = Level.CONNECTOR
    gt: ConnectorGT
    obs: ConnectorObs = Field(default_factory=ConnectorObs)


class ObjectNode(BaseModel):
    id: str
    level: Literal[Level.OBJECT] = Level.OBJECT
    gt: ObjectGT
    obs: ObjectObs = Field(default_factory=ObjectObs)


AnyNode = FloorNode | RegionNode | ConnectorNode | ObjectNode


class NodeSets(BaseModel):
    floor: list[FloorNode] = Field(default_factory=list)
    region: list[RegionNode] = Field(default_factory=list)
    connector: list[ConnectorNode] = Field(default_factory=list)
    object: list[ObjectNode] = Field(default_factory=list)


class Edge(BaseModel):
    src: str
    dst: str
    type: EdgeType
    via: str | None = None

    @model_validator(mode="after")
    def _connected_by_requires_via(self) -> Edge:
        if self.type == EdgeType.CONNECTED_BY and not self.via:
            raise ValueError("connected_by edges must carry via = connector node id")
        return self


class SceneGraph(BaseModel):
    schema_version: str = SCHEMA_VERSION
    scene_id: str
    nodes: NodeSets = Field(default_factory=NodeSets)
    edges: list[Edge] = Field(default_factory=list)

    def iter_nodes(self) -> Iterator[AnyNode]:
        yield from self.nodes.floor
        yield from self.nodes.region
        yield from self.nodes.connector
        yield from self.nodes.object

    def get_node(self, node_id: str) -> AnyNode | None:
        for node in self.iter_nodes():
            if node.id == node_id:
                return node
        return None

    def node_ids(self) -> set[str]:
        return {node.id for node in self.iter_nodes()}

    def to_networkx(self) -> nx.Graph:
        g = nx.Graph()
        for node in self.iter_nodes():
            g.add_node(
                node.id,
                level=node.level.value,
                gt=node.gt.model_dump(mode="json"),
                obs=node.obs.model_dump(mode="json"),
            )
        for edge in self.edges:
            g.add_edge(
                edge.src,
                edge.dst,
                type=edge.type.value,
                via=edge.via,
            )
        return g

    @classmethod
    def from_networkx(cls, g: nx.Graph, *, scene_id: str, schema_version: str = SCHEMA_VERSION) -> SceneGraph:
        floors: list[FloorNode] = []
        regions: list[RegionNode] = []
        connectors: list[ConnectorNode] = []
        objects: list[ObjectNode] = []
        for nid, attrs in g.nodes(data=True):
            level = Level(attrs["level"])
            gt_data = attrs.get("gt", {})
            obs_data = attrs.get("obs", {})
            if level == Level.FLOOR:
                floors.append(FloorNode(id=nid, gt=FloorGT.model_validate(gt_data), obs=FloorObs.model_validate(obs_data)))
            elif level == Level.REGION:
                regions.append(RegionNode(id=nid, gt=RegionGT.model_validate(gt_data), obs=RegionObs.model_validate(obs_data)))
            elif level == Level.CONNECTOR:
                connectors.append(
                    ConnectorNode(id=nid, gt=ConnectorGT.model_validate(gt_data), obs=ConnectorObs.model_validate(obs_data))
                )
            elif level == Level.OBJECT:
                objects.append(ObjectNode(id=nid, gt=ObjectGT.model_validate(gt_data), obs=ObjectObs.model_validate(obs_data)))
        edges: list[Edge] = []
        for src, dst, attrs in g.edges(data=True):
            edges.append(
                Edge(
                    src=src,
                    dst=dst,
                    type=EdgeType(attrs.get("type", EdgeType.ADJACENT_TO.value)),
                    via=attrs.get("via"),
                )
            )
        return cls(
            schema_version=schema_version,
            scene_id=scene_id,
            nodes=NodeSets(floor=floors, region=regions, connector=connectors, object=objects),
            edges=edges,
        )

    @classmethod
    def from_ai2thor(
        cls,
        house: dict[str, Any],
        objects: list[dict[str, Any]],
        scene_id: str,
    ) -> SceneGraph:
        """Build a ground-truth graph from ProcTHOR house dict + AI2-THOR object metadata."""
        floor_id = "floor_0"
        floors = [FloorNode(id=floor_id, gt=FloorGT(floor_index=0))]

        rooms = house.get("rooms") or []
        region_nodes: list[RegionNode] = []
        region_polygons: dict[str, list[tuple[float, float]]] = {}
        type_counts: dict[str, int] = {}

        for room in rooms:
            room_type = str(room.get("roomType") or room.get("room_type") or "room")
            type_counts[room_type] = type_counts.get(room_type, 0) + 1
            idx = type_counts[room_type] - 1
            rid = str(room.get("id") or f"room_{_slug(room_type)}_{idx}")

            polygon = _polygon_from_room(room)
            bounds = [[p[0], 0.0, p[1]] for p in polygon] if polygon else [[0.0, 0.0, 0.0]]
            centroid = _polygon_centroid(polygon) if polygon else [0.0, 0.0, 0.0]
            region_polygons[rid] = polygon

            region_nodes.append(
                RegionNode(
                    id=rid,
                    gt=RegionGT(
                        semantic_type=room_type,
                        floor=floor_id,
                        bounds=bounds,
                        centroid=centroid,
                    ),
                )
            )

        doors = house.get("doors") or []
        connector_nodes: list[ConnectorNode] = []
        conn_counts: dict[str, int] = {}

        for door in doors:
            conn_type = _door_connector_type(door)
            conn_counts[conn_type.value] = conn_counts.get(conn_type.value, 0) + 1
            idx = conn_counts[conn_type.value] - 1
            cid = str(door.get("id") or f"conn_{conn_type.value}_{idx}")

            room0 = str(door.get("room0") or door.get("roomId0") or "")
            room1 = str(door.get("room1") or door.get("roomId1") or "")
            world_pos, width = _hole_polygon_metrics(door.get("holePolygon") or door.get("hole_polygon") or [])
            openness = float(door.get("openness") or 0.0)
            openable = bool(door.get("openable", True))
            passage_state = PassageState.OPEN if (openness > 0 or not openable) else PassageState.CLOSED

            connector_nodes.append(
                ConnectorNode(
                    id=cid,
                    gt=ConnectorGT(
                        connector_type=conn_type,
                        endpoints=[room0, room1],
                        world_pos=world_pos,
                        width=width,
                    ),
                    obs=ConnectorObs(passage_state=passage_state, clearance=width),
                )
            )

        obj_counts: dict[str, int] = {}
        thor_to_node: dict[str, str] = {}
        object_specs: list[tuple[dict[str, Any], str, str]] = []

        for obj in objects:
            oid = str(obj.get("objectId") or obj.get("id") or "")
            if not oid:
                continue
            category = str(obj.get("objectType") or obj.get("category") or "Object")
            obj_counts[category] = obj_counts.get(category, 0) + 1
            idx = obj_counts[category] - 1
            node_id = f"obj_{_slug(category)}_{idx}"
            thor_to_node[oid] = node_id
            object_specs.append((obj, oid, node_id))

        object_nodes: list[ObjectNode] = []
        for obj, oid, node_id in object_specs:
            pos = obj.get("position") or {}
            rot = obj.get("rotation") or {}
            pose = _pose_from_position_rotation(pos, rot)
            bbox = obj.get("axisAlignedBoundingBox") or {}
            size = bbox.get("size") or {}
            bbox_extent = [
                float(size.get("x", 0.0)),
                float(size.get("y", 0.0)),
                float(size.get("z", 0.0)),
            ]
            movable = bool(obj.get("pickupable") or obj.get("moveable"))

            region_id = _assign_region(pos, region_polygons, region_nodes)
            parents = obj.get("parentReceptacles") or []
            support_parent: str | None = None
            if parents:
                parent_oid = str(parents[0])
                support_parent = thor_to_node.get(parent_oid, parent_oid)
            else:
                support_parent = region_id

            obs_state = _object_obs_state(obj)
            object_nodes.append(
                ObjectNode(
                    id=node_id,
                    gt=ObjectGT(
                        category=str(obj.get("objectType") or obj.get("category") or "Object"),
                        region=region_id,
                        pose=pose,
                        bbox_extent=bbox_extent,
                        movable=movable,
                        support_parent=support_parent,
                    ),
                    obs=ObjectObs(state=obs_state),
                )
            )

        edges: list[Edge] = []
        for region in region_nodes:
            edges.append(Edge(src=floor_id, dst=region.id, type=EdgeType.CONTAINS))
        for obj_node in object_nodes:
            edges.append(Edge(src=obj_node.gt.region, dst=obj_node.id, type=EdgeType.CONTAINS))
        for conn in connector_nodes:
            r0, r1 = conn.gt.endpoints
            if r0 and r1:
                edges.append(Edge(src=r0, dst=r1, type=EdgeType.CONNECTED_BY, via=conn.id))
        valid_ids = {n.id for n in region_nodes} | {n.id for n in object_nodes}
        for obj_node in object_nodes:
            parent = obj_node.gt.support_parent
            if parent and parent != obj_node.gt.region and parent in valid_ids:
                edges.append(Edge(src=parent, dst=obj_node.id, type=EdgeType.SUPPORTS))

        return cls(
            scene_id=scene_id,
            nodes=NodeSets(floor=floors, region=region_nodes, connector=connector_nodes, object=object_nodes),
            edges=edges,
        )


# --- Helpers (no simulator imports) ---

_ID_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    return _ID_SLUG_RE.sub("_", name.lower()).strip("_") or "item"


def _euler_to_quat(rx_deg: float, ry_deg: float, rz_deg: float) -> tuple[float, float, float, float]:
    """Convert Unity euler degrees (XYZ order) to quaternion [qx, qy, qz, qw]."""
    rx = math.radians(rx_deg) * 0.5
    ry = math.radians(ry_deg) * 0.5
    rz = math.radians(rz_deg) * 0.5
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    qw = cx * cy * cz + sx * sy * sz
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    return qx, qy, qz, qw


def _pose_from_position_rotation(pos: dict[str, Any], rot: dict[str, Any]) -> list[float]:
    x = float(pos.get("x", 0.0))
    y = float(pos.get("y", 0.0))
    z = float(pos.get("z", 0.0))
    qx, qy, qz, qw = _euler_to_quat(
        float(rot.get("x", 0.0)),
        float(rot.get("y", 0.0)),
        float(rot.get("z", 0.0)),
    )
    return [x, y, z, qx, qy, qz, qw]


def _polygon_from_room(room: dict[str, Any]) -> list[tuple[float, float]]:
    raw = room.get("floorPolygon") or room.get("floor_polygon") or []
    polygon: list[tuple[float, float]] = []
    for pt in raw:
        if isinstance(pt, dict):
            polygon.append((float(pt.get("x", 0.0)), float(pt.get("z", pt.get("y", 0.0)))))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            polygon.append((float(pt[0]), float(pt[2] if len(pt) >= 3 else pt[1])))
    return polygon


def _polygon_centroid(polygon: list[tuple[float, float]]) -> list[float]:
    if not polygon:
        return [0.0, 0.0, 0.0]
    xs = [p[0] for p in polygon]
    zs = [p[1] for p in polygon]
    return [sum(xs) / len(xs), 0.0, sum(zs) / len(zs)]


def _point_in_polygon(x: float, z: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test on the XZ plane."""
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _nearest_region_id(
    x: float,
    z: float,
    region_nodes: list[RegionNode],
    region_polygons: dict[str, list[tuple[float, float]]],
) -> str:
    best_id = region_nodes[0].id if region_nodes else "room_unknown_0"
    best_dist = float("inf")
    for region in region_nodes:
        poly = region_polygons.get(region.id, [])
        if poly and _point_in_polygon(x, z, poly):
            return region.id
        cx, _, cz = region.gt.centroid
        dist = (x - cx) ** 2 + (z - cz) ** 2
        if dist < best_dist:
            best_dist = dist
            best_id = region.id
    return best_id


def _assign_region(
    pos: dict[str, Any],
    region_polygons: dict[str, list[tuple[float, float]]],
    region_nodes: list[RegionNode],
) -> str:
    if not region_nodes:
        return "room_unknown_0"
    x = float(pos.get("x", 0.0))
    z = float(pos.get("z", 0.0))
    for region in region_nodes:
        poly = region_polygons.get(region.id, [])
        if poly and _point_in_polygon(x, z, poly):
            return region.id
    return _nearest_region_id(x, z, region_nodes, region_polygons)


def _door_connector_type(door: dict[str, Any]) -> ConnectorType:
    explicit = door.get("connector_type") or door.get("connectorType")
    if explicit:
        try:
            return ConnectorType(str(explicit))
        except ValueError:
            pass
    if door.get("openable", True):
        return ConnectorType.DOOR
    return ConnectorType.DOORWAY


def _hole_polygon_metrics(hole: list[Any]) -> tuple[list[float], float]:
    if not hole:
        return [0.0, 0.0, 0.0], 0.0
    xs: list[float] = []
    zs: list[float] = []
    for pt in hole:
        if isinstance(pt, dict):
            xs.append(float(pt.get("x", 0.0)))
            zs.append(float(pt.get("z", pt.get("y", 0.0))))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            xs.append(float(pt[0]))
            zs.append(float(pt[2] if len(pt) >= 3 else pt[1]))
    if not xs:
        return [0.0, 0.0, 0.0], 0.0
    cx = sum(xs) / len(xs)
    cz = sum(zs) / len(zs)
    width = max(max(xs) - min(xs), max(zs) - min(zs))
    return [cx, 0.0, cz], width


def _object_obs_state(obj: dict[str, Any]) -> str | None:
    if obj.get("openable"):
        return "open" if obj.get("isOpen") else "closed"
    if obj.get("isBroken"):
        return "broken"
    if obj.get("isCooked"):
        return "cooked"
    if obj.get("isMoving"):
        return "moving"
    toggled = obj.get("isToggled")
    if toggled is not None:
        return "on" if toggled else "off"
    return None
