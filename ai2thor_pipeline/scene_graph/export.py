"""Glue between iTHOR hazard runs and the canonical scene-graph schema."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.thor import CURATED_ITHOR_SCENES
from scene_graph.panel import (
    _as_dict_list,
    _specific_parent,
    compute_passage_clearance,
    heat_state_ids,
    parent_snapshot,
    severed_support_edges,
)
from scene_graph.schema import (
    Edge,
    EdgeType,
    HazardState,
    PassageState,
    SceneGraph,
)
from scene_graph.validators import ValidationReport, validate_scene_graph

FLOOR_ID = "floor_0"


def _room_type_for_scene(scene: str) -> str:
    for name, room_type in CURATED_ITHOR_SCENES:
        if name == scene:
            return room_type
    return "room"


def _bounds_from_map_props(map_props: dict[str, Any] | None) -> list[list[float]]:
    if not map_props or not map_props.get("position"):
        return [[-3.0, 0.0, -3.0], [3.0, 0.0, -3.0], [3.0, 0.0, 3.0], [-3.0, 0.0, 3.0]]
    pos = map_props["position"]
    cx = float(pos.get("x", 0.0))
    cz = float(pos.get("z", 0.0))
    half = float(map_props.get("orthographicSize", 3.0))
    return [
        [cx - half, 0.0, cz - half],
        [cx + half, 0.0, cz - half],
        [cx + half, 0.0, cz + half],
        [cx - half, 0.0, cz + half],
    ]


def _bounds_from_objects(objects: list[dict[str, Any]]) -> list[list[float]]:
    xs: list[float] = []
    zs: list[float] = []
    for obj in objects:
        bbox = obj.get("axisAlignedBoundingBox") or {}
        center = bbox.get("center") or obj.get("position") or {}
        if center:
            xs.append(float(center.get("x", 0.0)))
            zs.append(float(center.get("z", 0.0)))
    if not xs:
        return [[-3.0, 0.0, -3.0], [3.0, 0.0, 3.0]]
    pad = 0.5
    return [
        [min(xs) - pad, 0.0, min(zs) - pad],
        [max(xs) + pad, 0.0, min(zs) - pad],
        [max(xs) + pad, 0.0, max(zs) + pad],
        [min(xs) - pad, 0.0, max(zs) + pad],
    ]


def _centroid_from_bounds(bounds: list[list[float]]) -> list[float]:
    xs = [b[0] for b in bounds]
    zs = [b[2] for b in bounds]
    return [sum(xs) / len(xs), 0.0, sum(zs) / len(zs)]


def _split_bounds_along_forward(
    bounds: list[list[float]],
    origin: tuple[float, float],
    forward: tuple[float, float],
) -> tuple[list[list[float]], list[list[float]], float]:
    """Split room bounds into here/ahead halves along the agent forward axis."""
    xs = [b[0] for b in bounds]
    zs = [b[2] for b in bounds]
    corners = [(x, z) for x in xs for z in zs]
    projections = [
        (x - origin[0]) * forward[0] + (z - origin[1]) * forward[1]
        for x, z in corners
    ]
    split = sum(projections) / len(projections)

    def _half(filter_fn) -> list[list[float]]:
        hx = [b[0] for b in bounds if filter_fn(b)]
        hz = [b[2] for b in bounds if filter_fn(b)]
        if not hx:
            return bounds
        return [
            [min(hx), 0.0, min(hz)],
            [max(hx), 0.0, min(hz)],
            [max(hx), 0.0, max(hz)],
            [min(hx), 0.0, max(hz)],
        ]

    here_bounds = _half(lambda b: (b[0] - origin[0]) * forward[0] + (b[2] - origin[1]) * forward[1] <= split)
    ahead_bounds = _half(lambda b: (b[0] - origin[0]) * forward[0] + (b[2] - origin[1]) * forward[1] > split)
    return here_bounds, ahead_bounds, split


def house_from_ithor(
    scene: str,
    map_props: dict[str, Any] | None,
    objects: list[dict[str, Any]],
    *,
    hazard: str = "",
    viewpoint: dict[str, float] | None = None,
    view_yaw: float = 0.0,
) -> dict[str, Any]:
    """Build a ProcTHOR-shaped house dict from iTHOR metadata."""
    room_type = _room_type_for_scene(scene)
    bounds = _bounds_from_map_props(map_props)
    if not map_props or not map_props.get("position"):
        bounds = _bounds_from_objects(objects)

    rooms: list[dict[str, Any]] = []
    doors: list[dict[str, Any]] = []

    if hazard == "obstruction" and viewpoint is not None:
        yaw_rad = math.radians(view_yaw)
        fwd = (math.sin(yaw_rad), math.cos(yaw_rad))
        origin = (float(viewpoint["x"]), float(viewpoint["z"]))
        here_bounds, ahead_bounds, split = _split_bounds_along_forward(bounds, origin, fwd)
        rooms = [
            {"id": "room_here_0", "roomType": "here", "floorPolygon": here_bounds},
            {"id": "room_ahead_0", "roomType": "ahead", "floorPolygon": ahead_bounds},
        ]
        conn_x = origin[0] + fwd[0] * split * 0.5
        conn_z = origin[1] + fwd[1] * split * 0.5
        doors = [
            {
                "id": "conn_passage_0",
                "connector_type": "passage",
                "room0": "room_here_0",
                "room1": "room_ahead_0",
                "openable": False,
                "openness": 1.0,
                "holePolygon": [
                    [conn_x - 0.5, 0.0, conn_z - 0.5],
                    [conn_x + 0.5, 0.0, conn_z - 0.5],
                    [conn_x + 0.5, 0.0, conn_z + 0.5],
                    [conn_x - 0.5, 0.0, conn_z + 0.5],
                ],
            }
        ]
    else:
        rooms = [{"id": f"room_{room_type}_0", "roomType": room_type, "floorPolygon": bounds}]

    return {"rooms": rooms, "doors": doors}


def build_initial_scene_graph(
    scene: str,
    event,
    *,
    hazard: str = "",
    setup_info: dict[str, Any] | None = None,
    map_props: dict[str, Any] | None = None,
) -> SceneGraph:
    """Construct the oracle ground-truth graph at hazard setup time."""
    objects = list(event.metadata.get("objects") or [])
    viewpoint = (setup_info or {}).get("viewpoint")
    view_yaw = float((setup_info or {}).get("view_yaw") or 0.0)
    house = house_from_ithor(
        scene,
        map_props,
        objects,
        hazard=hazard,
        viewpoint=viewpoint if isinstance(viewpoint, dict) else None,
        view_yaw=view_yaw,
    )
    sg = SceneGraph.from_ai2thor(house, objects, scene_id=scene)
    return sg


def _thor_id_to_node_id(sg: SceneGraph, objects: list[dict[str, Any]]) -> dict[str, str]:
    """Map AI2-THOR objectId to canonical schema node id by category + position."""
    mapping: dict[str, str] = {}
    for raw in objects:
        oid = raw.get("objectId")
        if not oid:
            continue
        pos = raw.get("position") or {}
        cat = str(raw.get("objectType") or "")
        for obj_node in sg.nodes.object:
            if obj_node.gt.category != cat:
                continue
            if abs(obj_node.gt.pose[0] - float(pos.get("x", 0))) < 0.15 and abs(obj_node.gt.pose[2] - float(pos.get("z", 0))) < 0.15:
                mapping[str(oid)] = obj_node.id
                break
    return mapping


def _passage_state_from_report(edge_state: str) -> PassageState:
    mapping = {
        "open": PassageState.OPEN,
        "closed": PassageState.CLOSED,
        "blocked": PassageState.BLOCKED,
        "constrained": PassageState.CONSTRAINED,
    }
    return mapping.get(edge_state, PassageState.OPEN)


def apply_hazard_obs(
    sg: SceneGraph,
    hazard: str,
    event,
    report: dict[str, Any],
    timestep: int,
    *,
    heat_field=None,
    hot_threshold_c: float = 70.0,
    parent_baseline: dict[str, str | None] | None = None,
    tracked_ids: list[str] | None = None,
    setup_info: dict[str, Any] | None = None,
    thor_id_map: dict[str, str] | None = None,
) -> SceneGraph:
    """Update obs layer and hazard-specific edges; stamp last_updated."""
    sg = deepcopy(sg)
    objects = list(event.metadata.get("objects") or [])
    obj_map = {obj["objectId"]: obj for obj in objects if obj.get("objectId")}
    tracked = set(tracked_ids or [])
    id_map = thor_id_map or {}

    def _schema_id(thor_id: str) -> str:
        return id_map.get(thor_id, thor_id)

    for floor in sg.nodes.floor:
        floor.obs.last_updated = timestep

    for region in sg.nodes.region:
        region.obs.last_updated = timestep

    for conn in sg.nodes.connector:
        conn.obs.last_updated = timestep

    for obj in sg.nodes.object:
        obj.obs.last_updated = timestep

    if hazard == "smoke":
        for floor in sg.nodes.floor:
            floor.obs.hazard_state = HazardState.SMOKE
            floor.obs.last_updated = timestep
        severity = float(report.get("room_fill") or report.get("agent_density") or 0.0)
        for region in sg.nodes.region:
            region.obs.hazard_severity = severity
            region.obs.visible = True
        hot_ids = {_schema_id(oid) for oid in heat_state_ids(heat_field, list(tracked), threshold_c=hot_threshold_c)} if heat_field else set()
        for item in _as_dict_list(report.get("newly_ignited")):
            oid = item.get("objectId")
            if oid:
                hot_ids.add(_schema_id(str(oid)))
        for obj_node in sg.nodes.object:
            if obj_node.id in hot_ids:
                obj_node.obs.state = "hot"

    elif hazard == "earthquake":
        for floor in sg.nodes.floor:
            floor.obs.hazard_state = HazardState.DEBRIS
            floor.obs.last_updated = timestep
        parents_now = parent_snapshot(event, list(tracked)) if tracked else {}
        severed = severed_support_edges(parents_now, parent_baseline or {})
        severed_ids = {_schema_id(e.subject_id) for e in severed}
        for item in _as_dict_list(report.get("newly_broken")):
            oid = item.get("objectId")
            if oid:
                severed_ids.add(_schema_id(str(oid)))
        for obj_node in sg.nodes.object:
            if obj_node.id in severed_ids:
                obj_node.obs.fallen = True
                obj_node.obs.state = "broken"
            raw = obj_map.get(obj_node.id)
            if raw is None:
                for thor_id, node_id in id_map.items():
                    if node_id == obj_node.id:
                        raw = obj_map.get(thor_id)
                        break
            if raw and raw.get("isBroken"):
                obj_node.obs.state = "broken"

    elif hazard == "obstruction":
        edge_state = str(report.get("edge_state") or "open")
        passage_state = _passage_state_from_report(edge_state)
        viewpoint = dict((setup_info or {}).get("viewpoint") or {})
        view_yaw = float((setup_info or {}).get("view_yaw") or 0.0)
        placed_ids = list(report.get("placed_ids") or [])
        clearance_info = compute_passage_clearance(
            event, viewpoint, view_yaw, placed_ids,
        )
        gap_m = float(clearance_info.get("gap_m", 0.0))

        for conn in sg.nodes.connector:
            conn.obs.passage_state = passage_state
            conn.obs.clearance = gap_m
            conn.obs.last_updated = timestep

        sg.edges = [e for e in sg.edges if e.type != EdgeType.BLOCKS]
        conn_id = sg.nodes.connector[0].id if sg.nodes.connector else "conn_passage_0"
        for oid in placed_ids:
            schema_oid = _schema_id(str(oid))
            if sg.get_node(schema_oid):
                sg.edges.append(Edge(src=schema_oid, dst=conn_id, type=EdgeType.BLOCKS))

    return sg


def _edge_key(e: Edge) -> tuple[str, str, str, str | None]:
    return (e.src, e.dst, e.type.value, e.via)


def _edge_from_key(key: tuple[str, str, str, str | None]) -> Edge:
    src, dst, typ, via = key
    return Edge(src=src, dst=dst, type=EdgeType(typ), via=via)


def diff_scene_graph(prev: SceneGraph, curr: SceneGraph, step: int) -> dict[str, Any]:
    """Per-tick obs/edge delta. Replaying deltas onto ``initial`` reconstructs any tick."""
    node_deltas: dict[str, dict[str, Any]] = {}
    for node in curr.iter_nodes():
        before = prev.get_node(node.id)
        curr_obs = node.obs.model_dump(mode="json")
        if before is None:
            node_deltas[node.id] = curr_obs
            continue
        prev_obs = before.obs.model_dump(mode="json")
        changed = {k: v for k, v in curr_obs.items() if prev_obs.get(k) != v}
        if changed:
            node_deltas[node.id] = changed

    prev_edges = {_edge_key(e) for e in prev.edges}
    curr_edges = {_edge_key(e) for e in curr.edges}
    return {
        "step": step,
        "nodes": node_deltas,
        "edges_added": [list(k) for k in sorted(curr_edges - prev_edges)],
        "edges_removed": [list(k) for k in sorted(prev_edges - curr_edges)],
    }


def apply_timeline_delta(sg: SceneGraph, delta: dict[str, Any]) -> SceneGraph:
    """Apply one timeline delta onto a graph copy."""
    from scene_graph.schema import (
        ConnectorObs,
        FloorObs,
        Level,
        ObjectObs,
        RegionObs,
    )

    sg = deepcopy(sg)
    obs_classes = {
        Level.FLOOR: FloorObs,
        Level.REGION: RegionObs,
        Level.CONNECTOR: ConnectorObs,
        Level.OBJECT: ObjectObs,
    }
    for node_id, obs_changes in (delta.get("nodes") or {}).items():
        node = sg.get_node(node_id)
        if node is None:
            continue
        obs_data = node.obs.model_dump(mode="json")
        obs_data.update(obs_changes)
        node.obs = obs_classes[node.level].model_validate(obs_data)

    def _parse_edge(entry: list[Any]) -> Edge:
        src = str(entry[0])
        dst = str(entry[1])
        typ = str(entry[2])
        via = str(entry[3]) if len(entry) > 3 and entry[3] is not None else None
        return Edge(src=src, dst=dst, type=EdgeType(typ), via=via)

    remove_keys = {_edge_key(_parse_edge(list(e))) for e in (delta.get("edges_removed") or [])}
    sg.edges = [e for e in sg.edges if _edge_key(e) not in remove_keys]
    existing = {_edge_key(e) for e in sg.edges}
    for entry in delta.get("edges_added") or []:
        edge = _parse_edge(list(entry))
        key = _edge_key(edge)
        if key not in existing:
            sg.edges.append(edge)
            existing.add(key)
    return sg


def replay_timeline(initial: SceneGraph, timeline: list[dict[str, Any]]) -> list[SceneGraph]:
    """Reconstruct per-tick graph states by applying cumulative deltas."""
    graphs: list[SceneGraph] = []
    current = deepcopy(initial)
    for delta in timeline:
        current = apply_timeline_delta(current, delta)
        graphs.append(deepcopy(current))
    return graphs


def write_scene_graph(
    paths: dict[str, Path],
    initial: SceneGraph,
    final: SceneGraph,
    timeline: list[dict[str, Any]] | None = None,
) -> tuple[Path, ValidationReport]:
    """Write canonical scene graph JSON beside the hazard MP4."""
    validation = validate_scene_graph(final)
    out_path = paths["video"].with_suffix(".scenegraph.json")
    payload: dict[str, Any] = {
        "schema_version": initial.schema_version,
        "scene_id": initial.scene_id,
        "initial": initial.model_dump(mode="json"),
        "final": final.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
    }
    if timeline is not None:
        payload["timeline"] = timeline
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path, validation


def scene_graph_summary_block(
    out_path: Path,
    initial: SceneGraph,
    final: SceneGraph,
    validation: ValidationReport,
) -> dict[str, Any]:
    return {
        "artifact": str(out_path.name),
        "schema_version": initial.schema_version,
        "node_counts": {
            "floor": len(final.nodes.floor),
            "region": len(final.nodes.region),
            "connector": len(final.nodes.connector),
            "object": len(final.nodes.object),
            "edges": len(final.edges),
        },
        "validation_ok": validation.ok,
        "validation_errors": len(validation.errors),
        "validation_warnings": len(validation.warnings),
    }
