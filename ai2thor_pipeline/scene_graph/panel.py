"""Ground-truth scene graph built from AI2-THOR object metadata.

Follows ThinkGraphs representation conventions: node label + state attributes,
authoritative on/in edges from parentReceptacles (resolved to the smallest
parent AABB), and deterministic directional predicates from axis-aligned
bounding boxes with a room-center anchor. Spatial predicates are computed but
not drawn. Obstruction uses a dedicated passage-clearance panel instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

GRAPH_PANEL_WIDTH = 560
MAX_TRACKED_NODES = 10
PROXIMITY_THRESHOLD_M = 2.0
TOP_K_SPATIAL = 5

GRAPH_AREA_TOP = 38
GRAPH_AREA_BOTTOM = 350
LEGEND_TOP = 356
MAX_LAYOUT_ROWS = 3
MAX_COLS = 4
PANEL_MARGIN_X = 20

WALKWAY_Y_MAX = 0.5
FORWARD_CHOKE_M = 0.75
FORWARD_TOL_M = 0.6
LATERAL_SPAN_M = 1.0

SUPPORT_COLOR = (180, 180, 180)
SPATIAL_COLOR = (100, 100, 100)
HAZARD_COLOR = (0, 140, 255)
SEVERED_COLOR = (0, 80, 255)
CHANGED_EDGE_COLOR = (255, 200, 0)
LABEL_BG = (16, 16, 16)
PENDING_COLOR = (90, 90, 90)
FREE_COLOR = (0, 180, 80)
BLOCKED_BAR_COLOR = (0, 80, 220)
ABOVE_WALKWAY_COLOR = (120, 120, 120)

STATE_COLORS = {
    "broken": (0, 0, 220),
    "hot": (0, 120, 255),
    "cooked": (0, 180, 180),
    "moving": (0, 200, 0),
    "open": (200, 120, 0),
    "normal": (220, 220, 220),
}

PASSAGE_BAR_COLORS = {
    "open": FREE_COLOR,
    "constrained": (0, 160, 220),
    "blocked": BLOCKED_BAR_COLOR,
}


@dataclass
class GraphNode:
    node_id: str
    label: str
    attributes: dict[str, Any]
    position: tuple[float, float] = (0.5, 0.5)


@dataclass
class GraphEdge:
    subject_id: str
    predicate: str
    object_id: str
    distance_m: float
    kind: str = "support"  # support | spatial | hazard | severed


@dataclass
class SceneGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)

    def serialize_edges(self) -> list[list[Any]]:
        return [
            [e.subject_id, e.predicate, e.object_id, round(e.distance_m, 3)]
            for e in self.edges
        ]


def _object_map(event) -> dict[str, dict[str, Any]]:
    return {o["objectId"]: o for o in (event.metadata.get("objects") or [])}


def _short_label(obj: dict[str, Any]) -> str:
    obj_type = str(obj.get("objectType") or "Object")
    return obj_type[:12]


def _aabb_bounds(obj: dict[str, Any]) -> tuple[float, float, float, float, float, float] | None:
    box = obj.get("axisAlignedBoundingBox") or {}
    center = box.get("center") or obj.get("position")
    size = box.get("size")
    if not center:
        return None
    cx = float(center["x"])
    cy = float(center["y"])
    cz = float(center["z"])
    if size:
        sx = float(size["x"]) / 2.0
        sy = float(size["y"]) / 2.0
        sz = float(size["z"]) / 2.0
        return (cx - sx, cy - sy, cz - sz, cx + sx, cy + sy, cz + sz)
    pad = 0.05
    return (cx - pad, cy - pad, cz - pad, cx + pad, cy + pad, cz + pad)


def _aabb_volume(obj: dict[str, Any]) -> float:
    bounds = _aabb_bounds(obj)
    if bounds is None:
        return float("inf")
    return (bounds[3] - bounds[0]) * (bounds[4] - bounds[1]) * (bounds[5] - bounds[2])


def _specific_parent(obj: dict[str, Any], obj_map: dict[str, dict[str, Any]]) -> str | None:
    parents = obj.get("parentReceptacles") or []
    candidates = [pid for pid in parents if pid in obj_map]
    if not candidates:
        return None
    return min(candidates, key=lambda pid: (_aabb_volume(obj_map[pid]), pid))


def _node_states(obj: dict[str, Any]) -> list[str]:
    states: list[str] = []
    if obj.get("isBroken"):
        states.append("broken")
    temp = obj.get("temperature")
    if temp in ("Hot", "Warm"):
        states.append("hot")
    if obj.get("isCooked"):
        states.append("cooked")
    if obj.get("isMoving"):
        states.append("moving")
    if obj.get("isOpen"):
        states.append("open")
    if not states:
        states.append("normal")
    return states


def _aabb_center(obj: dict[str, Any]) -> tuple[float, float, float] | None:
    bounds = _aabb_bounds(obj)
    if bounds is None:
        return None
    return (
        (bounds[0] + bounds[3]) / 2.0,
        (bounds[1] + bounds[4]) / 2.0,
        (bounds[2] + bounds[5]) / 2.0,
    )


def _dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _child_inside_parent(child: dict[str, Any], parent: dict[str, Any]) -> bool:
    center = _aabb_center(child)
    pb = _aabb_bounds(parent)
    if center is None or pb is None:
        return False
    return pb[0] <= center[0] <= pb[3] and pb[1] <= center[1] <= pb[4] and pb[2] <= center[2] <= pb[5]


def room_center_from_map_props(props: dict[str, Any] | None) -> tuple[float, float]:
    if props and props.get("position"):
        pos = props["position"]
        return float(pos["x"]), float(pos["z"])
    return 0.0, 0.0


def room_center_from_objects(event) -> tuple[float, float]:
    xs: list[float] = []
    zs: list[float] = []
    for obj in event.metadata.get("objects") or []:
        c = _aabb_center(obj)
        if c is None:
            continue
        xs.append(c[0])
        zs.append(c[2])
    if not xs:
        agent = event.metadata.get("agent", {}).get("position") or {}
        return float(agent.get("x", 0.0)), float(agent.get("z", 0.0))
    return sum(xs) / len(xs), sum(zs) / len(zs)


def select_tracked_nodes(event, hazard: str = "") -> list[str]:
    """Deterministic node set shared across variant runs of the same hazard."""
    if hazard == "obstruction":
        return []

    obj_map = _object_map(event)
    candidates: list[tuple[int, str]] = []
    for obj in obj_map.values():
        score = 0
        if obj.get("breakable"):
            score += 3
        if obj.get("pickupable"):
            score += 2
        if obj.get("moveable"):
            score += 1
        if score <= 0:
            continue
        candidates.append((score, obj["objectId"]))
    candidates.sort(key=lambda item: (-item[0], item[1]))

    tracked: list[str] = []
    tracked_set: set[str] = set()
    for _, oid in candidates:
        if len(tracked) >= MAX_TRACKED_NODES:
            break
        obj = obj_map.get(oid)
        if obj is None:
            continue
        parent_id = _specific_parent(obj, obj_map)
        slots_needed = 1 + (1 if parent_id and parent_id not in tracked_set else 0)
        if len(tracked) + slots_needed > MAX_TRACKED_NODES:
            continue
        if parent_id and parent_id not in tracked_set:
            tracked.append(parent_id)
            tracked_set.add(parent_id)
        if oid not in tracked_set:
            tracked.append(oid)
            tracked_set.add(oid)
    return tracked[:MAX_TRACKED_NODES]


def _resolved_parent_map(
    tracked: list[str],
    obj_map: dict[str, dict[str, Any]],
) -> dict[str, str]:
    parent_of: dict[str, str] = {}
    tracked_set = set(tracked)
    for nid in tracked:
        obj = obj_map.get(nid)
        if obj is None:
            continue
        parent_id = _specific_parent(obj, obj_map)
        if parent_id and parent_id in tracked_set:
            parent_of[nid] = parent_id
    return parent_of


def _layout_tier_order(tracked: list[str], event) -> list[str]:
    """Supporters first, then their children, then unsupported nodes."""
    obj_map = _object_map(event)
    parent_of = _resolved_parent_map(tracked, obj_map)
    children_of: dict[str, list[str]] = {}
    for child, parent in parent_of.items():
        children_of.setdefault(parent, []).append(child)

    supporters = sorted(set(parent_of.values()))
    ordered: list[str] = list(supporters)
    for parent in supporters:
        ordered.extend(sorted(children_of.get(parent, [])))
    orphans = sorted(
        nid for nid in tracked
        if nid not in parent_of and nid not in supporters
    )
    ordered.extend(orphans)

    seen: set[str] = set()
    deduped: list[str] = []
    for nid in ordered:
        if nid not in seen:
            seen.add(nid)
            deduped.append(nid)
    for nid in tracked:
        if nid not in seen:
            deduped.append(nid)
    return deduped


def _row_x_centers(n: int, width: int) -> list[int]:
    usable = width - 2 * PANEL_MARGIN_X
    if n <= 1:
        return [width // 2]
    pitch = usable / (n - 1)
    return [int(PANEL_MARGIN_X + i * pitch) for i in range(n)]


def compute_layout(tracked: list[str], event) -> dict[str, tuple[int, int]]:
    """Fixed pixel positions in the graph area; only edge colors change between frames."""
    width = GRAPH_PANEL_WIDTH
    ordered = _layout_tier_order(tracked, event)
    rows: list[list[str]] = []
    for i in range(0, len(ordered), MAX_COLS):
        rows.append(ordered[i : i + MAX_COLS])
    rows = rows[:MAX_LAYOUT_ROWS]

    graph_h = GRAPH_AREA_BOTTOM - GRAPH_AREA_TOP
    n_rows = max(1, len(rows))
    row_pitch = graph_h / n_rows

    layout: dict[str, tuple[int, int]] = {}
    for row_idx, row_nodes in enumerate(rows):
        y = int(GRAPH_AREA_TOP + row_pitch * (row_idx + 0.5))
        for x, nid in zip(_row_x_centers(len(row_nodes), width), row_nodes):
            layout[nid] = (x, y)

    for nid in tracked:
        layout.setdefault(nid, (width // 2, (GRAPH_AREA_TOP + GRAPH_AREA_BOTTOM) // 2))
    return layout


def _support_predicate(child: dict[str, Any], parent: dict[str, Any]) -> str:
    if parent.get("openable"):
        return "in"
    if _child_inside_parent(child, parent):
        return "in"
    return "on"


def _support_edges(tracked: set[str], obj_map: dict[str, dict[str, Any]]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for nid in tracked:
        obj = obj_map.get(nid)
        if obj is None:
            continue
        parent_id = _specific_parent(obj, obj_map)
        if not parent_id or parent_id not in tracked:
            continue
        parent_obj = obj_map[parent_id]
        pred = _support_predicate(obj, parent_obj)
        child_c = _aabb_center(obj)
        parent_c = _aabb_center(parent_obj)
        dist = _dist3(child_c, parent_c) if child_c and parent_c else 0.0
        edges.append(GraphEdge(nid, pred, parent_id, dist, kind="support"))
    return edges


def _spatial_predicate(
    subj_c: tuple[float, float, float],
    obj_c: tuple[float, float, float],
    room_center: tuple[float, float],
) -> str:
    dx = obj_c[0] - subj_c[0]
    dy = obj_c[1] - subj_c[1]
    dz = obj_c[2] - subj_c[2]
    horiz = math.hypot(dx, dz)
    if abs(dy) > max(0.08, horiz * 0.6):
        return "above" if dy > 0 else "under"
    rcx, rcz = room_center
    rel_x = subj_c[0] - rcx
    rel_z = subj_c[2] - rcz
    to_x = obj_c[0] - subj_c[0]
    to_z = obj_c[2] - subj_c[2]
    cross = rel_x * to_z - rel_z * to_x
    if abs(cross) < 0.05 and horiz < 0.15:
        return "near"
    return "left" if cross > 0 else "right"


def _spatial_edges(
    tracked: set[str],
    obj_map: dict[str, dict[str, Any]],
    room_center: tuple[float, float],
    support_pairs: set[tuple[str, str]],
) -> list[GraphEdge]:
    centers: dict[str, tuple[float, float, float]] = {}
    for nid in tracked:
        obj = obj_map.get(nid)
        if obj is None:
            continue
        c = _aabb_center(obj)
        if c:
            centers[nid] = c

    candidates: dict[str, list[tuple[float, GraphEdge]]] = {nid: [] for nid in tracked}
    ids = sorted(centers.keys())
    for i, sid in enumerate(ids):
        for oid in ids[i + 1 :]:
            if (sid, oid) in support_pairs or (oid, sid) in support_pairs:
                continue
            sc, oc = centers[sid], centers[oid]
            dist = _dist3(sc, oc)
            if dist > PROXIMITY_THRESHOLD_M:
                continue
            pred = _spatial_predicate(sc, oc, room_center)
            candidates[sid].append((dist, GraphEdge(sid, pred, oid, dist, kind="spatial")))
            rev = {"left": "right", "right": "left", "above": "under", "under": "above"}.get(pred, "near")
            candidates[oid].append((dist, GraphEdge(oid, rev, sid, dist, kind="spatial")))

    edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for nid in sorted(candidates):
        for _, edge in sorted(candidates[nid], key=lambda item: item[0])[:TOP_K_SPATIAL]:
            key = (edge.subject_id, edge.predicate, edge.object_id)
            rev = (edge.object_id, edge.predicate, edge.subject_id)
            if key in seen or rev in seen:
                continue
            seen.add(key)
            edges.append(edge)
    return edges


def _disambiguate_labels(nodes: dict[str, GraphNode]) -> None:
    by_label: dict[str, list[str]] = {}
    for nid, node in nodes.items():
        by_label.setdefault(node.label, []).append(nid)
    for label, nids in by_label.items():
        if len(nids) <= 1:
            continue
        for i, nid in enumerate(sorted(nids), start=1):
            nodes[nid].label = f"{label[:10]}{i}"


def build_graph(
    event,
    tracked: list[str],
    room_center: tuple[float, float],
) -> SceneGraph:
    obj_map = _object_map(event)
    tracked_set = set(tracked)
    nodes: dict[str, GraphNode] = {}

    for nid in tracked:
        obj = obj_map.get(nid)
        if obj is None:
            continue
        nodes[nid] = GraphNode(
            node_id=nid,
            label=_short_label(obj),
            attributes={
                "objectType": obj.get("objectType"),
                "state": _node_states(obj),
                "visible": bool(obj.get("visible", False)),
            },
        )

    _disambiguate_labels(nodes)
    support = _support_edges(tracked_set, obj_map)
    support_pairs = {(e.subject_id, e.object_id) for e in support}
    spatial = _spatial_edges(tracked_set, obj_map, room_center, support_pairs)
    return SceneGraph(nodes=nodes, edges=support + spatial)


def parent_snapshot(event, tracked: list[str]) -> dict[str, str | None]:
    obj_map = _object_map(event)
    return {
        nid: _specific_parent(obj_map[nid], obj_map)
        for nid in tracked
        if nid in obj_map
    }


def severed_support_edges(
    current: dict[str, str | None],
    baseline: dict[str, str | None],
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for nid, before in baseline.items():
        after = current.get(nid)
        if before and before != after:
            edges.append(GraphEdge(nid, "severed_from", before, 0.0, kind="severed"))
    return edges


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def heat_state_ids(
    heat_field,
    tracked: list[str],
    threshold_c: float = 70.0,
) -> set[str]:
    if heat_field is None:
        return set()
    tracked_set = set(tracked)
    return {
        sample.object_id
        for sample in heat_field.objects
        if sample.object_id in tracked_set and sample.temperature_c >= threshold_c
    }


def hazard_graph_edges(
    hazard: str,
    event,
    tracked: list[str],
    report: dict[str, Any],
    *,
    heat_field=None,
    hot_threshold_c: float = 70.0,
    parent_snapshot_now: dict[str, str | None] | None = None,
    parent_baseline: dict[str, str | None] | None = None,
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    tracked_set = set(tracked)
    if hazard == "smoke":
        for item in _as_dict_list(report.get("newly_ignited")):
            oid = item.get("objectId")
            if oid and oid in tracked_set:
                edges.append(GraphEdge(oid, "ignited", oid, 0.0, kind="hazard"))

    elif hazard == "earthquake":
        if parent_snapshot_now and parent_baseline:
            edges.extend(severed_support_edges(parent_snapshot_now, parent_baseline))
        for item in _as_dict_list(report.get("newly_broken")):
            oid = item.get("objectId")
            if oid and oid in tracked_set:
                edges.append(GraphEdge(oid, "broken", oid, 0.0, kind="hazard"))

    return edges


def _forward_lateral_axes(view_yaw: float) -> tuple[tuple[float, float], tuple[float, float]]:
    yaw_rad = math.radians(view_yaw)
    fwd = (math.sin(yaw_rad), math.cos(yaw_rad))
    lat = (math.cos(yaw_rad), -math.sin(yaw_rad))
    return fwd, lat


def _project_xz(
    point: tuple[float, float],
    origin: tuple[float, float],
    axis: tuple[float, float],
) -> float:
    dx = point[0] - origin[0]
    dz = point[1] - origin[1]
    return dx * axis[0] + dz * axis[1]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda iv: iv[0])
    merged = [sorted_iv[0]]
    for lo, hi in sorted_iv[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _max_gap(
    intervals: list[tuple[float, float]],
    lo_bound: float,
    hi_bound: float,
) -> float:
    if not intervals:
        return hi_bound - lo_bound
    merged = _merge_intervals(intervals)
    best = 0.0
    prev = lo_bound
    for lo, hi in merged:
        best = max(best, lo - prev)
        prev = max(prev, hi)
    return max(best, hi_bound - prev)


def compute_passage_clearance(
    event,
    viewpoint: dict[str, float],
    view_yaw: float,
    placed_ids: list[str],
    *,
    floor_y_max: float = WALKWAY_Y_MAX,
) -> dict[str, Any]:
    obj_map = _object_map(event)
    vx = float(viewpoint["x"])
    vz = float(viewpoint["z"])
    fwd, lat = _forward_lateral_axes(view_yaw)
    choke = (vx + fwd[0] * FORWARD_CHOKE_M, vz + fwd[1] * FORWARD_CHOKE_M)

    intervals: list[tuple[float, float]] = []
    in_walkway_ids: set[str] = set()
    above_walkway_ids: set[str] = set()

    for oid in placed_ids:
        obj = obj_map.get(oid)
        if obj is None:
            continue
        bounds = _aabb_bounds(obj)
        if bounds is None:
            continue
        cx = (bounds[0] + bounds[3]) / 2.0
        cz = (bounds[2] + bounds[5]) / 2.0
        fwd_dist = _project_xz((cx, cz), (vx, vz), fwd)
        if abs(fwd_dist - FORWARD_CHOKE_M) > FORWARD_TOL_M:
            continue

        if bounds[1] >= floor_y_max:
            above_walkway_ids.add(oid)
            continue

        in_walkway_ids.add(oid)
        corners = [
            (bounds[0], bounds[2]), (bounds[0], bounds[5]),
            (bounds[3], bounds[2]), (bounds[3], bounds[5]),
        ]
        lats = [_project_xz(c, choke, lat) for c in corners]
        lo = max(-LATERAL_SPAN_M, min(lats))
        hi = min(LATERAL_SPAN_M, max(lats))
        if hi > lo:
            intervals.append((lo, hi))

    gap_m = _max_gap(intervals, -LATERAL_SPAN_M, LATERAL_SPAN_M)
    return {
        "gap_m": gap_m,
        "intervals": _merge_intervals(intervals),
        "in_walkway_ids": in_walkway_ids,
        "above_walkway_ids": above_walkway_ids,
    }


def _blocker_labels(blocker_ids: list[str], event) -> dict[str, str]:
    obj_map = _object_map(event)
    type_counts: dict[str, int] = {}
    labels: dict[str, str] = {}
    for oid in blocker_ids:
        obj = obj_map.get(oid)
        obj_type = str((obj or {}).get("objectType") or "Object")
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        idx = type_counts[obj_type]
        base = obj_type[:10]
        labels[oid] = base if idx == 1 else f"{base}{idx}"
    return labels


def _node_color(
    states: list[str],
    *,
    hot_ids: set[str] | None = None,
    nid: str = "",
) -> tuple[int, int, int]:
    if hot_ids and nid in hot_ids and "broken" not in states:
        return STATE_COLORS["hot"]
    for key in ("broken", "hot", "cooked", "moving", "open"):
        if key in states:
            return STATE_COLORS[key]
    return STATE_COLORS["normal"]


def _draw_dashed_line(
    panel: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if dist < 1:
        return
    dash = 6
    gap = 4
    steps = max(1, int(dist / (dash + gap)))
    for i in range(steps):
        t0 = i / steps
        t1 = min(1.0, (i + dash / (dash + gap)) / steps)
        a = (int(p0[0] + (p1[0] - p0[0]) * t0), int(p0[1] + (p1[1] - p0[1]) * t0))
        b = (int(p0[0] + (p1[0] - p0[0]) * t1), int(p0[1] + (p1[1] - p0[1]) * t1))
        cv2.line(panel, a, b, color, thickness, cv2.LINE_AA)


def _draw_arc_line(
    panel: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    if dist < 24:
        cv2.line(panel, p0, p1, color, thickness, cv2.LINE_AA)
        return
    mid_x = (p0[0] + p1[0]) / 2.0
    mid_y = (p0[1] + p1[1]) / 2.0
    nx, ny = -dy / dist, dx / dist
    bulge = min(36.0, dist * 0.22)
    ctrl = (int(mid_x + nx * bulge), int(mid_y + ny * bulge))
    for step in range(20):
        u0 = step / 20.0
        u1 = (step + 1) / 20.0
        a = (
            int((1 - u0) ** 2 * p0[0] + 2 * (1 - u0) * u0 * ctrl[0] + u0 ** 2 * p1[0]),
            int((1 - u0) ** 2 * p0[1] + 2 * (1 - u0) * u0 * ctrl[1] + u0 ** 2 * p1[1]),
        )
        b = (
            int((1 - u1) ** 2 * p0[0] + 2 * (1 - u1) * u1 * ctrl[0] + u1 ** 2 * p1[0]),
            int((1 - u1) ** 2 * p0[1] + 2 * (1 - u1) * u1 * ctrl[1] + u1 ** 2 * p1[1]),
        )
        cv2.line(panel, a, b, color, thickness, cv2.LINE_AA)


def _draw_label_below(
    panel: np.ndarray,
    text: str,
    center_x: int,
    center_y: int,
    radius: int,
    font_scale: float = 0.45,
) -> None:
    label = text[:12]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    lx = center_x - tw // 2
    ly = center_y + radius + th + 6
    cv2.rectangle(panel, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3), LABEL_BG, -1)
    cv2.putText(
        panel, label, (lx, ly),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (230, 230, 230), 1, cv2.LINE_AA,
    )


def _draw_legend(panel: np.ndarray) -> None:
    cv2.line(panel, (8, LEGEND_TOP - 4), (GRAPH_PANEL_WIDTH - 8, LEGEND_TOP - 4), (60, 60, 60), 1)

    col_w = GRAPH_PANEL_WIDTH // 3
    font = 0.38
    y0 = LEGEND_TOP + 18
    dy = 22

    node_swatches = [
        ("normal", STATE_COLORS["normal"]),
        ("moving", STATE_COLORS["moving"]),
        ("hot", STATE_COLORS["hot"]),
        ("broken", STATE_COLORS["broken"]),
    ]
    for i, (name, color) in enumerate(node_swatches):
        x = 12
        y = y0 + i * dy
        cv2.circle(panel, (x + 6, y - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(panel, name, (x + 18, y), cv2.FONT_HERSHEY_SIMPLEX, font, (190, 190, 190), 1, cv2.LINE_AA)

    col2_swatches = [
        ("open", STATE_COLORS["open"]),
        ("cooked", STATE_COLORS["cooked"]),
    ]
    for i, (name, color) in enumerate(col2_swatches):
        x = 12 + col_w
        y = y0 + i * dy
        cv2.circle(panel, (x + 6, y - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(panel, name, (x + 18, y), cv2.FONT_HERSHEY_SIMPLEX, font, (190, 190, 190), 1, cv2.LINE_AA)
    x = 12 + col_w
    y = y0 + 2 * dy
    cv2.circle(panel, (x + 6, y - 4), 5, STATE_COLORS["normal"], -1, cv2.LINE_AA)
    cv2.circle(panel, (x + 6, y - 4), 8, CHANGED_EDGE_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        panel, "detached/changed", (x + 18, y),
        cv2.FONT_HERSHEY_SIMPLEX, font, (190, 190, 190), 1, cv2.LINE_AA,
    )

    edge_items = [
        ("on/in (support)", SUPPORT_COLOR, False),
        ("hazard", HAZARD_COLOR, False),
        ("severed (lost)", SEVERED_COLOR, True),
    ]
    for i, (name, color, dashed) in enumerate(edge_items):
        x = 12 + 2 * col_w
        y = y0 + i * dy
        if dashed:
            _draw_dashed_line(panel, (x, y - 4), (x + 22, y - 4), color, 2)
        else:
            cv2.line(panel, (x, y - 4), (x + 22, y - 4), color, 2, cv2.LINE_AA)
        cv2.putText(panel, name, (x + 28, y), cv2.FONT_HERSHEY_SIMPLEX, font, (190, 190, 190), 1, cv2.LINE_AA)


def _draw_passage_legend(panel: np.ndarray) -> None:
    cv2.line(panel, (8, LEGEND_TOP - 4), (GRAPH_PANEL_WIDTH - 8, LEGEND_TOP - 4), (60, 60, 60), 1)
    font = 0.38
    y0 = LEGEND_TOP + 18
    dy = 22
    items = [
        ("pending", PENDING_COLOR, "hollow"),
        ("in walkway", HAZARD_COLOR, "fill"),
        ("above walkway", ABOVE_WALKWAY_COLOR, "dash"),
        ("open", PASSAGE_BAR_COLORS["open"], "bar"),
        ("constrained", PASSAGE_BAR_COLORS["constrained"], "bar"),
        ("blocked", PASSAGE_BAR_COLORS["blocked"], "bar"),
    ]
    col_w = GRAPH_PANEL_WIDTH // 2
    for i, (name, color, kind) in enumerate(items):
        col = i // 3
        row = i % 3
        x = 12 + col * col_w
        y = y0 + row * dy
        if kind == "hollow":
            cv2.circle(panel, (x + 6, y - 4), 5, color, 1, cv2.LINE_AA)
        elif kind == "fill":
            cv2.circle(panel, (x + 6, y - 4), 5, color, -1, cv2.LINE_AA)
        elif kind == "dash":
            _draw_dashed_line(panel, (x, y - 4), (x + 12, y - 4), color, 2)
        else:
            cv2.rectangle(panel, (x, y - 8), (x + 12, y), color, -1)
        cv2.putText(panel, name, (x + 18, y), cv2.FONT_HERSHEY_SIMPLEX, font, (190, 190, 190), 1, cv2.LINE_AA)


def render_passage_panel(
    event,
    blocker_ids: list[str],
    placed_ids: list[str],
    viewpoint: dict[str, float],
    view_yaw: float,
    passage_state: str,
    report: dict[str, Any],
    height: int,
) -> np.ndarray:
    width = GRAPH_PANEL_WIDTH
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    placed_set = set(placed_ids)
    clearance = compute_passage_clearance(event, viewpoint, view_yaw, placed_ids)
    labels = _blocker_labels(blocker_ids, event)

    here_x, ahead_x = 70, width - 70
    bar_y = 130
    bar_x0, bar_x1 = 100, width - 100
    bar_w = bar_x1 - bar_x0
    bar_h = 22
    bar_center = ((bar_x0 + bar_x1) // 2, bar_y + bar_h // 2)

    cv2.putText(panel, "PASSAGE", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    cv2.circle(panel, (here_x, bar_y + bar_h // 2), 12, STATE_COLORS["normal"], -1, cv2.LINE_AA)
    _draw_label_below(panel, "HERE", here_x, bar_y + bar_h // 2, 12, font_scale=0.4)
    cv2.circle(panel, (ahead_x, bar_y + bar_h // 2), 12, PASSAGE_BAR_COLORS.get(passage_state, STATE_COLORS["normal"]), -1, cv2.LINE_AA)
    _draw_label_below(panel, "AHEAD", ahead_x, bar_y + bar_h // 2, 12, font_scale=0.4)
    cv2.line(panel, (here_x + 14, bar_y + bar_h // 2), (bar_x0 - 6, bar_y + bar_h // 2), SUPPORT_COLOR, 2, cv2.LINE_AA)
    cv2.line(panel, (bar_x1 + 6, bar_y + bar_h // 2), (ahead_x - 14, bar_y + bar_h // 2), SUPPORT_COLOR, 2, cv2.LINE_AA)

    cv2.rectangle(panel, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (40, 40, 40), -1)
    cv2.rectangle(panel, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (80, 80, 80), 1)

    def lat_to_px(lat_m: float) -> int:
        frac = (lat_m + LATERAL_SPAN_M) / (2.0 * LATERAL_SPAN_M)
        return int(bar_x0 + frac * bar_w)

    for lo, hi in clearance["intervals"]:
        x0 = lat_to_px(lo)
        x1 = lat_to_px(hi)
        cv2.rectangle(panel, (x0, bar_y + 2), (x1, bar_y + bar_h - 2), BLOCKED_BAR_COLOR, -1)

    gap_m = float(clearance["gap_m"])
    gap_label = f"gap={gap_m:.2f}m"
    cv2.putText(panel, gap_label, (bar_x0, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, FREE_COLOR, 1, cv2.LINE_AA)

    blocker_rows = [blocker_ids[i : i + 3] for i in range(0, len(blocker_ids), 3)]
    blocker_rows = blocker_rows[:2]
    row_ys = [220, 290]
    for row_nodes, row_y in zip(blocker_rows, row_ys):
        xs = _row_x_centers(len(row_nodes), width)
        for oid, bx in zip(row_nodes, xs):
            label = labels.get(oid, "?")
            if oid not in placed_set:
                cv2.circle(panel, (bx, row_y), 10, PENDING_COLOR, 1, cv2.LINE_AA)
            elif oid in clearance["above_walkway_ids"]:
                cv2.circle(panel, (bx, row_y), 10, ABOVE_WALKWAY_COLOR, -1, cv2.LINE_AA)
                _draw_dashed_line(panel, (bx, row_y - 10), bar_center, ABOVE_WALKWAY_COLOR, 1)
                tag = "above"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
                cv2.putText(panel, tag, (bx - tw // 2, row_y - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA)
            else:
                cv2.circle(panel, (bx, row_y), 10, HAZARD_COLOR, -1, cv2.LINE_AA)
                cv2.line(panel, (bx, row_y - 10), bar_center, HAZARD_COLOR, 2, cv2.LINE_AA)
            _draw_label_below(panel, label, bx, row_y, 10, font_scale=0.42)

    placed_count = int(report.get("placed_count", len(placed_ids)))
    total = max(1, len(blocker_ids))
    move_ok = report.get("moveahead_ok")
    move_str = "True" if move_ok else "False" if move_ok is not None else "?"
    readout = f"placed={placed_count}/{total}  {gap_label}  MoveAhead={move_str}  state={passage_state}"
    (rw, _), _ = cv2.getTextSize(readout, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(
        panel, readout, (max(8, width - rw - 8), 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA,
    )

    if height >= LEGEND_TOP + 80:
        _draw_passage_legend(panel)
    return panel


def render_graph_panel(
    graph: SceneGraph,
    layout: dict[str, tuple[int, int]],
    height: int,
    hazard_edges: list[GraphEdge] | None = None,
    changed_ids: set[str] | None = None,
    hot_ids: set[str] | None = None,
) -> np.ndarray:
    width = GRAPH_PANEL_WIDTH
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    hazard_edges = hazard_edges or []
    changed_ids = changed_ids or set()
    hot_ids = hot_ids or set()
    all_edges = graph.edges + hazard_edges

    support_count = sum(1 for e in graph.edges if e.kind == "support")
    spatial_count = sum(1 for e in graph.edges if e.kind == "spatial")
    drawn_hazard = sum(
        1 for e in hazard_edges
        if not (e.subject_id == e.object_id and e.predicate in ("ignited", "broken"))
    )
    severed_count = sum(1 for e in hazard_edges if e.kind == "severed")

    def to_px(nid: str) -> tuple[int, int]:
        return layout.get(nid, (width // 2, (GRAPH_AREA_TOP + GRAPH_AREA_BOTTOM) // 2))

    for edge in all_edges:
        if edge.kind == "spatial":
            continue
        if edge.subject_id not in layout or edge.object_id not in layout:
            continue
        if edge.subject_id == edge.object_id and edge.predicate in ("ignited", "broken"):
            continue
        p0, p1 = to_px(edge.subject_id), to_px(edge.object_id)
        if edge.kind == "hazard":
            cv2.line(panel, p0, p1, HAZARD_COLOR, 2, cv2.LINE_AA)
        elif edge.kind == "severed":
            _draw_dashed_line(panel, p0, p1, SEVERED_COLOR, 2)
        else:
            _draw_arc_line(panel, p0, p1, SUPPORT_COLOR, 2)

    node_radius = 10
    for nid, node in graph.nodes.items():
        if nid not in layout:
            continue
        px, py = to_px(nid)
        states = list(node.attributes.get("state") or ["normal"])
        if nid in hot_ids and "hot" not in states and "broken" not in states:
            states = ["hot"] + [s for s in states if s != "normal"]
        color = _node_color(states, hot_ids=hot_ids, nid=nid)
        cv2.circle(panel, (px, py), node_radius, color, -1, cv2.LINE_AA)
        if nid in changed_ids:
            cv2.circle(panel, (px, py), node_radius + 4, CHANGED_EDGE_COLOR, 2, cv2.LINE_AA)
        _draw_label_below(panel, node.label, px, py, node_radius)

    cv2.putText(
        panel, "SCENE GRAPH", (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
    )
    readout = (
        f"nodes={len(graph.nodes)} support={support_count} "
        f"hazard={drawn_hazard + severed_count} spatial={spatial_count} (hidden)"
    )
    (rw, _), _ = cv2.getTextSize(readout, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(
        panel, readout, (max(8, width - rw - 8), 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA,
    )

    if height >= LEGEND_TOP + 80:
        _draw_legend(panel)
    return panel
