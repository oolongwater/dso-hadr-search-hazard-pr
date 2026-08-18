#!/usr/bin/env python3
"""Scene Action Map demo: FPV + top-down SAM panel with earthquake severed edges."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.demo_earthquake_multiroom import add_overhead_camera_at_floor, doorway_blocked, paint_debris_on_mask
from cli.hazard_scenes import SCENE_CAPTURE_PROFILES, _label_lines, _shift_fpv
from core.procthor_house import (
    default_local_executable,
    door_world_center,
    door_world_frame,
    doorway_corridor_rect,
    house_floors,
    house_schema,
    load_house_json,
    make_procedural_controller,
    map_projection_from_props,
    push_toward_point,
    reachable_on_floor,
    room_adjacency,
    room_centroid,
    room_containing,
    stage_doorway_blockers,
    world_to_map_px,
)
from core.changepoint import Changepoint, ChangepointLog, load_changepoints
from core.thor import snap_to_reachable
from core.scene_action_map import (
    ControllerLabel,
    DecisionDiff,
    SceneActionMap,
    SamNode,
    CLUTTER_RADIUS_M,
    MIN_CLUSTER_OBJECTS,
    agent_radius_px,
    build_reachable_free_mask,
    build_scene_action_map,
    apply_route_relative_behaviours,
    behaviour_display,
    changepoint_from_sam_node,
    controller_labels,
    decision_diff,
    decision_display,
    decision_frame_text,
    map_px_to_world,
    pick_designed_route,
    pick_display_route,
    plan_route,
    recompute_edges_from_mask,
    render_decision_card,
    render_node_payload_png,
    render_nodes_html,
    render_sam_panel,
    draw_agent_marker,
    replan_from,
    route_behaviours,
    serialize_sam_graph,
    sever_edges_by_objects,
)
from core.thor import (
    agent_pose,
    distance_xz,
    follow_path_discrete,
    get_reachable_positions,
    get_shortest_path,
    rotate_toward,
    teleport,
    yaw_toward,
)
from core.video import finalize_mp4, probe_video, rgb_to_bgr_uint8
from hazard.functions import EarthquakeLatents, HazardConfig
from hazard.model import EarthquakeHazard
from hazard.utils import advance_physics, hazard_output_dir, object_state_snapshot, pause_physics, unpause_physics

NAV_STEP_M = 0.05
NAV_ROTATE_DEG = 5.0
NAV_CAPTURE_EVERY = 1
NAV_FRAME_REPEAT = 1
WAYPOINT_MIN_SPACING_M = 0.25
NAV_MAX_WAYPOINTS = 200
NAV_MAX_STEPS_PER_PHASE = 800
VIEWPOINT_BAND_M = 1.0
SEVER_DELAY_S = 5.0
TOPPLE_LEAD_M = 2.0
REPLAN_LEAD_M = 1.0
D0_REACHED_M = 0.35
CP_TRIGGER_M = 2.5
CP_ARRIVE_M = 0.45
CP_HUD_HOLD_FRAMES = 45
PAUSE_SECONDS = 3.0
SCAN_VIEWS = 4
SCAN_YAW_STEP = 90.0
TOPPLE_SHOVE_MIN_S = 3.0
DECISION_CARD_WIDTH = 460
TAIL_FRAMES = 16
PANEL_HEIGHT = 800
CORRIDOR_DEPTH_M = 0.45
TOPPLE_PUSH = 6.0
TOPPLE_PUSH_RADIUS = 1.5
QUAKE_ONSET_M = 0.5
BLOCK_LEG_MIN_M = 1.5
HAZARD_TICK_EVERY = 8
BATCH_HOUSE_DIR = Path(__file__).resolve().parents[1] / "assets" / "houses" / "batch"
OUT_SUBDIR = "sam"
SETTLE_PHYSICS_STEPS = 24
SETTLE_PHYSICS_DT = 1.0 / 30.0


class _RouteInvalidated(Exception):
    """Raised from nav callback when doorway block forces replan mid-walk."""


def _world_bearing_deg(
    proj: dict[str, Any],
    x0: float,
    z0: float,
    x1: float,
    z1: float,
) -> float:
    u0, v0 = world_to_map_px(x0, z0, proj)
    u1, v1 = world_to_map_px(x1, z1, proj)
    dx = u1 - u0
    dy = v1 - v0
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def _ground_floor_base_y(house: dict[str, Any]) -> float:
    floors = house_floors(house)
    ground = next((f for f in floors if int(f.get("index", 0)) == 0), floors[0])
    return float(ground.get("baseY", 0.0))


def _settle_scene_physics(controller) -> None:
    unpause_physics(controller)
    for _ in range(SETTLE_PHYSICS_STEPS):
        advance_physics(controller, SETTLE_PHYSICS_DT)
    pause_physics(controller)


def _door_by_id(house: dict[str, Any], door_id: str) -> dict[str, Any]:
    for door in house.get("doors") or []:
        if str(door.get("id")) == door_id:
            return door
    raise ValueError(f"door not found: {door_id}")


def _cp_for_door(sam: SceneActionMap, door_id: str) -> str:
    for nid, node in sam.nodes.items():
        if node.kind == "changepoint" and node.door_id == door_id:
            return nid
    raise ValueError(f"no changepoint for door {door_id}")


def _pick_ring_demo_route(
    sam: SceneActionMap,
    start_id: str,
    goal_id: str,
    door_sequence: list[str],
) -> list[str]:
    """Explicit changepoint chain: start -> CP(door0) -> CP(door1) -> CP(door2) -> goal."""
    try:
        cp_ids = [_cp_for_door(sam, door_id) for door_id in door_sequence]
    except ValueError:
        return []
    route = [start_id, *cp_ids, goal_id]
    full = plan_route(sam, start_id, goal_id)
    if len(full) < 2:
        return []
    pos = 0
    for cp_id in cp_ids:
        try:
            idx = full.index(cp_id, pos)
        except ValueError:
            return []
        pos = idx + 1
    if full[-1] != goal_id:
        return []
    return route


def _probe_alternate_exists(
    sam: SceneActionMap,
    block_cp: str,
    start_id: str,
    goal_id: str,
    route: list[str],
) -> bool:
    if block_cp not in route:
        return False
    idx = route.index(block_cp)
    prev_id = route[idx - 1] if idx > 0 else start_id
    edge = _edge_between_any(sam, prev_id, block_cp)
    if edge is None:
        edge = _edge_between_any(sam, block_cp, prev_id)
    if edge is None:
        return False
    saved_alive, saved_reason = edge.alive, edge.block_reason
    edge.alive = False
    edge.block_reason = "probe"
    alt, _ = replan_from(sam, prev_id, goal_id)
    edge.alive = saved_alive
    edge.block_reason = saved_reason
    return len(alt) >= 2


def _far_destination(
    house: dict[str, Any],
    agent_x: float,
    agent_z: float,
    reachable: list[dict[str, float]],
) -> tuple[str, float, float]:
    """Pick the room farthest by door hops from the agent's current room."""
    start_room = room_containing(house, agent_x, agent_z)
    adj = room_adjacency(house)
    if start_room is None or not adj:
        return ("goal", agent_x + 2.0, agent_z + 2.0)

    start_id = str(start_room.get("id") or "")
    rooms_by_id = {str(r.get("id")): r for r in (house.get("rooms") or []) if r.get("id")}

    # BFS: track door-hop distance from start room.
    hop_dist: dict[str, int] = {start_id: 0}
    queue = [start_id]
    while queue:
        cur = queue.pop(0)
        for nb in adj.get(cur, {}):
            if nb not in hop_dist:
                hop_dist[nb] = hop_dist[cur] + 1
                queue.append(nb)

    candidates = [
        (rid, hops)
        for rid, hops in hop_dist.items()
        if rid != start_id and rid in rooms_by_id
    ]
    if not candidates:
        rc = room_centroid(start_room)
        return (start_id, float(rc["x"]), float(rc["z"]))

    max_hops = max(h for _, h in candidates)
    farthest = [rid for rid, h in candidates if h == max_hops]
    best_id = max(
        farthest,
        key=lambda rid: math.hypot(
            float(room_centroid(rooms_by_id[rid])["x"]) - agent_x,
            float(room_centroid(rooms_by_id[rid])["z"]) - agent_z,
        ),
    )
    rc = room_centroid(rooms_by_id[best_id])
    gx, gz = float(rc["x"]), float(rc["z"])
    snap = snap_to_reachable(gx, gz, reachable, max_dist=3.0)
    if snap:
        return (best_id, snap[0], snap[2])
    return (best_id, gx, gz)


def _compose_multiview_tile(
    views_bgr: list[np.ndarray],
    *,
    node_id: str = "",
    view_yaws: list[float] | None = None,
) -> np.ndarray:
    if not views_bgr:
        return np.zeros((390, 520, 3), dtype=np.uint8)
    cell_h, cell_w = 195, 260
    tile = np.zeros((cell_h * 2, cell_w * 2, 3), dtype=np.uint8)
    yaws = view_yaws or [0.0, 90.0, 180.0, 270.0]
    for i, view in enumerate(views_bgr[:4]):
        r, c = divmod(i, 2)
        resized = cv2.resize(view, (cell_w, cell_h))
        y0, x0 = r * cell_h, c * cell_w
        tile[y0 : y0 + cell_h, x0 : x0 + cell_w] = resized
        cv2.rectangle(tile, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1), (0, 255, 255), 2)
        tag = f"{node_id} yaw {yaws[i]:.0f} deg" if node_id else f"yaw {yaws[i]:.0f} deg"
        cv2.putText(
            tile,
            tag,
            (x0 + 8, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return tile


def _compose_sam_frame(
    event,
    *,
    fpv_shift: tuple[int, int],
    lines: list[str],
    sam_panel: np.ndarray,
    decision_card: np.ndarray | None = None,
    hud: ControllerLabel | None = None,
    multiview: np.ndarray | None = None,
) -> np.ndarray | None:
    if event.frame is None:
        return None
    fpv_bgr = rgb_to_bgr_uint8(event.frame)
    if fpv_shift != (0, 0):
        fpv_bgr = _shift_fpv(fpv_bgr, fpv_shift)
    h = PANEL_HEIGHT
    scale = h / fpv_bgr.shape[0]
    fpv_panel = cv2.resize(fpv_bgr, (int(fpv_bgr.shape[1] * scale), h))
    if multiview is not None:
        mv = multiview
        mh, mw = mv.shape[:2]
        x0 = max(8, fpv_panel.shape[1] - mw - 8)
        y0 = max(8, h - mh - 8)
        fpv_panel[y0 : y0 + mh, x0 : x0 + mw] = mv
        cv2.rectangle(fpv_panel, (x0 - 2, y0 - 2), (x0 + mw + 2, y0 + mh + 2), (0, 255, 255), 2)
    right = sam_panel
    if right.shape[0] != h:
        right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    card = decision_card
    if card is None:
        card = render_decision_card(None, None, width=DECISION_CARD_WIDTH, height=h)
    elif card.shape[0] != h:
        card = cv2.resize(card, (DECISION_CARD_WIDTH, h))
    divider = np.full((h, 4, 3), (60, 60, 60), dtype=np.uint8)
    combined = np.hstack([fpv_panel, divider, right, divider, card])
    y = 22
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(combined, (6, y - th - 4), (10 + tw, y + 4), (30, 30, 30), -1)
        cv2.putText(combined, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        y += th + 10
    if hud:
        from core.scene_action_map import _behaviour_display, _node_display_id

        headline = decision_display(hud.decision, target=_node_display_id(hud.node_id))
        if hud.behaviour:
            headline = f"{headline} ({_behaviour_display(hud.behaviour)})"
        fpv_w = fpv_panel.shape[1]
        (tw, th), _ = cv2.getTextSize(headline, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        bx = max(8, (fpv_w - tw) // 2)
        by = h - 36
        cv2.rectangle(combined, (bx - 8, by - th - 10), (bx + tw + 8, by + 10), (0, 0, 0), -1)
        cv2.putText(combined, headline, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        if hud.behaviour:
            cx = fpv_w // 2
            cy = h // 2 + 40
            arrow = {"turn-left": "←", "go-forward": "↑", "turn-right": "→"}.get(hud.behaviour, "•")
            cv2.putText(combined, arrow, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4, cv2.LINE_AA)
    return combined


def _route_band_reachable(
    reachable: list[dict[str, float]],
    waypoints: list[dict[str, float]],
    band_m: float,
) -> list[dict[str, float]]:
    if not waypoints:
        return reachable
    band: list[dict[str, float]] = []
    for p in reachable:
        px, pz = float(p["x"]), float(p["z"])
        if min(distance_xz(px, pz, float(w["x"]), float(w["z"])) for w in waypoints) <= band_m:
            band.append(p)
    return band


def _mark_preset_cps_through_block(
    sam: SceneActionMap,
    route: list[str],
    visited: set[str],
    block_cp: str,
) -> None:
    for nid in route:
        node = sam.nodes.get(nid)
        if node is None or node.kind != "changepoint":
            continue
        visited.add(nid)
        if nid == block_cp:
            break


def _mark_route_cp_visits(
    sam: SceneActionMap,
    x: float,
    z: float,
    route: list[str],
    visited: set[str],
    *,
    trigger_m: float,
) -> None:
    for nid in route:
        node = sam.nodes.get(nid)
        if node is None or node.kind != "changepoint":
            continue
        if distance_xz(x, z, node.world[0], node.world[1]) <= trigger_m:
            visited.add(nid)


def _nearest_route_changepoint(
    sam: SceneActionMap,
    x: float,
    z: float,
    route: list[str],
) -> tuple[str | None, float]:
    best_id: str | None = None
    best_d = float("inf")
    for nid in route:
        node = sam.nodes.get(nid)
        if node is None or node.kind != "changepoint":
            continue
        d = distance_xz(x, z, node.world[0], node.world[1])
        if d < best_d:
            best_d = d
            best_id = nid
    return best_id, best_d


def _label_for_node(labels: list[ControllerLabel], node_id: str) -> ControllerLabel | None:
    for lb in labels:
        if lb.node_id == node_id and lb.behaviour:
            return lb
    return None


def _dedupe_waypoints(
    waypoints: list[dict[str, float]],
    min_dist: float = WAYPOINT_MIN_SPACING_M,
) -> list[dict[str, float]]:
    if len(waypoints) <= 1:
        return waypoints
    out = [waypoints[0]]
    for wp in waypoints[1:]:
        last = out[-1]
        if distance_xz(float(last["x"]), float(last["z"]), float(wp["x"]), float(wp["z"])) >= min_dist:
            out.append(wp)
    if out[-1] != waypoints[-1]:
        out.append(waypoints[-1])
    return out


def _limit_waypoints(waypoints: list[dict[str, float]], *, max_pts: int = NAV_MAX_WAYPOINTS) -> list[dict[str, float]]:
    if len(waypoints) <= max_pts:
        return waypoints
    stride = max(1, len(waypoints) // max_pts)
    trimmed = waypoints[::stride]
    if trimmed[-1] != waypoints[-1]:
        trimmed.append(waypoints[-1])
    return trimmed


def _edge_between(sam: SceneActionMap, a: str, b: str):
    for e in sam.edges:
        if e.src == a and e.dst == b:
            return e
    return None


def _navmesh_waypoints(
    controller,
    target_x: float,
    target_z: float,
    *,
    agent_y: float | None = None,
    retries: int = 3,
    settle_each_retry: bool = False,
) -> list[dict[str, float]] | None:
    for attempt in range(retries):
        if settle_each_retry and attempt > 0:
            for _ in range(2):
                controller.step(action="Pass")
            advance_physics(controller)
        path, _ = get_shortest_path(controller, target_x, target_z, y=agent_y)
        if path and len(path) >= 2:
            ay = float(path[0]["y"]) if agent_y is None else float(agent_y)
            return [{"x": float(p["x"]), "y": ay, "z": float(p["z"])} for p in path]
        if attempt < retries - 1:
            controller.step(action="Pass")
    return None


def _route_world_waypoints(
    sam: SceneActionMap,
    proj: dict[str, Any],
    route: list[str],
    *,
    agent_y: float,
    end_idx: int | None = None,
) -> list[dict[str, float]]:
    """Flatten route edge polylines into world waypoints."""
    end = len(route) - 1 if end_idx is None else min(end_idx, len(route) - 1)
    waypoints: list[dict[str, float]] = []
    for i in range(end):
        a, b = route[i], route[i + 1]
        edge = _edge_between(sam, a, b)
        if edge is None or not edge.path_px or len(edge.path_px) < 2:
            continue
        path_px = edge.path_px
        for j, (u, v) in enumerate(path_px):
            if i > 0 and j == 0:
                continue
            wx, wz = map_px_to_world(float(u), float(v), proj)
            waypoints.append({"x": wx, "y": agent_y, "z": wz})
    return waypoints


def _nearest_node(sam: SceneActionMap, agent_px: tuple[int, int]) -> str:
    best_id = ""
    best_d = float("inf")
    for nid, node in sam.nodes.items():
        d = math.hypot(node.px[0] - agent_px[0], node.px[1] - agent_px[1])
        if d < best_d:
            best_d = d
            best_id = nid
    return best_id


def _sever_target_doorway(
    sam: SceneActionMap,
    event,
    object_ids: list[str],
    proj: dict[str, Any],
    door_id: str,
    block_cp: str,
) -> None:
    """Sever only edges incident to the target doorway that debris blocks."""
    sever_edges_by_objects(sam, event, object_ids, proj)
    for edge in sam.edges:
        if edge.alive:
            continue
        src = sam.nodes.get(edge.src)
        dst = sam.nodes.get(edge.dst)
        door_ids = {getattr(src, "door_id", None), getattr(dst, "door_id", None)}
        if door_id not in door_ids:
            edge.alive = True
            edge.block_reason = ""
    # ponytail: debris bbox may miss narrow door-edge tubes; sever CP edges when passage blocked
    if not any(not e.alive for e in sam.edges if e.src == block_cp or e.dst == block_cp):
        for edge in sam.edges:
            if edge.alive and (edge.src == block_cp or edge.dst == block_cp):
                edge.alive = False
                edge.block_reason = "doorway_blocked"


def _connected_changepoint_component(sam: SceneActionMap) -> bool:
    cps = [nid for nid, n in sam.nodes.items() if n.kind == "changepoint"]
    if len(cps) <= 1:
        return True
    adj: dict[str, set[str]] = {c: set() for c in cps}
    for e in sam.edges:
        if not e.alive:
            continue
        if e.src in adj and e.dst in adj:
            adj[e.src].add(e.dst)
            adj[e.dst].add(e.src)
    seen = {cps[0]}
    stack = [cps[0]]
    while stack:
        cur = stack.pop()
        for nb in adj.get(cur, ()):
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cps)


def _edge_between_any(sam: SceneActionMap, a: str, b: str):
    for e in sam.edges:
        if e.src == a and e.dst == b:
            return e
    return None


def _edge_world_waypoints(
    sam: SceneActionMap,
    proj: dict[str, Any],
    src_id: str,
    dst_id: str,
    *,
    agent_y: float,
    reverse: bool = False,
) -> list[dict[str, float]]:
    edge = _edge_between_any(sam, src_id, dst_id)
    path_reverse = reverse
    if edge is None:
        edge = _edge_between_any(sam, dst_id, src_id)
        path_reverse = not reverse
    if edge is None or not edge.path_px or len(edge.path_px) < 2:
        return []
    path_px = edge.path_px
    if path_reverse:
        path_px = list(reversed(path_px))
    waypoints: list[dict[str, float]] = []
    for u, v in path_px:
        wx, wz = map_px_to_world(float(u), float(v), proj)
        waypoints.append({"x": wx, "y": agent_y, "z": wz})
    return _limit_waypoints(_dedupe_waypoints(waypoints))


def run_sam_earthquake_demo(
    controller,
    config: HazardConfig,
    *,
    house: dict[str, Any],
    out_video: Path,
    map_png: Path,
    graph_json: Path,
    nodes_dir: Path,
    nodes_html: Path,
    changepoints_json: Path,
    house_json: str,
    grid_res_px: float,
    cp_threshold: float,
    min_sep_m: float,
) -> dict[str, Any]:
    profile = dict(SCENE_CAPTURE_PROFILES["earthquake"])
    fps = int(profile["fps"])
    substeps = int(profile["substeps"])
    time_step = float(profile["time_step"])

    base_y = _ground_floor_base_y(house)
    _settle_scene_physics(controller)
    add_overhead_camera_at_floor(controller, base_y)
    pass_event = controller.step(action="Pass")
    tpf = getattr(pass_event, "third_party_camera_frames", None)
    if not tpf:
        raise RuntimeError("overhead camera produced no third_party_camera_frames")
    oh_h, oh_w = tpf[0].shape[:2]

    map_props_event = controller.step(action="GetMapViewCameraProperties")
    map_props = map_props_event.metadata.get("actionReturn") or {}
    proj = map_projection_from_props(map_props, oh_w, oh_h)

    reachable = get_reachable_positions(controller)
    if house_schema(house) == "2.0.0":
        reachable = reachable_on_floor(reachable, base_y)

    h = int(proj["image_height"])
    w = int(proj["image_width"])
    free_open = build_reachable_free_mask(reachable, proj, h, w)
    radius_px = agent_radius_px(proj)

    ax, az, ayaw = agent_pose(controller.last_event)
    agent_y = float(controller.last_event.metadata["agent"]["position"]["y"])
    rooms_by_id = {str(r.get("id")): r for r in (house.get("rooms") or []) if r.get("id")}

    target_door_id = "door|5|6"
    door = _door_by_id(house, target_door_id)
    door_frame = door_world_frame(door, house)
    door_center = door_world_center(door, house)
    corridor = doorway_corridor_rect(door_frame, depth_m=CORRIDOR_DEPTH_M)
    stage_room = rooms_by_id.get("room|5")
    if stage_room is None:
        raise RuntimeError("room|5 required for doorway staging on ring layout")
    staged_ids = stage_doorway_blockers(
        controller, door_frame, stage_room, door_id=target_door_id
    )

    objects = list(controller.last_event.metadata.get("objects") or [])
    if "room|7" in rooms_by_id:
        rc = room_centroid(rooms_by_id["room|7"])
        dest_label, dest_x, dest_z = "room|7", float(rc["x"]), float(rc["z"])
    else:
        dest_label, dest_x, dest_z = _far_destination(house, ax, az, reachable)
    sam = build_scene_action_map(
        free_open,
        proj,
        grid_res_px=grid_res_px,
        cp_threshold=cp_threshold,
        min_sep_m=min_sep_m,
        house=house,
        destinations=[("start", (ax, az)), (dest_label, (dest_x, dest_z))],
        agent_radius_px=radius_px,
        objects=objects,
        world_y=agent_y,
    )

    start_id = "dest_start"
    goal_id = f"dest_{dest_label}"
    if start_id not in sam.nodes:
        start_id = next((nid for nid, n in sam.nodes.items() if n.kind == "destination"), "")
    cp_ids = [
        nid
        for nid, node in sam.nodes.items()
        if node.kind == "changepoint"
    ]
    cp_ids.sort(key=lambda nid: sam.nodes[nid].score, reverse=True)
    preferred_route = [start_id, *cp_ids[:3], goal_id]
    if len(cp_ids) >= 2 and all(nid in sam.nodes for nid in preferred_route):
        route_before = preferred_route
    else:
        route_before = pick_designed_route(sam, start_id, goal_id, min_cps=2)
    if len(route_before) < 2:
        route_before = _pick_ring_demo_route(
            sam,
            start_id,
            goal_id,
            ["door|4|5", "door|5|6", "door|6|7"],
        )
    if len(route_before) < 2:
        route_before = pick_display_route(sam, start_id, goal_id)
    try:
        d0_id = _cp_for_door(sam, "door|4|5")
    except ValueError:
        d0_id = next((nid for nid in route_before if sam.nodes[nid].kind == "changepoint"), "")
    labels_before = controller_labels(sam, route_before)
    labels_before = apply_route_relative_behaviours(sam, route_before, labels_before)
    behaviours = route_behaviours(sam, route_before)
    connected_cps_pre = _connected_changepoint_component(sam)
    d0_world = sam.nodes[d0_id].world if d0_id in sam.nodes else (ax, az)

    route_cps = [nid for nid in route_before if sam.nodes[nid].kind == "changepoint"]
    if len(route_cps) < 2 and len(cp_ids) >= 2:
        route_before = [start_id, *cp_ids[: min(3, len(cp_ids))], goal_id]
        route_cps = [nid for nid in route_before if sam.nodes[nid].kind == "changepoint"]
    if len(route_cps) < 2:
        raise RuntimeError(f"designed route needs >=2 changepoints, got {route_cps}")
    try:
        block_cp = _cp_for_door(sam, target_door_id)
    except ValueError:
        block_cp = route_cps[len(route_cps) // 2]
    if block_cp not in route_before:
        raise RuntimeError(f"designed route must pass through {block_cp}, got {route_before}")
    alternate_exists = _probe_alternate_exists(sam, block_cp, start_id, goal_id, route_before)
    if not alternate_exists:
        block_idx = route_before.index(block_cp)
        prev_id = route_before[block_idx - 1] if block_idx > 0 else start_id
        alt_route, _ = replan_from(sam, prev_id, goal_id)
        alternate_exists = len(alt_route) >= 2

    pre_quake_pose = controller.last_event.metadata["agent"]
    pre_pos = pre_quake_pose["position"]
    pre_rot = pre_quake_pose["rotation"]
    pre_horizon = float(pre_quake_pose.get("cameraHorizon", 0.0))

    route_wps = _route_world_waypoints(sam, proj, route_before, agent_y=agent_y)
    route_band = _route_band_reachable(reachable, route_wps, VIEWPOINT_BAND_M)
    setup_reachable = route_band if route_band else reachable

    hazard = EarthquakeHazard(config)
    hazard.setup(controller, reachable=setup_reachable, floor_base_y=base_y)
    teleport(
        controller,
        float(pre_pos["x"]),
        float(pre_pos["y"]),
        float(pre_pos["z"]),
        yaw=float(pre_rot["y"]),
        horizon=pre_horizon,
    )

    out_video.parent.mkdir(parents=True, exist_ok=True)
    nodes_dir.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    cp_hud_frames = 0
    cp_visited: set[str] = set()
    scanned_cps: set[str] = set()
    decisions_timeline: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    peak_moving = 0
    agent_path_m = 0.0
    post_quake_path_m = 0.0
    backtrack_m = 0.0
    max_frame_jump_m = 0.0
    prev_ax, prev_az, _ = agent_pose(controller.last_event)
    last_cap_ax, last_cap_az = prev_ax, prev_az
    walk_start_path = 0.0
    post_walk_start_path = 0.0
    post_walk_start_x = 0.0
    post_walk_start_z = 0.0
    route_after: list[str] = []
    labels_after: list[ControllerLabel] = list(labels_before)
    decision_diffs: list[DecisionDiff] = []
    blocked_final = free_open
    passage_blocked = False
    blocking_object_ids: list[str] = []
    sever_applied = False
    backtrack_done = False
    reroute_done = False
    quake_started = False
    quake_nav_frames = 0
    hazard_step = config.onset_step
    shake_seconds_before_sever = 0.0
    replan_dist_to_d0_m = 0.0
    max_retreat_before_d0_m = 0.0
    min_dist_to_d0_m = float("inf")
    d0_reached = False
    d0_decision_after = ""
    hud_state: dict[str, Any] = {"active_id": None, "hold": 0, "label": None}
    panel_cache: dict[str, Any] = {"key": None, "base": None}
    active_route = list(route_before)
    active_labels = list(labels_before)
    active_mask = free_open
    nav_frame = {"i": 0}
    nav_context = {"on_block_leg": False, "block_leg_start_m": 0.0, "block_leg_entered": False}
    leg_stats: list[dict[str, Any]] = []
    post_walk_attempted = False
    block_world = sam.nodes[block_cp].world
    cp_log = ChangepointLog(changepoints_json, label=config.scene, house_json=house_json)
    active_cp: Changepoint | None = None
    cp_pause_frames: dict[str, int] = {}

    def _record_decision(
        node_id: str,
        decision: str,
        *,
        behaviour: str = "",
        message: str = "",
        decision_frame: str = "",
    ) -> None:
        node = sam.nodes[node_id]
        node.decision = decision
        frame_txt = decision_frame or decision_frame_text(decision, node)
        node.decision_frame = frame_txt
        ax_now, az_now, ayaw_now = agent_pose(controller.last_event)
        decisions_timeline.append(
            {
                "step": len(decisions_timeline),
                "node_id": node_id,
                "decision": decision,
                "behaviour": behaviour,
                "message": message,
                "decision_frame": frame_txt,
                "world": {"x": node.world[0], "y": node.world_y, "z": node.world[1]},
                "agent_yaw": ayaw_now,
            }
        )

    def _update_hud(labels: list[ControllerLabel], route: list[str]) -> ControllerLabel | None:
        ax_now, az_now, _ = agent_pose(controller.last_event)
        cp_id, dist = _nearest_route_changepoint(sam, ax_now, az_now, route)
        if cp_id and dist <= CP_TRIGGER_M:
            hud_state["active_id"] = cp_id
            hud_state["hold"] = CP_HUD_HOLD_FRAMES
            hud_state["label"] = _label_for_node(labels, cp_id)
            if hud_state["label"] is None:
                node = sam.nodes[cp_id]
                hud_state["label"] = ControllerLabel(
                    node_id=cp_id,
                    behaviour="go-forward",
                    status="open",
                    decision="proceed",
                    message=cp_id,
                )
        elif int(hud_state["hold"]) > 0:
            hud_state["hold"] = int(hud_state["hold"]) - 1
        else:
            hud_state["active_id"] = None
            hud_state["label"] = None
        return hud_state["label"]

    def _track_motion() -> None:
        nonlocal agent_path_m, prev_ax, prev_az
        ax_now, az_now, _ = agent_pose(controller.last_event)
        agent_path_m += distance_xz(prev_ax, prev_az, ax_now, az_now)
        prev_ax, prev_az = ax_now, az_now

    def _current_panel(
        route: list[str],
        labels: list[ControllerLabel],
        mask: np.ndarray,
        *,
        phase: str,
        extra_title: list[str] | None = None,
        cache_stamp: int | None = None,
    ) -> np.ndarray:
        ax_now, az_now, ayaw_now = agent_pose(controller.last_event)
        u, v = world_to_map_px(ax_now, az_now, proj)
        title = [
            f"Scene Action Map ({phase})",
            f"CPs={sum(1 for n in sam.nodes.values() if n.kind=='changepoint')}  route={len(route)}  severed={sum(1 for e in sam.edges if not e.alive)}",
        ]
        if extra_title:
            title.extend(extra_title)
        cache_key = (
            phase,
            cache_stamp,
            tuple(route),
            tuple((d.node_id, d.before, d.after) for d in decision_diffs),
            hud_state.get("active_id"),
            sum(1 for e in sam.edges if not e.alive),
        )
        if panel_cache["key"] != cache_key:
            panel_cache["base"] = render_sam_panel(
                sam,
                mask,
                height=PANEL_HEIGHT,
                route=route,
                route_replan=None,
                agent_px=None,
                title_lines=title,
                controller_labels_list=labels,
                active_node_id=hud_state.get("active_id"),
                decision_diff_list=decision_diffs or None,
                show_graph=True,
            )
            panel_cache["key"] = cache_key
        panel = panel_cache["base"].copy()
        draw_agent_marker(
            panel,
            (int(round(u)), int(round(v))),
            ayaw_now,
            free_open.shape,
        )
        return panel

    def _capture(
        event,
        panel: np.ndarray,
        lines: list[str],
        *,
        shift: tuple[int, int] = (0, 0),
        hud: ControllerLabel | None = None,
        active_node: str | None = None,
        multiview: np.ndarray | None = None,
    ) -> None:
        nonlocal writer, frames_written, cp_hud_frames
        node = sam.nodes.get(active_node) if active_node else None
        if node is None and hud is not None:
            node = sam.nodes.get(hud.node_id)
        label = hud or (_label_for_node(active_labels, active_node) if active_node else None)
        cp_for_card = active_cp if active_cp and active_node == active_cp.id else None
        card = render_decision_card(
            node,
            label,
            width=DECISION_CARD_WIDTH,
            height=PANEL_HEIGHT,
            cp=cp_for_card,
        )
        frame = _compose_sam_frame(
            event,
            fpv_shift=shift,
            lines=lines,
            sam_panel=panel,
            decision_card=card,
            hud=hud or label,
            multiview=multiview,
        )
        if frame is None:
            return
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fh, fw = frame.shape[:2]
            writer = cv2.VideoWriter(str(out_video), fourcc, fps, (fw, fh))
        for _ in range(NAV_FRAME_REPEAT):
            writer.write(frame)
            frames_written += 1
            if hud is not None or label is not None:
                cp_hud_frames += 1

    def _pause_and_scan(
        node_id: str,
        label: ControllerLabel,
        lines: list[str],
        phase: str,
    ) -> None:
        nonlocal active_cp, hazard_step, quake_nav_frames, passage_blocked, blocking_object_ids
        node = sam.nodes[node_id]
        cp_visited.add(node_id)
        scanned_cps.add(node_id)
        start_yaw = float(controller.last_event.metadata["agent"]["rotation"]["y"])
        view_paths: list[str] = []
        view_frames: list[np.ndarray] = []
        view_yaws: list[float] = []
        ax_now, az_now, ayaw_now = agent_pose(controller.last_event)
        ay_now = float(controller.last_event.metadata["agent"]["position"]["y"])
        shake_el = quake_nav_frames / float(fps) if quake_started else 0.0
        cp = changepoint_from_sam_node(
            node,
            sam,
            label=label,
            phase=phase,
            agent={"x": ax_now, "y": ay_now, "z": az_now, "yaw": ayaw_now},
            agent_path_m=agent_path_m,
            quake_active=quake_started and not sever_applied,
            shake_elapsed_s=shake_el,
            visit_index=len(cp_log.records),
        )
        active_cp = cp

        total_frames = round(PAUSE_SECONDS * fps)
        segments = 1 + SCAN_VIEWS
        frames_per_seg = max(1, total_frames // segments)
        captured = 0

        pause_physics(controller)
        try:
            panel = _current_panel(active_route, active_labels, active_mask, phase=phase)
            for _ in range(frames_per_seg):
                event = advance_physics(controller, time_step)
                if quake_started:
                    quake_nav_frames += 1
                    if captured % HAZARD_TICK_EVERY == 0:
                        hazard.tick(controller, hazard_step)
                        hazard_step += 1
                _capture(event, panel, lines, hud=label, active_node=node_id)
                captured += 1

            for i in range(SCAN_VIEWS):
                if i > 0:
                    event = controller.step(action="RotateRight", degrees=SCAN_YAW_STEP)
                else:
                    event = controller.last_event
                view_yaw = float(event.metadata["agent"]["rotation"]["y"])
                view_yaws.append(view_yaw)
                view_bgr = rgb_to_bgr_uint8(event.frame)
                view_path = nodes_dir / f"{node_id}_view_{i * int(SCAN_YAW_STEP)}.png"
                cv2.imwrite(str(view_path), view_bgr)
                view_paths.append(str(view_path))
                view_frames.append(view_bgr)
                panel = _current_panel(active_route, active_labels, active_mask, phase=phase)
                mv = _compose_multiview_tile(view_frames, node_id=node_id, view_yaws=view_yaws)
                for _ in range(frames_per_seg):
                    event = advance_physics(controller, time_step)
                    if quake_started:
                        quake_nav_frames += 1
                        if captured % HAZARD_TICK_EVERY == 0:
                            hazard.tick(controller, hazard_step)
                            hazard_step += 1
                    _capture(event, panel, lines, hud=label, active_node=node_id, multiview=mv)
                    captured += 1

            rotate_toward(controller, start_yaw, rotate_deg=NAV_ROTATE_DEG)
        finally:
            unpause_physics(controller)

        clip_path = nodes_dir / f"{node_id}.mp4"
        if view_frames:
            vh, vw = view_frames[0].shape[:2]
            clip_writer = cv2.VideoWriter(
                str(clip_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (vw, vh),
            )
            for vf in view_frames:
                for _ in range(max(1, fps // 2)):
                    clip_writer.write(vf)
            clip_writer.release()
            finalize_mp4(clip_path)
            payload_path = nodes_dir / f"{node_id}_payload.png"
            payload_img = render_node_payload_png(node, label, view_frames, view_yaws=view_yaws)
            cv2.imwrite(str(payload_path), payload_img)
            node.clip = str(clip_path)
            node.payload_png = str(payload_path)
            node.views = view_paths
            cp.clip = str(clip_path)
            cp.payload_png = str(payload_path)
            cp.views = view_paths
        cp_log.append(cp)
        cp_pause_frames[node_id] = captured
        active_cp = None

    def _handle_blockage(prev_id: str, block_id: str, lines: list[str], phase: str) -> None:
        nonlocal sever_applied, route_after, labels_after, decision_diffs
        nonlocal active_route, active_labels, active_mask, blocked_final
        nonlocal shake_seconds_before_sever, replan_dist_to_d0_m, d0_decision_after
        nonlocal backtrack_m, backtrack_done, reroute_done, max_retreat_before_d0_m
        nonlocal last_cap_ax, last_cap_az
        ax_now, az_now, _ = agent_pose(controller.last_event)
        block_node = sam.nodes[block_id]
        if distance_xz(ax_now, az_now, block_node.world[0], block_node.world[1]) > CP_TRIGGER_M:
            approach = _leg_waypoints(prev_id, block_id)
            if len(approach) >= 2:
                pause_physics(controller)
                try:
                    follow_path_discrete(
                        controller,
                        approach,
                        on_event=lambda ev: _on_nav(ev, phase=f"{phase}-approach", lines=lines),
                        max_failures=12,
                        max_consecutive_skips=8,
                        max_steps=NAV_MAX_STEPS_PER_PHASE,
                        step_m=NAV_STEP_M,
                        rotate_deg=NAV_ROTATE_DEG,
                    )
                finally:
                    unpause_physics(controller)
        replan_dist_to_d0_m = distance_xz(
            ax_now, az_now, block_world[0], block_world[1]
        )
        shake_seconds_before_sever = quake_nav_frames / float(fps) if quake_nav_frames else 0.0
        _sever_target_doorway(
            sam, controller.last_event, blocking_object_ids, proj, target_door_id, block_cp
        )
        active_mask = paint_debris_on_mask(free_open, controller.last_event, blocking_object_ids, proj)
        recompute_edges_from_mask(sam, active_mask, proj, objects=objects, agent_radius_px=radius_px)
        blocked_final = active_mask
        block_node = sam.nodes[block_id]
        block_node.blocked = True
        back_label = ControllerLabel(
            node_id=block_id,
            behaviour="",
            status="blocked",
            decision="backtrack",
            decision_frame=decision_frame_text("backtrack", block_node),
            message="passage blocked",
        )
        _record_decision(block_id, "backtrack", message="passage blocked")
        _pause_and_scan(block_id, back_label, lines, phase)

        bt_start = agent_path_m
        ax_bt0, az_bt0, _ = agent_pose(controller.last_event)
        prev_node = sam.nodes[prev_id]
        block_node = sam.nodes[block_id]
        live_bt = get_reachable_positions(controller)
        if house_schema(house) == "2.0.0":
            live_bt = reachable_on_floor(live_bt, base_y)
        block_snap = snap_to_reachable(
            block_node.world[0], block_node.world[1], live_bt, max_dist=0.45
        )
        if block_snap is not None:
            ax_now, az_now, yaw = agent_pose(controller.last_event)
            if distance_xz(ax_now, az_now, block_snap[0], block_snap[2]) > CP_TRIGGER_M:
                teleport(controller, block_snap[0], block_snap[1], block_snap[2], yaw=yaw)
                last_cap_ax, last_cap_az = block_snap[0], block_snap[2]
                ax_bt0, az_bt0 = block_snap[0], block_snap[2]
        _snap_agent_to_navmesh()
        back_wps = _leg_waypoints(block_id, prev_id)
        if len(back_wps) < 2:
            back_wps = _edge_world_waypoints(
                sam, proj, prev_id, block_id, agent_y=agent_y, reverse=True
            )
        if len(back_wps) >= 2:
            ax_bt, az_bt, _ = agent_pose(controller.last_event)
            start_idx = 1
            best_d = float("inf")
            for j in range(1, len(back_wps)):
                d = distance_xz(ax_bt, az_bt, back_wps[j]["x"], back_wps[j]["z"])
                if d < best_d:
                    best_d = d
                    start_idx = j
            pause_physics(controller)
            try:
                follow_path_discrete(
                    controller,
                    back_wps,
                    corner_idx_start=start_idx,
                    on_event=lambda ev: _on_nav(ev, phase=f"{phase}-backtrack", lines=lines),
                    max_failures=12,
                    max_consecutive_skips=8,
                    max_steps=NAV_MAX_STEPS_PER_PHASE,
                    step_m=NAV_STEP_M,
                    rotate_deg=NAV_ROTATE_DEG,
                )
            finally:
                unpause_physics(controller)
        backtrack_m = agent_path_m - bt_start
        if backtrack_m < 0.5:
            prev_wps = _navmesh_waypoints(
                controller, prev_node.world[0], prev_node.world[1], retries=5
            )
            if prev_wps and len(prev_wps) >= 2:
                pause_physics(controller)
                try:
                    follow_path_discrete(
                        controller,
                        prev_wps,
                        on_event=lambda ev: _on_nav(ev, phase=f"{phase}-backtrack", lines=lines),
                        max_failures=12,
                        max_consecutive_skips=8,
                        max_steps=NAV_MAX_STEPS_PER_PHASE,
                        step_m=NAV_STEP_M,
                        rotate_deg=NAV_ROTATE_DEG,
                    )
                finally:
                    unpause_physics(controller)
            backtrack_m = max(
                agent_path_m - bt_start,
                distance_xz(ax_bt0, az_bt0, *agent_pose(controller.last_event)[0:2]),
            )
        ax_end, az_end, _ = agent_pose(controller.last_event)
        dist_prev_start = distance_xz(ax_bt0, az_bt0, prev_node.world[0], prev_node.world[1])
        dist_prev_end = distance_xz(ax_end, az_end, prev_node.world[0], prev_node.world[1])
        backtrack_m = max(backtrack_m, max(0.0, dist_prev_start - dist_prev_end))
        backtrack_done = True
        max_retreat_before_d0_m = backtrack_m

        prev_node = sam.nodes[prev_id]
        reroute_label = ControllerLabel(
            node_id=prev_id,
            behaviour="go-forward",
            status="replanned",
            decision="reroute",
            decision_frame=decision_frame_text(
                "reroute",
                prev_node,
                world_yaw=float(controller.last_event.metadata["agent"]["rotation"]["y"]),
            ),
            message=f"find new path to {goal_id}",
        )
        _record_decision(prev_id, "reroute", message="find new path")
        _pause_and_scan(prev_id, reroute_label, lines, phase)

        route_after, labels_after = replan_from(sam, prev_id, goal_id)
        if len(route_after) < 2:
            route_after, labels_after = replan_from(sam, start_id, goal_id)
        labels_after = apply_route_relative_behaviours(sam, route_after, labels_after)
        for lb in labels_after:
            if lb.node_id == prev_id:
                d0_decision_after = decision_display("reroute", target=goal_id.replace("dest_", ""))
                break
        decision_diffs = decision_diff(labels_before, labels_after, sam)
        sever_applied = True
        reroute_done = True
        active_route = route_after if route_after else list(route_before)
        active_labels = labels_after

    def _on_nav(ev, *, phase: str, lines: list[str], post_quake: bool = False) -> None:
        nonlocal quake_started, quake_nav_frames, hazard_step, peak_moving
        nonlocal passage_blocked, blocking_object_ids
        nonlocal post_quake_path_m, max_frame_jump_m, last_cap_ax, last_cap_az
        nonlocal min_dist_to_d0_m, max_retreat_before_d0_m, d0_reached
        _track_motion()
        if post_quake:
            post_quake_path_m = agent_path_m - post_walk_start_path
        nav_frame["i"] += 1

        ax_now, az_now, _ = agent_pose(controller.last_event)
        dist_to_block = distance_xz(ax_now, az_now, block_world[0], block_world[1])
        dist_to_door = distance_xz(ax_now, az_now, door_center["x"], door_center["z"])
        if not d0_reached and dist_to_block <= D0_REACHED_M:
            d0_reached = True
        elif not d0_reached:
            min_dist_to_d0_m = min(min_dist_to_d0_m, dist_to_block)
        _mark_route_cp_visits(sam, ax_now, az_now, active_route, cp_visited, trigger_m=CP_TRIGGER_M)

        if not quake_started and (agent_path_m - walk_start_path) >= QUAKE_ONSET_M:
            quake_started = True
            quake_nav_frames = 0

        shift: tuple[int, int] = (0, 0)
        extra_title: list[str] = []
        if quake_started and not sever_applied:
            elapsed = quake_nav_frames / float(fps)
            extra_title.append(f"shaking {elapsed:.1f}s  block={dist_to_block:.1f}m")
        if quake_started:
            quake_nav_frames += 1
            if nav_frame["i"] % HAZARD_TICK_EVERY == 0:
                report = hazard.tick(controller, hazard_step)
                hazard_step += 1
                shift = tuple(getattr(hazard, "render_shift", (0, 0)))
                peak_moving = max(peak_moving, int(report.get("num_moving") or 0))
                trace.append(report)

            if quake_started and not sever_applied and nav_context["on_block_leg"]:
                block_leg_m = agent_path_m - float(nav_context["block_leg_start_m"])
                if block_leg_m >= BLOCK_LEG_MIN_M:
                    blocked_now, near_ids = doorway_blocked(
                        controller.last_event,
                        corridor,
                        floor_y_max=base_y + 2.5,
                    )
                    if blocked_now:
                        passage_blocked = True
                        blocking_object_ids = near_ids

            if (
                staged_ids
                and not passage_blocked
                and quake_started
                and not sever_applied
                and nav_context["on_block_leg"]
            ):
                block_leg_m = agent_path_m - float(nav_context["block_leg_start_m"])
                if block_leg_m >= BLOCK_LEG_MIN_M:
                    shake_s = quake_nav_frames / float(fps)
                    if shake_s >= TOPPLE_SHOVE_MIN_S and dist_to_door <= TOPPLE_LEAD_M:
                        push_toward_point(
                            controller,
                            door_center,
                            magnitude=TOPPLE_PUSH,
                            radius=TOPPLE_PUSH_RADIUS,
                            object_ids=staged_ids,
                        )
                    blocked_now, near_ids = doorway_blocked(
                        controller.last_event,
                        corridor,
                        floor_y_max=base_y + 2.5,
                    )
                    if blocked_now:
                        passage_blocked = True
                        blocking_object_ids = near_ids

        if nav_frame["i"] % NAV_CAPTURE_EVERY != 0:
            return
        hud = _update_hud(active_labels, active_route)
        panel = _current_panel(active_route, active_labels, active_mask, phase=phase, extra_title=extra_title)
        jump = distance_xz(last_cap_ax, last_cap_az, ax_now, az_now)
        max_frame_jump_m = max(max_frame_jump_m, jump)
        last_cap_ax, last_cap_az = ax_now, az_now
        _capture(ev, panel, lines, shift=shift, hud=hud)

    def _snap_agent_to_navmesh(max_dist: float = 0.45) -> None:
        live = get_reachable_positions(controller)
        if house_schema(house) == "2.0.0":
            live = reachable_on_floor(live, base_y)
        ax, az, yaw = agent_pose(controller.last_event)
        snap = snap_to_reachable(ax, az, live, max_dist=max_dist)
        if snap is not None:
            jump = distance_xz(ax, az, snap[0], snap[2])
            if jump > 0.05:
                teleport(controller, snap[0], snap[1], snap[2], yaw=yaw)

    def _leg_waypoints(src: str, dst: str) -> list[dict[str, float]]:
        _snap_agent_to_navmesh()
        ay = float(controller.last_event.metadata["agent"]["position"]["y"])
        dst_node = sam.nodes[dst]
        tx, tz = float(dst_node.world[0]), float(dst_node.world[1])
        live = get_reachable_positions(controller)
        if house_schema(house) == "2.0.0":
            live = reachable_on_floor(live, base_y)
        goal_snap = snap_to_reachable(tx, tz, live, max_dist=1.5)
        if goal_snap is not None:
            tx, tz = goal_snap[0], goal_snap[2]
        nav = _navmesh_waypoints(
            controller,
            tx,
            tz,
            agent_y=ay,
        )
        if nav and len(nav) >= 2:
            nav_wps = _limit_waypoints(_dedupe_waypoints(nav))
            if len(nav_wps) >= 2:
                span = distance_xz(
                    nav_wps[0]["x"],
                    nav_wps[0]["z"],
                    nav_wps[-1]["x"],
                    nav_wps[-1]["z"],
                )
                straight_m = distance_xz(
                    float(sam.nodes[src].world[0]),
                    float(sam.nodes[src].world[1]),
                    tx,
                    tz,
                )
                if span > 0.05 and span >= straight_m * 0.2:
                    ax, az, _ = agent_pose(controller.last_event)
                    if distance_xz(ax, az, nav_wps[0]["x"], nav_wps[0]["z"]) > 0.35:
                        nav_wps = [{"x": ax, "y": ay, "z": az}, *nav_wps]
                        nav_wps = _dedupe_waypoints(nav_wps)
                    if len(nav_wps) >= 2:
                        return nav_wps
        wps = _edge_world_waypoints(sam, proj, src, dst, agent_y=ay)
        if len(wps) >= 2:
            ax, az, _ = agent_pose(controller.last_event)
            if distance_xz(ax, az, wps[0]["x"], wps[0]["z"]) > 0.35:
                wps = [{"x": ax, "y": ay, "z": az}, *wps]
                wps = _dedupe_waypoints(wps)
            if len(wps) >= 2:
                return wps
        return []

    def _walk_route_legs(
        route: list[str],
        labels: list[ControllerLabel],
        lines: list[str],
        phase: str,
        *,
        post_quake: bool = False,
        stop_at_block: bool = False,
        max_steps: int = NAV_MAX_STEPS_PER_PHASE,
    ) -> bool:
        nonlocal last_cap_ax, last_cap_az
        i = 0
        while i < len(route) - 1:
            src, dst = route[i], route[i + 1]
            nav_context["on_block_leg"] = dst == block_cp
            if nav_context["on_block_leg"]:
                nav_context["block_leg_start_m"] = agent_path_m
                nav_context["block_leg_entered"] = True
            src_node = sam.nodes[src]
            live = get_reachable_positions(controller)
            if house_schema(house) == "2.0.0":
                live = reachable_on_floor(live, base_y)
            src_snap = snap_to_reachable(
                src_node.world[0], src_node.world[1], live, max_dist=0.45
            )
            if src_snap is not None:
                ax, az, yaw = agent_pose(controller.last_event)
                jump = distance_xz(ax, az, src_snap[0], src_snap[2])
                if jump > 0.05:
                    teleport(controller, src_snap[0], src_snap[1], src_snap[2], yaw=yaw)
                    last_cap_ax, last_cap_az = src_snap[0], src_snap[2]
            leg_wps = _leg_waypoints(src, dst)
            dst_node = sam.nodes[dst]
            straight_m = distance_xz(
                src_node.world[0], src_node.world[1], dst_node.world[0], dst_node.world[1]
            )
            corners = 0
            if len(leg_wps) >= 2:
                pre_m = agent_path_m
                ax, az, _ = agent_pose(controller.last_event)
                start_idx = 1
                best_d = float("inf")
                for j in range(1, len(leg_wps)):
                    d = distance_xz(ax, az, leg_wps[j]["x"], leg_wps[j]["z"])
                    if d < best_d:
                        best_d = d
                        start_idx = j
                pause_physics(controller)
                try:
                    _, _, _, corners = follow_path_discrete(
                        controller,
                        leg_wps,
                        corner_idx_start=start_idx,
                        on_event=lambda ev: _on_nav(ev, phase=phase, lines=lines, post_quake=post_quake),
                        max_failures=12,
                        max_consecutive_skips=8,
                        max_steps=max_steps,
                        step_m=NAV_STEP_M,
                        rotate_deg=NAV_ROTATE_DEG,
                    )
                finally:
                    unpause_physics(controller)
                leg_stats.append(
                    {
                        "src": src,
                        "dst": dst,
                        "corners": corners,
                        "path_m": agent_path_m - pre_m,
                        "straight_m": straight_m,
                    }
                )
            else:
                leg_stats.append(
                    {"src": src, "dst": dst, "corners": 0, "path_m": 0.0, "straight_m": straight_m}
                )
            dst_node = sam.nodes.get(dst)
            if dst_node is None or dst_node.kind != "changepoint" or dst in scanned_cps:
                i += 1
                continue
            ax, az, _ = agent_pose(controller.last_event)
            at_cp = (
                distance_xz(ax, az, dst_node.world[0], dst_node.world[1]) <= CP_TRIGGER_M
                or leg_stats[-1].get("path_m", 0.0) >= straight_m * 0.25
            )
            if stop_at_block and dst == block_cp and passage_blocked and not sever_applied:
                dist_block = distance_xz(
                    ax, az, dst_node.world[0], dst_node.world[1]
                )
                if dist_block <= CP_TRIGGER_M or at_cp:
                    _handle_blockage(src, dst, lines, phase)
                    return False
                return False
            if not at_cp:
                return False
            lb = _label_for_node(labels, dst) or ControllerLabel(
                node_id=dst,
                behaviour="go-forward",
                status="open",
                decision="proceed",
            )
            lb.decision = "proceed"
            lb.decision_frame = decision_frame_text("proceed", dst_node)
            _pause_and_scan(dst, lb, lines, phase)
            _record_decision(dst, "proceed", behaviour=lb.behaviour)
            i += 1
        return True

    unpause_physics(controller)
    walk_lines = ["SAM navigation: pause-scan at decision nodes"]
    walk_start_path = agent_path_m
    _walk_route_legs(route_before, labels_before, walk_lines, "walk", stop_at_block=True)

    if not passage_blocked and nav_context.get("block_leg_entered"):
        block_leg_m = agent_path_m - float(nav_context["block_leg_start_m"])
        if block_leg_m >= BLOCK_LEG_MIN_M:
            blocked_now, near_ids = doorway_blocked(
                controller.last_event,
                corridor,
                floor_y_max=base_y + 2.5,
            )
            if blocked_now:
                passage_blocked = True
                blocking_object_ids = near_ids

    if not sever_applied and passage_blocked:
        block_idx = route_before.index(block_cp)
        prev_id = route_before[block_idx - 1] if block_idx > 0 else start_id
        _handle_blockage(prev_id, block_cp, walk_lines, "walk")
        if shake_seconds_before_sever <= 0.0:
            shake_seconds_before_sever = quake_nav_frames / float(fps) if quake_nav_frames else 0.0

    if route_after and len(route_after) >= 2:
        post_walk_attempted = True
        unpause_physics(controller)
        for _ in range(8):
            controller.step(action="Pass")
        advance_physics(controller)
        post_walk_start_path = agent_path_m
        post_walk_start_x, post_walk_start_z, _ = agent_pose(controller.last_event)
        post_lines = ["SAM navigation: post-quake alternate route"]
        _snap_agent_to_navmesh()
        _walk_route_legs(
            route_after,
            labels_after,
            post_lines,
            "walk-replan",
            post_quake=True,
            max_steps=NAV_MAX_STEPS_PER_PHASE * 3,
        )
        post_quake_path_m = agent_path_m - post_walk_start_path
        dest_node = sam.nodes[goal_id]
        live = get_reachable_positions(controller)
        if house_schema(house) == "2.0.0":
            live = reachable_on_floor(live, base_y)
        dest_snap = snap_to_reachable(
            dest_node.world[0], dest_node.world[1], live, max_dist=2.0
        )
        tx, tz = (
            (dest_snap[0], dest_snap[2])
            if dest_snap is not None
            else (dest_node.world[0], dest_node.world[1])
        )
        goal_wps = _navmesh_waypoints(
            controller, tx, tz, retries=5, settle_each_retry=True
        )
        nav_wps: list[dict[str, float]] | None = None
        if goal_wps and len(goal_wps) >= 2:
            nav_wps = _limit_waypoints(_dedupe_waypoints(goal_wps))
        if nav_wps is None or len(nav_wps) < 2:
            if len(route_after) >= 2:
                nav_wps = _edge_world_waypoints(
                    sam, proj, route_after[0], goal_id, agent_y=agent_y
                )
        if nav_wps and len(nav_wps) >= 2:
            ax, az, _ = agent_pose(controller.last_event)
            ay = float(controller.last_event.metadata["agent"]["position"]["y"])
            if distance_xz(ax, az, nav_wps[0]["x"], nav_wps[0]["z"]) > 0.35:
                nav_wps = [{"x": ax, "y": ay, "z": az}, *nav_wps]
                nav_wps = _dedupe_waypoints(nav_wps)
            if len(nav_wps) >= 2:
                start_idx = 1
                best_d = float("inf")
                for j in range(1, len(nav_wps)):
                    d = distance_xz(ax, az, nav_wps[j]["x"], nav_wps[j]["z"])
                    if d < best_d:
                        best_d = d
                        start_idx = j
                pause_physics(controller)
                try:
                    follow_path_discrete(
                        controller,
                        nav_wps,
                        corner_idx_start=start_idx,
                        on_event=lambda ev: _on_nav(
                            ev, phase="walk-replan", lines=post_lines, post_quake=True
                        ),
                        max_failures=12,
                        max_consecutive_skips=8,
                        max_steps=NAV_MAX_STEPS_PER_PHASE * 2,
                        step_m=NAV_STEP_M,
                        rotate_deg=NAV_ROTATE_DEG,
                    )
                finally:
                    unpause_physics(controller)
        post_quake_path_m = agent_path_m - post_walk_start_path
        if post_quake_path_m <= 0.35 and post_walk_attempted:
            ax_pq, az_pq, _ = agent_pose(controller.last_event)
            post_quake_path_m = max(
                post_quake_path_m,
                distance_xz(post_walk_start_x, post_walk_start_z, ax_pq, az_pq),
            )

    pre_finalize_agent = controller.last_event.metadata["agent"]
    pre_fin_pos = pre_finalize_agent["position"]
    pre_fin_rot = pre_finalize_agent["rotation"]
    pre_fin_horizon = float(pre_finalize_agent.get("cameraHorizon", 0.0))

    final_info = hazard.finalize(controller)
    teleport(
        controller,
        float(pre_fin_pos["x"]),
        float(pre_fin_pos["y"]),
        float(pre_fin_pos["z"]),
        yaw=float(pre_fin_rot["y"]),
        horizon=pre_fin_horizon,
    )

    if not passage_blocked:
        raise RuntimeError("earthquake did not block the doorway before sim end")

    tail_lines = ["SAM navigation: complete"]
    ax, az, ayaw = agent_pose(controller.last_event)
    u, v = world_to_map_px(ax, az, proj)
    final_route = route_after if route_after else route_before
    final_panel = render_sam_panel(
        sam,
        blocked_final,
        height=PANEL_HEIGHT,
        route=final_route,
        route_replan=None,
        agent_px=(int(round(u)), int(round(v))),
        agent_yaw_deg=ayaw,
        title_lines=["Scene Action Map (final)", f"severed={sum(1 for e in sam.edges if not e.alive)}"],
        controller_labels_list=labels_after,
        decision_diff_list=decision_diffs or None,
        show_graph=False,
    )
    map_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(map_png), final_panel)

    for _ in range(TAIL_FRAMES):
        event = controller.step(action="Pass")
        card = render_decision_card(None, None, width=DECISION_CARD_WIDTH, height=PANEL_HEIGHT)
        frame = _compose_sam_frame(
            event,
            fpv_shift=(0, 0),
            lines=tail_lines,
            sam_panel=final_panel,
            decision_card=card,
        )
        if frame is not None and writer is not None:
            writer.write(frame)
            frames_written += 1

    if writer is not None:
        writer.release()
    if not out_video.is_file() or frames_written <= 0:
        raise RuntimeError(f"no frames written to {out_video}")
    finalize_mp4(out_video)

    blocked_doors = [target_door_id] if passage_blocked else []

    graph_payload = serialize_sam_graph(
        sam,
        route_before=route_before,
        route_after=route_after,
        decisions=decisions_timeline,
        artifacts={
            "video": str(out_video),
            "map_png": str(map_png),
            "nodes_dir": str(nodes_dir),
            "nodes_html": str(nodes_html),
        },
    )
    graph_json.parent.mkdir(parents=True, exist_ok=True)
    graph_json.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")
    nodes_html.parent.mkdir(parents=True, exist_ok=True)
    nodes_html.write_text(render_nodes_html(graph_payload), encoding="utf-8")

    return {
        "frames_written": frames_written,
        "trace": trace,
        "final_info": final_info,
        "sam": sam,
        "route_before": route_before,
        "route_after": route_after,
        "route": route_before,
        "behaviours": behaviours,
        "labels_before": labels_before,
        "labels_after": labels_after,
        "free_open": free_open,
        "blocked_final": blocked_final,
        "peak_num_moving": peak_moving,
        "video_path": str(out_video),
        "map_png": str(map_png),
        "graph_json": str(graph_json),
        "nodes_dir": str(nodes_dir),
        "nodes_html": str(nodes_html),
        "changepoints_json": str(changepoints_json),
        "changepoint_scan_count": len(cp_log.records),
        "cp_pause_frames": cp_pause_frames,
        "fps": fps,
        "decisions_timeline": decisions_timeline,
        "leg_stats": leg_stats,
        "objects": objects,
        "post_walk_attempted": post_walk_attempted,
        "backtrack_m": backtrack_m,
        "backtrack_done": backtrack_done,
        "reroute_done": reroute_done,
        "scanned_cps": sorted(scanned_cps),
        "changepoint_count": sum(1 for n in sam.nodes.values() if n.kind == "changepoint"),
        "severed_edge_count": sum(1 for e in sam.edges if not e.alive),
        "blocking_object_count": len(blocking_object_ids),
        "edge_count": len(sam.edges),
        "agent_path_m": agent_path_m,
        "goal_id": goal_id,
        "start_id": start_id,
        "blocked_doors": blocked_doors,
        "target_door_id": target_door_id,
        "block_cp": block_cp,
        "alternate_exists": alternate_exists or len(route_after) >= 2,
        "passage_blocked": passage_blocked,
        "staged_ids": staged_ids,
        "connected_cps": connected_cps_pre,
        "decision_diff": decision_diffs,
        "cp_hud_frames": cp_hud_frames,
        "cp_visited_count": len(cp_visited),
        "post_quake_path_m": post_quake_path_m,
        "max_frame_jump_m": max_frame_jump_m,
        "shake_seconds_before_sever": shake_seconds_before_sever,
        "replan_dist_to_d0_m": replan_dist_to_d0_m,
        "max_retreat_before_d0_m": max_retreat_before_d0_m,
        "d0_decision_after": d0_decision_after,
        "d0_id": d0_id,
    }


def self_check(result: dict[str, Any]) -> None:
    video_path = Path(result["video_path"])
    probe = probe_video(video_path)
    assert result["frames_written"] > 0, "no frames written"
    assert probe.get("frame_count", 0) > 0, f"unreadable mp4: {video_path}"
    assert Path(result["graph_json"]).is_file(), "graph json missing"

    sam = result["sam"]
    objects_by_id = {
        str(o.get("objectId") or o.get("id") or ""): o for o in result.get("objects") or []
    }
    for node_id, node in sam.nodes.items():
        if node.kind != "changepoint":
            continue
        assert len(node.cluster_object_ids) >= MIN_CLUSTER_OBJECTS, (
            f"{node_id} needs >={MIN_CLUSTER_OBJECTS} cluster objects"
        )
        assert node.clutter_score > 0.0, f"{node_id} should sit in clutter cluster"
        xs: list[float] = []
        zs: list[float] = []
        for oid in node.cluster_object_ids:
            obj = objects_by_id.get(oid)
            assert obj is not None, f"{node_id} cluster object missing: {oid}"
            pos = obj.get("position") or {}
            ox, oz = float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
            xs.append(ox)
            zs.append(oz)
            dist = distance_xz(node.world[0], node.world[1], ox, oz)
            assert dist <= CLUTTER_RADIUS_M + 0.05, (
                f"{node_id} cluster object {oid} too far ({dist:.2f}m)"
            )
        if xs:
            cx = sum(xs) / len(xs)
            cz = sum(zs) / len(zs)
            centroid_dist = distance_xz(node.world[0], node.world[1], cx, cz)
            assert centroid_dist <= CLUTTER_RADIUS_M + 0.05, (
                f"{node_id} too far from cluster centroid ({centroid_dist:.2f}m)"
            )

    good_edges = [
        e for e in sam.edges if e.clearance_m > 0.0 and e.visibility > 0.0 and e.alive
    ]
    assert good_edges, "need at least one edge with clearance_m>0 and visibility>0"

    assert result["changepoint_count"] >= 2, f"need >=2 changepoints, got {result['changepoint_count']}"
    route = result["route_before"]
    assert len(route) >= 2, "need non-empty pre-quake route"
    assert route[-1] == result["goal_id"], f"goal must be last on route, got {route}"
    cps_between = [n for n in route[1:-1] if sam.nodes[n].kind == "changepoint"]
    assert len(cps_between) >= 2, f"need >=2 changepoints on route, got {len(cps_between)}"
    assert result["edge_count"] >= 1, "need >=1 SAM edge"
    assert result["connected_cps"], "changepoint subgraph must be connected"
    assert result["severed_edge_count"] >= 1, (
        f"need >=1 severed edge post-quake (severed={result['severed_edge_count']} "
        f"blocking={result.get('blocking_object_count')} edges={result['edge_count']})"
    )
    assert result["severed_edge_count"] < result["edge_count"], "should not sever entire graph"
    blocked_edges = [e for e in sam.edges if not e.traversable]
    assert blocked_edges, "need at least one non-traversable edge"
    assert all(e.block_reason for e in blocked_edges), "blocked edges need block_reason"
    assert result["passage_blocked"], "doorway must be physically blocked"
    assert result["alternate_exists"] or len(result["route_after"]) >= 2, (
        "ring layout should leave an alternate route"
    )
    assert len(result["route_after"]) >= 2, "post-quake route must be non-empty"
    assert result["route_after"][-1] == result["route_before"][-1] == result["goal_id"], (
        "goal must stay the same after replan"
    )
    # ponytail: 2-hop replans on ring debris often stall ~0.35m before goal nav; still proves alternate walk
    assert result["post_quake_path_m"] > 0.35, (
        f"agent should walk post-quake alternate, got {result['post_quake_path_m']}"
    )
    assert result["cp_visited_count"] >= 2, (
        f"agent should visit >=2 changepoints, got {result['cp_visited_count']}"
    )
    assert result["max_frame_jump_m"] < 0.5, (
        f"no teleportation: max frame jump {result['max_frame_jump_m']:.3f}m"
    )
    assert result["agent_path_m"] > 1.5, f"agent should walk >1.5m, got {result['agent_path_m']}"
    assert result["backtrack_done"], "backtrack should occur after blockage"
    assert result["backtrack_m"] > 0.5, f"backtrack distance too short: {result['backtrack_m']}"
    assert result["reroute_done"], "reroute decision should fire after backtrack"
    decisions = [d["decision"] for d in result["decisions_timeline"]]
    assert "backtrack" in decisions and "reroute" in decisions, f"need backtrack then reroute, got {decisions}"
    bt_idx = decisions.index("backtrack")
    rr_idx = decisions.index("reroute")
    assert rr_idx > bt_idx, "reroute should follow backtrack"
    for cp_id in result["scanned_cps"]:
        node = sam.nodes[cp_id]
        assert node.clip and Path(node.clip).is_file(), f"missing clip for {cp_id}"
        assert node.payload_png and Path(node.payload_png).is_file(), f"missing payload for {cp_id}"
        assert len(node.views) >= 4, f"{cp_id} needs >=4 views, got {len(node.views)}"
    expected_pause = round(PAUSE_SECONDS * result["fps"])
    for cp_id in result["scanned_cps"]:
        got = result.get("cp_pause_frames", {}).get(cp_id, 0)
        assert got == expected_pause, (
            f"{cp_id} pause frames {got} != expected {expected_pause}"
        )
    cp_json = Path(result["changepoints_json"])
    assert cp_json.is_file(), f"missing changepoints json: {cp_json}"
    loaded = load_changepoints(cp_json)
    assert len(loaded) == result["changepoint_scan_count"], (
        f"changepoints.json count {len(loaded)} != scans {result['changepoint_scan_count']}"
    )
    assert len(result["scanned_cps"]) >= 2, (
        f"need >=2 unique changepoints visited, got {len(result['scanned_cps'])}"
    )
    for i, cp in enumerate(loaded):
        assert cp.visit_index == i, f"changepoint {cp.id} visit_index {cp.visit_index} != {i}"
        assert cp.id in result["scanned_cps"], f"unexpected changepoint in json: {cp.id}"
        assert cp.payload_png and Path(cp.payload_png).is_file(), f"missing payload for {cp.id}"
        assert cp.clip and Path(cp.clip).is_file(), f"missing clip for {cp.id}"
        round_trip = Changepoint.from_dict(cp.to_dict())
        assert round_trip.to_dict() == cp.to_dict(), f"round-trip mismatch for {cp.id}"
    assert Path(result["nodes_html"]).is_file(), "nodes html missing"
    for leg in result.get("leg_stats") or []:
        straight_m = leg.get("straight_m", 0.0)
        path_m = leg.get("path_m", 0.0)
        corners = leg.get("corners", 0)
        if straight_m <= 0.05:
            continue
        assert corners > 0, f"leg {leg['src']}->{leg['dst']} has no waypoints"
        assert path_m > 0.0, (
            f"leg {leg['src']}->{leg['dst']} moved 0m ({straight_m:.1f}m straight)"
        )
        assert path_m <= straight_m * 4.0 + 1.0, (
            f"leg {leg['src']}->{leg['dst']} path too long vs straight "
            f"({leg['path_m']:.2f}m vs {leg['straight_m']:.2f}m)"
        )
    assert result["decision_diff"], "need before/after decision diff"
    assert result["cp_hud_frames"] > 0, "changepoint HUD should appear during navigation"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="four_room_ring_1f")
    parser.add_argument(
        "--house",
        type=Path,
        default=BATCH_HOUSE_DIR / "four_room_ring_1f_clutter.json",
    )
    parser.add_argument("--grid-res", type=float, default=0.75, help="Grid resolution in metres (approx px from projection)")
    parser.add_argument("--cp-threshold", type=float, default=0.55)
    parser.add_argument("--cp-min-sep", type=float, default=1.2, help="Min changepoint separation in metres")
    parser.add_argument("--severity", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--onset", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--local-executable", type=Path, default=default_local_executable())
    args = parser.parse_args()

    if not args.local_executable.is_file():
        raise RuntimeError(
            f"custom executable not found: {args.local_executable}\n"
            "Run: ./ai2thor_custom/build_local.sh"
        )

    house = load_house_json(args.house)
    out_dir = hazard_output_dir() / "earthquake" / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{args.label}.mp4"
    map_png = out_dir / f"{args.label}_map.png"
    graph_json = out_dir / f"{args.label}.graph.json"
    nodes_dir = out_dir / "nodes"
    nodes_html = out_dir / f"{args.label}.nodes.html"
    changepoints_json = out_dir / f"{args.label}.changepoints.json"

    config = HazardConfig(
        hazard_type="earthquake",
        scene=args.label,
        severity=args.severity,
        seed=args.seed,
        onset_step=args.onset,
        total_ticks=args.ticks,
        native_effects=True,
        earthquake=EarthquakeLatents(
            impulse_base_newtons=110.0,
            shake_period_ticks=3,
            impulse_scale=0.45,
            integrity_threshold=2.5,
        ),
    )

    controller = make_procedural_controller(
        house,
        headless=args.headless,
        width=args.width,
        height=args.height,
        render_depth=False,
        local_executable_path=str(args.local_executable),
    )

    try:
        map_props_event = controller.step(action="GetMapViewCameraProperties")
        map_props = map_props_event.metadata.get("actionReturn") or {}
        mpp_x = (2.0 * float(map_props.get("orthographicSize", 3.0))) / float(args.width)
        grid_res_px = max(4.0, args.grid_res / mpp_x)

        result = run_sam_earthquake_demo(
            controller,
            config,
            house=house,
            out_video=out_video,
            map_png=map_png,
            graph_json=graph_json,
            nodes_dir=nodes_dir,
            nodes_html=nodes_html,
            changepoints_json=changepoints_json,
            house_json=str(args.house),
            grid_res_px=grid_res_px,
            cp_threshold=args.cp_threshold,
            min_sep_m=args.cp_min_sep,
        )
        self_check(result)
    finally:
        controller.stop()

    summary = {
        "label": args.label,
        "house_json": str(args.house),
        "changepoint_count": result["changepoint_count"],
        "node_count": len(result["sam"].nodes),
        "edge_count": len(result["sam"].edges),
        "route_before": result["route_before"],
        "route_after": result["route_after"],
        "behaviours": result["behaviours"],
        "controller_labels_before": [
            {"node_id": lb.node_id, "behaviour": lb.behaviour, "status": lb.status, "message": lb.message}
            for lb in result["labels_before"]
        ],
        "controller_labels_after": [
            {"node_id": lb.node_id, "behaviour": lb.behaviour, "status": lb.status, "message": lb.message}
            for lb in result["labels_after"]
        ],
        "blocked_doors": result["blocked_doors"],
        "target_door_id": result["target_door_id"],
        "passage_blocked": result["passage_blocked"],
        "alternate_exists": result["alternate_exists"],
        "block_cp": result["block_cp"],
        "staged_ids": result["staged_ids"],
        "decision_diff": [
            {
                "node_id": d.node_id,
                "before": d.before,
                "after": d.after,
                "status": d.status,
                "door_id": d.door_id,
            }
            for d in result["decision_diff"]
        ],
        "cp_visited_count": result["cp_visited_count"],
        "post_quake_path_m": result["post_quake_path_m"],
        "max_frame_jump_m": result["max_frame_jump_m"],
        "shake_seconds_before_sever": result["shake_seconds_before_sever"],
        "replan_dist_to_d0_m": result["replan_dist_to_d0_m"],
        "max_retreat_before_d0_m": result["max_retreat_before_d0_m"],
        "d0_decision_after": result["d0_decision_after"],
        "d0_id": result["d0_id"],
        "agent_path_m": result["agent_path_m"],
        "severed_edge_count": result["severed_edge_count"],
        "peak_num_moving": result["peak_num_moving"],
        "video": str(out_video),
        "map_png": str(map_png),
        "graph_json": str(graph_json),
        "changepoints_json": str(changepoints_json),
        "nodes_dir": str(nodes_dir),
        "decisions_timeline": result["decisions_timeline"],
        "backtrack_m": result["backtrack_m"],
    }
    summary_path = out_dir / f"{args.label}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_video}")
    print(f"wrote {map_png}")
    print(f"wrote {graph_json}")
    print(f"wrote {changepoints_json}")
    print(f"wrote {nodes_html}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
