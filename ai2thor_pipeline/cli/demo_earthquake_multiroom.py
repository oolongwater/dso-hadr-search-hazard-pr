#!/usr/bin/env python3
"""Multi-room ProcTHOR earthquake with native C# shake + traversability overlay."""

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

from cli.demo_earthquake_traversability import (
    connected_pairs,
    grid_shortest_path,
    pixel_shortest_path,
    render_demo_video,
    snap_waypoint,
)
from cli.hazard_scenes import (
    SCENE_CAPTURE_PROFILES,
    _compose_frame,
    _label_lines,
)
from core.procthor_house import (
    _point_in_polygon,
    _polygon_from_room,
    default_house_path,
    default_local_executable,
    door_world_center,
    door_world_frame,
    doorway_corridor_rect,
    floor_base_y,
    house_floors,
    house_scene_id,
    house_schema,
    load_house_json,
    make_procedural_controller,
    map_projection_from_props,
    pick_primary_connector,
    push_toward_point,
    reachable_on_floor,
    room_centroid,
    rooms_on_floor,
    stage_doorway_blockers,
    vertical_connectors,
    world_to_map_px,
)
from core.thor import distance_xz, teleport
from core.video import finalize_mp4, probe_video, rgb_to_bgr_uint8
from hazard.functions import EarthquakeLatents, EarthquakeParams, HazardConfig, start_earthquake, stop_earthquake
from hazard.model import EarthquakeHazard
from hazard.utils import advance_physics, hazard_output_dir, pause_physics

DEMO_PREFIX = "demo_earthquake_multiroom_traversability"
SCENE_PREFIX = "scene_earthquake_multiroom"
CORRIDOR_DEPTH_M = 0.45
AGENT_RADIUS_M = 0.3
MIN_DOOR_DIST_M = CORRIDOR_DEPTH_M + AGENT_RADIUS_M + 0.6
TOPPLE_PUSH = 6.0
TOPPLE_PUSH_RADIUS = 1.5
TOPPLE_TICKS = 4
VIEW_DOOR_MAX_DIST_M = 3.5
VIEW_BACK_FROM_DOOR_M = 2.2
TAIL_FRAMES = 16
LOOP_PAIRS = [
    ("wp_a0", "wp_a1"),
    ("wp_a1", "wp_b0"),
    ("wp_b0", "wp_b1"),
    ("wp_b1", "wp_a0"),
]
CROSS_ROOM_PAIRS = {("wp_a1", "wp_b0"), ("wp_b0", "wp_a1"), ("wp_b1", "wp_a0"), ("wp_a0", "wp_b1")}


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def _verify_native_earthquake(controller) -> None:
    res = start_earthquake(controller, EarthquakeParams(magnitude=2.5, frequency_hz=2.0))
    if not res.metadata.get("lastActionSuccess", False):
        raise RuntimeError(
            "native StartEarthquake failed — rebuild custom player with Procedural scene: "
            "./ai2thor_custom/build_local.sh"
        )
    stop_earthquake(controller)


def _rect_xz_bounds(corners: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [c[0] for c in corners]
    zs = [c[1] for c in corners]
    return min(xs), max(xs), min(zs), max(zs)


def _obj_xz_bounds(obj: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = obj.get("axisAlignedBoundingBox") or {}
    center = bbox.get("center") or obj.get("position") or {}
    size = bbox.get("size") or {}
    cx = float(center.get("x", 0.0))
    cz = float(center.get("z", 0.0))
    hx = float(size.get("x", 0.3)) / 2.0
    hz = float(size.get("z", 0.3)) / 2.0
    return cx - hx, cx + hx, cz - hz, cz + hz


def _aabb_overlaps_rect(
    obj_bounds: tuple[float, float, float, float],
    rect_bounds: tuple[float, float, float, float],
) -> bool:
    ox0, ox1, oz0, oz1 = obj_bounds
    rx0, rx1, rz0, rz1 = rect_bounds
    return ox0 <= rx1 and ox1 >= rx0 and oz0 <= rz1 and oz1 >= rz0


def doorway_blocked(
    event,
    corridor_corners: list[tuple[float, float]],
    *,
    floor_y_max: float = 1.2,
) -> tuple[bool, list[str]]:
    blocking: list[str] = []
    rect_bounds = _rect_xz_bounds(corridor_corners)
    for obj in event.metadata.get("objects") or []:
        oid = obj.get("objectId")
        if not oid:
            continue
        if not (obj.get("pickupable") or obj.get("moveable")):
            continue
        pos = obj.get("position") or {}
        py = float(pos.get("y", 0.0))
        if py > floor_y_max:
            continue
        if _aabb_overlaps_rect(_obj_xz_bounds(obj), rect_bounds):
            blocking.append(str(oid))
    return len(blocking) >= 1, blocking


def _corridor_pixels(
    corridor_corners: list[tuple[float, float]],
    proj: dict[str, Any],
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    xs = [world_to_map_px(x, z, proj)[0] for x, z in corridor_corners]
    ys = [world_to_map_px(x, z, proj)[1] for x, z in corridor_corners]
    x0, x1 = int(min(xs)), int(max(xs))
    y0, y1 = int(min(ys)), int(max(ys))
    pixels: set[tuple[int, int]] = set()
    for y in range(max(0, y0), min(height, y1 + 1)):
        for x in range(max(0, x0), min(width, x1 + 1)):
            pixels.add((x, y))
    return pixels


def build_reachable_free_mask(
    reachable: list[dict[str, float]],
    proj: dict[str, Any],
    height: int,
    width: int,
) -> np.ndarray:
    free = np.zeros((height, width), dtype=bool)
    for pt in reachable:
        u, v = world_to_map_px(float(pt["x"]), float(pt["z"]), proj)
        ui, vi = int(round(u)), int(round(v))
        if 0 <= ui < width and 0 <= vi < height:
            free[vi, ui] = True
    if not free.any():
        return free
    dilated = cv2.dilate(free.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2)
    return dilated.astype(bool)


def add_overhead_camera_at_floor(controller, base_y: float) -> bool:
    props_event = controller.step(action="GetMapViewCameraProperties")
    props = props_event.metadata.get("actionReturn")
    if not props or not props_event.metadata.get("lastActionSuccess", False):
        return False
    position = dict(props["position"])
    position["y"] = float(base_y) + float(position.get("y", 8.0))
    event = controller.step(
        action="AddThirdPartyCamera",
        position=position,
        rotation=props["rotation"],
        orthographic=props.get("orthographic", True),
        orthographicSize=props.get("orthographicSize", 3.0),
    )
    return bool(event.metadata.get("lastActionSuccess", False))


def _room_floor_base_y(house: dict[str, Any], room: dict[str, Any]) -> float:
    floor_id = str(room.get("floorId") or "floor|0")
    return floor_base_y(house, floor_id)


def _reachable_in_room(
    reachable: list[dict[str, float]],
    room: dict[str, Any],
    *,
    house: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    poly = _polygon_from_room(room)
    pts = [
        p for p in reachable
        if _point_in_polygon(float(p["x"]), float(p["z"]), poly)
    ]
    if house is not None and house_schema(house) == "2.0.0":
        base_y = _room_floor_base_y(house, room)
        pts = reachable_on_floor(pts, base_y)
    return pts


def _reachable_to_px(
    pt: dict[str, float],
    free: np.ndarray,
    proj: dict[str, Any],
) -> tuple[int, int]:
    u, v = world_to_map_px(float(pt["x"]), float(pt["z"]), proj)
    x, y = int(round(u)), int(round(v))
    h, w = free.shape
    if 0 <= x < w and 0 <= y < h and free[y, x]:
        return x, y
    return snap_waypoint(free, x, y)


def _outside_corridor(
    pt: dict[str, float],
    corridor_corners: list[tuple[float, float]],
) -> bool:
    x = float(pt["x"])
    z = float(pt["z"])
    rx0, rx1, rz0, rz1 = _rect_xz_bounds(corridor_corners)
    return not (rx0 <= x <= rx1 and rz0 <= z <= rz1)


def _door_relative_waypoints(
    room_pts: list[dict[str, float]],
    door_center: dict[str, float],
    free: np.ndarray,
    proj: dict[str, Any],
    *,
    door_side_id: str,
    far_id: str,
    corridor_corners: list[tuple[float, float]] | None = None,
    must_be_free: np.ndarray | None = None,
) -> dict[str, tuple[int, int]]:
    if len(room_pts) < 2:
        raise RuntimeError(f"room needs >=2 reachable points for {door_side_id}/{far_id}")
    cx = float(door_center["x"])
    cz = float(door_center["z"])

    def _dist(pt: dict[str, float]) -> float:
        return distance_xz(float(pt["x"]), float(pt["z"]), cx, cz)

    def _px_free(px: tuple[int, int]) -> bool:
        if must_be_free is None:
            return True
        x, y = px
        return bool(must_be_free[y, x])

    candidates = list(room_pts)
    if corridor_corners:
        outside = [p for p in candidates if _outside_corridor(p, corridor_corners)]
        if outside:
            candidates = outside

    sorted_pts = sorted(candidates, key=_dist)
    near_pt = sorted_pts[0]
    for pt in sorted_pts:
        px = _reachable_to_px(pt, free, proj)
        if _dist(pt) >= MIN_DOOR_DIST_M and _px_free(px):
            near_pt = pt
            break
    else:
        for pt in sorted_pts:
            px = _reachable_to_px(pt, free, proj)
            if _px_free(px):
                near_pt = pt
                break

    far_pt = max(room_pts, key=_dist)
    if far_pt is near_pt or _dist(far_pt) <= _dist(near_pt) + 0.5:
        for pt in sorted(room_pts, key=_dist, reverse=True):
            if pt is not near_pt and _dist(pt) > _dist(near_pt) + 0.5:
                far_pt = pt
                break

    door_px = _reachable_to_px(near_pt, free, proj)
    far_px = _reachable_to_px(far_pt, free, proj)
    if door_px == far_px:
        for pt in sorted(room_pts, key=_dist, reverse=True):
            candidate = _reachable_to_px(pt, free, proj)
            if candidate != door_px:
                far_px = candidate
                break
    return {door_side_id: door_px, far_id: far_px}


def route_loop(
    free: np.ndarray,
    waypoints: dict[str, tuple[int, int]],
) -> dict[str, list[list[int]]]:
    cross_keys = {_pair_key(a, b) for a, b in CROSS_ROOM_PAIRS}
    paths: dict[str, list[list[int]]] = {}
    for a, b in LOOP_PAIRS:
        start = waypoints[a]
        goal = waypoints[b]
        key = _pair_key(a, b)
        if key in cross_keys:
            route = pixel_shortest_path(free, start, goal)
        else:
            route = grid_shortest_path(free, start, goal)
        if route is None:
            continue
        paths[key] = [[int(x), int(y)] for x, y in route]
    return paths


def _agent_radius_px(proj: dict[str, Any]) -> int:
    mpp_x = (2.0 * float(proj["half_extent_x"])) / float(proj["image_width"])
    mpp_z = (2.0 * float(proj["half_extent_z"])) / float(proj["image_height"])
    return max(1, int(math.ceil(AGENT_RADIUS_M / min(mpp_x, mpp_z))))


def paint_debris_on_mask(
    free: np.ndarray,
    event,
    blocking_ids: list[str],
    proj: dict[str, Any],
    *,
    erode: bool = True,
) -> np.ndarray:
    blocked = free.copy()
    if not blocking_ids:
        return blocked
    h, w = blocked.shape
    pad = _agent_radius_px(proj)
    id_set = set(blocking_ids)
    for obj in event.metadata.get("objects") or []:
        oid = obj.get("objectId")
        if oid not in id_set:
            continue
        ox0, ox1, oz0, oz1 = _obj_xz_bounds(obj)
        corners = [(ox0, oz0), (ox1, oz0), (ox1, oz1), (ox0, oz1)]
        us = [world_to_map_px(x, z, proj)[0] for x, z in corners]
        vs = [world_to_map_px(x, z, proj)[1] for x, z in corners]
        x0 = max(0, int(min(us)) - pad)
        x1 = min(w - 1, int(max(us)) + pad)
        y0 = max(0, int(min(vs)) - pad)
        y1 = min(h - 1, int(max(vs)) + pad)
        blocked[y0 : y1 + 1, x0 : x1 + 1] = False
    if not erode:
        return blocked
    eroded = cv2.erode(blocked.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    return eroded.astype(bool)


def extract_overhead_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    div = 4
    frames: list[np.ndarray] = []
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError(f"could not read frame from {video_path}")
    h, w = first.shape[:2]
    panel_w = (w - div) // 2
    overhead_slice = slice(panel_w + div, w)
    frames.append(first[:, overhead_slice].copy())
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame[:, overhead_slice].copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"no overhead frames from {video_path}")
    return frames


def _frame_viewpoint_toward_doorway(
    controller,
    hazard: EarthquakeHazard,
    *,
    doorway: dict[str, float],
    room: dict[str, Any],
    house: dict[str, Any],
) -> None:
    """Prefer an FPV that faces the primary doorway on the connector floor."""
    vp = hazard.viewpoint
    if vp is not None and distance_xz(
        float(vp["x"]), float(vp["z"]), float(doorway["x"]), float(doorway["z"])
    ) <= VIEW_DOOR_MAX_DIST_M:
        return
    base_y = _room_floor_base_y(house, room)
    cx = float(doorway["x"])
    cz = float(doorway["z"])
    rc = room_centroid(room)
    dx = float(rc["x"]) - cx
    dz = float(rc["z"]) - cz
    dist = math.hypot(dx, dz) or 1.0
    px = cx + (dx / dist) * VIEW_BACK_FROM_DOOR_M
    pz = cz + (dz / dist) * VIEW_BACK_FROM_DOOR_M
    py = base_y + 0.95
    face_x = cx - px
    face_z = cz - pz
    yaw = math.degrees(math.atan2(face_x, face_z))
    teleport(controller, px, py, pz, yaw=yaw)
    controller.step(action="LookDown", degrees=18.0)
    hazard.viewpoint = {"x": px, "y": py, "z": pz}
    hazard.view_yaw = yaw


def run_earthquake_sim(
    controller,
    config: HazardConfig,
    *,
    house: dict[str, Any],
    door: dict[str, Any],
    room_a: dict[str, Any],
    room_b: dict[str, Any],
    out_video: Path,
) -> dict[str, Any]:
    profile = dict(SCENE_CAPTURE_PROFILES["earthquake"])
    fps = int(profile["fps"])
    substeps = int(profile["substeps"])
    hold_paused = bool(profile["hold_paused"])
    time_step = float(profile["time_step"])

    _verify_native_earthquake(controller)
    primary_base_y = _room_floor_base_y(house, room_a)
    has_overhead = add_overhead_camera_at_floor(controller, primary_base_y)
    hazard = EarthquakeHazard(config)
    hazard.setup(controller)

    door_frame = door_world_frame(door, house)
    doorway = door_world_center(door, house)
    _frame_viewpoint_toward_doorway(
        controller,
        hazard,
        doorway=doorway,
        room=room_a,
        house=house,
    )
    corridor = doorway_corridor_rect(door_frame, depth_m=CORRIDOR_DEPTH_M)
    connector_id = str(door.get("id") or "connector")
    staged_ids = stage_doorway_blockers(controller, door_frame, room_a)

    map_props_event = controller.step(action="GetMapViewCameraProperties")
    map_props = map_props_event.metadata.get("actionReturn") or {}

    block_frame: int | None = None
    blocking_object_ids: list[str] = []
    overhead_frames: list[np.ndarray] = []

    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    trace: list[dict[str, Any]] = []
    peak_moving = 0
    used_fallback_push = False

    if hold_paused:
        pause_physics(controller)

    for step in range(hazard.total_steps()):
        report = hazard.tick(controller, step)
        shift = tuple(getattr(hazard, "render_shift", (0, 0)))

        blocked_now, near_ids = doorway_blocked(
            controller.last_event,
            corridor,
            floor_y_max=primary_base_y + 2.5,
        )
        last_resort_start = config.onset_step + config.total_ticks - 6
        if (
            staged_ids
            and not blocked_now
            and step >= last_resort_start
            and step >= config.onset_step
        ):
            push_toward_point(
                controller,
                doorway,
                magnitude=TOPPLE_PUSH,
                radius=TOPPLE_PUSH_RADIUS,
                object_ids=staged_ids,
            )
            used_fallback_push = True

        if blocked_now and block_frame is None and step >= config.onset_step:
            block_frame = step * substeps
            blocking_object_ids = near_ids

        report = {
            **report,
            "passage_blocked": blocked_now,
            "blocking_object_ids": blocking_object_ids,
            "doorway_blocked": blocked_now,
            "native_earthquake": config.native_effects,
            "staged_object_ids": staged_ids,
        }
        peak_moving = max(peak_moving, int(report.get("num_moving") or 0))

        for _ in range(substeps):
            lines = _label_lines("earthquake", step, report)
            event = advance_physics(controller, time_step)
            frame = _compose_frame(event, 0.0, lines, fpv_shift=shift, graph_ctx=None)
            if frame is None:
                continue
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fh, fw = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_video), fourcc, fps, (fw, fh))
            writer.write(frame)
            frames_written += 1
            tpf = getattr(event, "third_party_camera_frames", None)
            if tpf:
                overhead_frames.append(rgb_to_bgr_uint8(tpf[0]).copy())

        trace.append(report)

    final_info = hazard.finalize(controller)
    tail_report = trace[-1] if trace else {}
    tail_lines = _label_lines("earthquake", hazard.total_steps() - 1, tail_report)
    for _ in range(TAIL_FRAMES):
        event = controller.step(action="Pass")
        frame = _compose_frame(event, 0.0, tail_lines, graph_ctx=None)
        if frame is not None and writer is not None:
            writer.write(frame)
            frames_written += 1
            tpf = getattr(event, "third_party_camera_frames", None)
            if tpf:
                overhead_frames.append(rgb_to_bgr_uint8(tpf[0]).copy())

    if writer is not None:
        writer.release()
    if not out_video.is_file() or frames_written <= 0:
        raise RuntimeError(f"no frames written to {out_video}")
    finalize_mp4(out_video)

    if block_frame is None:
        blocked_now, near_ids = doorway_blocked(
            controller.last_event,
            corridor,
            floor_y_max=primary_base_y + 2.5,
        )
        if blocked_now:
            block_frame = max(0, frames_written - 1)
            blocking_object_ids = near_ids
        else:
            raise RuntimeError("earthquake did not block the doorway before sim end")

    return {
        "frames_written": frames_written,
        "overhead_frames": overhead_frames,
        "trace": trace,
        "final_info": final_info,
        "has_overhead": has_overhead,
        "block_frame": block_frame,
        "blocking_object_ids": blocking_object_ids,
        "connector_id": connector_id,
        "doorway": doorway,
        "door_frame": door_frame,
        "corridor_corners": corridor,
        "map_props": map_props,
        "room_a": room_a,
        "room_b": room_b,
        "primary_floor_base_y": primary_base_y,
        "house_schema": house_schema(house),
        "staged_ids": staged_ids,
        "native_earthquake": True,
        "peak_num_moving": peak_moving,
        "used_fallback_push": used_fallback_push,
        "fps": fps,
        "controller": controller,
    }


def _teleport_agent_to_room(controller, house: dict[str, Any], room: dict[str, Any]) -> None:
    base_y = _room_floor_base_y(house, room)
    position = room_centroid(room)
    position["y"] = base_y + 0.95
    event = controller.step(
        action="TeleportFull",
        position=position,
        rotation={"x": 0.0, "y": 0.0, "z": 0.0},
        horizon=0.0,
        standing=True,
    )
    if not event.metadata.get("lastActionSuccess", False):
        raise RuntimeError(
            "TeleportFull failed before traversability analysis: "
            f"{event.metadata.get('errorMessage')}"
        )


def build_traversability_maps(
    controller,
    sim: dict[str, Any],
    overhead: np.ndarray,
    *,
    house: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from core.thor import get_reachable_positions

    h, w = overhead.shape[:2]
    proj = map_projection_from_props(sim["map_props"], w, h)
    primary_base_y = float(sim.get("primary_floor_base_y", 0.0))
    if house is not None and house_schema(house) == "2.0.0":
        _teleport_agent_to_room(controller, house, sim["room_a"])
    reachable = get_reachable_positions(controller)
    if house is not None and house_schema(house) == "2.0.0":
        reachable = reachable_on_floor(reachable, primary_base_y)
    if not reachable:
        reachable = get_reachable_positions(controller)
    free_open = build_reachable_free_mask(reachable, proj, h, w)
    if not free_open.any():
        raise RuntimeError("reachable free mask is empty")

    room_a_pts = _reachable_in_room(reachable, sim["room_a"], house=house)
    room_b_pts = _reachable_in_room(reachable, sim["room_b"], house=house)
    door_center = sim["doorway"]
    blocked_ids = sim["blocking_object_ids"] or [
        oid for oid in doorway_blocked(controller.last_event, sim["corridor_corners"])[1]
    ]
    free_blocked_check = paint_debris_on_mask(
        free_open, controller.last_event, blocked_ids, proj, erode=False,
    )
    waypoints = {}
    waypoints.update(
        _door_relative_waypoints(
            room_a_pts, door_center, free_open, proj,
            door_side_id="wp_a1", far_id="wp_a0",
            corridor_corners=sim["corridor_corners"],
            must_be_free=free_blocked_check,
        )
    )
    waypoints.update(
        _door_relative_waypoints(
            room_b_pts, door_center, free_open, proj,
            door_side_id="wp_b0", far_id="wp_b1",
            corridor_corners=sim["corridor_corners"],
            must_be_free=free_blocked_check,
        )
    )

    pair_paths_initial = route_loop(free_open, waypoints)
    if len(pair_paths_initial) < 4:
        raise RuntimeError(f"expected 4 loop paths, got {len(pair_paths_initial)}")

    free_blocked = paint_debris_on_mask(free_open, controller.last_event, blocked_ids, proj, erode=True)
    if sim.get("block_frame") is not None and blocked_ids:
        for x, y in _corridor_pixels(sim["corridor_corners"], proj, h, w):
            free_blocked[y, x] = False
    pair_paths_final = route_loop(free_blocked, waypoints)
    if sim.get("block_frame") is not None and blocked_ids:
        for key in {_pair_key(a, b) for a, b in CROSS_ROOM_PAIRS}:
            pair_paths_final.pop(key, None)

    open_initial = connected_pairs(pair_paths_initial)
    open_final = connected_pairs(pair_paths_final)
    severed = [p for p in open_initial if p not in open_final]
    if not severed:
        raise RuntimeError("no severed path pairs after debris block")

    cross_severed = [
        p for p in severed
        if _pair_key(p[0], p[1]) in {_pair_key(a, b) for a, b in CROSS_ROOM_PAIRS}
    ]
    intra_open = [
        p for p in open_final
        if _pair_key(p[0], p[1]) not in {_pair_key(a, b) for a, b in CROSS_ROOM_PAIRS}
    ]

    block_step = max(0, sim["block_frame"] // max(1, SCENE_CAPTURE_PROFILES["earthquake"]["substeps"]))
    floor_summaries: list[dict[str, Any]] = []
    stair_edges: list[dict[str, str]] = []
    if house is not None and house_schema(house) == "2.0.0":
        all_reachable = get_reachable_positions(controller)
        for floor in house_floors(house):
            floor_id = str(floor.get("id"))
            base_y = float(floor.get("baseY", 0.0))
            floor_pts = reachable_on_floor(all_reachable, base_y)
            floor_summaries.append(
                {
                    "floor_id": floor_id,
                    "base_y": base_y,
                    "room_count": len(rooms_on_floor(house, floor_id)),
                    "reachable_count": len(floor_pts),
                }
            )
        for connector in vertical_connectors(house):
            stair_edges.append(
                {
                    "id": str(connector.get("id")),
                    "lower_room_id": str(connector.get("lowerRoomId")),
                    "upper_room_id": str(connector.get("upperRoomId")),
                    "lower_floor_id": str(connector.get("lowerFloorId")),
                    "upper_floor_id": str(connector.get("upperFloorId")),
                }
            )
    return {
        "path_query": {
            "overhead_image": f"{DEMO_PREFIX}_overhead.png",
            "map_projection": proj,
            "waypoint_pixels": {k: [v[0], v[1]] for k, v in waypoints.items()},
            "pair_paths": {"initial": pair_paths_initial, "final": pair_paths_final},
            "connected_pairs": {"initial": open_initial, "final": open_final},
            "severed_pairs": severed,
            "block_frame": sim["block_frame"],
            "frame_count": len(sim["overhead_frames"]),
            "connector_id": sim["connector_id"],
            "floors": floor_summaries,
            "vertical_connectors": stair_edges,
        },
        "state_log": {
            "connector_id": sim["connector_id"],
            "passage_state": "open -> blocked",
            "block_step": block_step,
            "blocking_object_ids": blocked_ids,
            "staged_object_ids": sim.get("staged_ids") or [],
            "native_earthquake": sim.get("native_earthquake", False),
            "replanning_trigger": {
                "reason": "hallway_passage_blocked",
                "step": block_step,
                "connector_id": sim["connector_id"],
            },
        },
        "cross_severed": cross_severed,
        "intra_open": intra_open,
        "corridor_pixels": _corridor_pixels(sim["corridor_corners"], proj, h, w),
        "door_center_px": world_to_map_px(float(door_center["x"]), float(door_center["z"]), proj),
        "free_blocked": free_blocked_check,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Multi-room earthquake traversability</title>
  <style>
    body { margin: 0; font-family: "IBM Plex Sans", sans-serif; background: #12151a; color: #e8ecf1; }
    main { max-width: 1100px; margin: 0 auto; padding: 28px 20px; }
    canvas { width: 100%; background: #0f1318; border: 1px solid #2a3340; border-radius: 10px; }
    .side { margin-top: 16px; }
    input[type="range"] { width: 100%; }
  </style>
</head>
<body>
  <main>
    <h1>Multi-room hallway traversability</h1>
    <p>Four waypoints loop through two rooms; cross-room paths sever when debris blocks the doorway.</p>
    <canvas id="view" width="760" height="520"></canvas>
    <div class="side">
      <div>Frame: <span id="frame">1</span></div>
      <div>Connector: <span id="conn">—</span></div>
      <div>Cross-room: <span id="passage">open</span></div>
      <input id="scrub" type="range" min="0" max="0" step="1" value="0" />
    </div>
  </main>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const pq = payload.path_query;
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const blockFrame = pq.block_frame ?? 0;
    const frameCount = pq.frame_count ?? 1;
    const severed = new Set((pq.severed_pairs || []).map(p => p.slice().sort().join("|")));
    const scrub = document.getElementById("scrub");
    scrub.max = String(Math.max(0, frameCount - 1));
    document.getElementById("conn").textContent = pq.connector_id || "—";
    const img = new Image();
    img.src = pq.overhead_image;
    function toScreen(u,v){const m=36,pw=220,w=canvas.width-2*m-pw,h=canvas.height-2*m;return[m+u/pq.map_projection.image_width*w,m+v/pq.map_projection.image_height*h];}
    function draw(fi){
      const blocked = fi >= blockFrame;
      ctx.fillStyle="#12151a"; ctx.fillRect(0,0,canvas.width,canvas.height);
      if(img.complete){ctx.drawImage(img,36,36,canvas.width-72-220,canvas.height-72);}
      const paths = pq.pair_paths.initial||{};
      for(const [key,poly] of Object.entries(paths)){
        const isBlocked = blocked && severed.has(key);
        ctx.strokeStyle = isBlocked ? "#f87171" : "#34d399";
        ctx.lineWidth = isBlocked ? 3 : 4;
        ctx.setLineDash(isBlocked ? [8,5] : []);
        ctx.beginPath();
        const [x0,y0]=toScreen(poly[0][0],poly[0][1]); ctx.moveTo(x0,y0);
        for(let i=1;i<poly.length;i++){const [x,y]=toScreen(poly[i][0],poly[i][1]); ctx.lineTo(x,y);}
        ctx.stroke();
      }
      ctx.setLineDash([]);
      for(const [id,px] of Object.entries(pq.waypoint_pixels||{})){
        const [x,y]=toScreen(px[0],px[1]);
        ctx.fillStyle="#60a5fa"; ctx.beginPath(); ctx.arc(x,y,9,0,Math.PI*2); ctx.fill();
      }
      document.getElementById("frame").textContent = String(fi+1)+" / "+frameCount;
      document.getElementById("passage").textContent = blocked ? "blocked" : "open";
    }
    scrub.oninput = () => draw(Number(scrub.value));
    img.onload = () => draw(0);
    draw(0);
  </script>
</body>
</html>
"""


def render_html(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__PAYLOAD__", blob)


def _path_through_corridor(
    path: list[list[int]] | None,
    corridor_pixels: set[tuple[int, int]],
) -> bool:
    if not path:
        return False
    for pt in path:
        if (int(pt[0]), int(pt[1])) in corridor_pixels:
            return True
    return False


def _self_check(
    payload: dict[str, Any],
    maps: dict[str, Any],
    *,
    video_frame_count: int,
    image_width: int,
    image_height: int,
    peak_num_moving: int,
) -> None:
    pq = payload["path_query"]
    assert payload["state_log"]["native_earthquake"], "native earthquake required"
    assert peak_num_moving > 0, "quake must move objects"
    assert payload["state_log"]["passage_state"] == "open -> blocked"
    assert len(pq["waypoint_pixels"]) == 4, "need 4 waypoints"
    assert len(pq["pair_paths"]["initial"]) >= 4, "need 4 initial loop paths"
    assert pq["severed_pairs"], "need severed pairs"
    assert maps["cross_severed"], "cross-room pair must sever"
    assert maps["intra_open"], "intra-room pair must stay open"
    assert pq["block_frame"] is not None and pq["block_frame"] < video_frame_count
    assert video_frame_count == pq["frame_count"]

    du, dv = maps["door_center_px"]
    assert 0 <= du < image_width and 0 <= dv < image_height, "door center must be on map"

    corridor = maps["corridor_pixels"]
    initial = pq["pair_paths"]["initial"]
    cross_keys = {_pair_key(a, b) for a, b in CROSS_ROOM_PAIRS}
    cross_routed = [
        key for key in initial
        if key in cross_keys and _path_through_corridor(initial[key], corridor)
    ]
    assert len(cross_routed) >= 1, "cross-room route must pass through door corridor"

    free_blocked = maps["free_blocked"]
    for wid in ("wp_a1", "wp_b0"):
        px = tuple(pq["waypoint_pixels"][wid])
        assert free_blocked[px[1], px[0]], f"{wid} must stay free in blocked mask"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", type=Path, default=default_house_path())
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--onset", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=55)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--local-executable",
        type=Path,
        default=default_local_executable(),
        help="Custom AI2-THOR build with native hazard actions",
    )
    args = parser.parse_args()

    if not args.local_executable.is_file():
        raise RuntimeError(
            f"custom executable not found: {args.local_executable}\n"
            "Run: ./ai2thor_custom/build_local.sh"
        )

    house = load_house_json(args.house)
    scene_id = house_scene_id(house)
    door, room_a, room_b = pick_primary_connector(house)

    out_dir = hazard_output_dir() / "earthquake"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_video = out_dir / f"{SCENE_PREFIX}_{scene_id}.mp4"
    demo_video = out_dir / f"{DEMO_PREFIX}.mp4"
    demo_json = out_dir / f"{DEMO_PREFIX}.json"
    demo_html = out_dir / f"{DEMO_PREFIX}.html"
    overhead_png = out_dir / f"{DEMO_PREFIX}_overhead.png"

    config = HazardConfig(
        hazard_type="earthquake",
        scene=scene_id,
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
        sim = run_earthquake_sim(
            controller,
            config,
            house=house,
            door=door,
            room_a=room_a,
            room_b=room_b,
            out_video=scene_video,
        )
        overhead_frames = sim["overhead_frames"]
        if not overhead_frames:
            overhead_frames = extract_overhead_frames(scene_video)
            sim["overhead_frames"] = overhead_frames
        cv2.imwrite(str(overhead_png), overhead_frames[0])

        maps = build_traversability_maps(controller, sim, overhead_frames[0], house=house)
        payload = {"path_query": maps["path_query"], "state_log": maps["state_log"]}
        demo_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        demo_html.write_text(render_html(payload), encoding="utf-8")

        video_info = render_demo_video(
            overhead_frames,
            corners_px={k: (v[0], v[1]) for k, v in maps["path_query"]["waypoint_pixels"].items()},
            pair_paths_initial=maps["path_query"]["pair_paths"]["initial"],
            severed_pairs=maps["path_query"]["severed_pairs"],
            block_frame=sim["block_frame"],
            out_path=demo_video,
            connector_id=sim["connector_id"],
        )

        summary = {
            "scene_id": scene_id,
            "house": str(args.house),
            "house_schema": house_schema(house),
            "floor_count": len(house_floors(house)),
            "room_count": len(house.get("rooms") or []),
            "vertical_connector_count": len(vertical_connectors(house)),
            "peak_num_moving": sim["peak_num_moving"],
            "used_fallback_push": sim["used_fallback_push"],
            "connector_id": sim["connector_id"],
            "block_frame": sim["block_frame"],
            "blocking_object_ids": sim["blocking_object_ids"],
            "native_earthquake": True,
            "scene_video": str(scene_video),
            "demo_video": str(demo_video),
            "final": sim["final_info"],
        }
        summary_path = out_dir / f"{SCENE_PREFIX}_{scene_id}.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        _self_check(
            payload,
            maps,
            video_frame_count=video_info["frames_written"],
            image_width=overhead_frames[0].shape[1],
            image_height=overhead_frames[0].shape[0],
            peak_num_moving=int(sim["peak_num_moving"]),
        )
    finally:
        controller.stop()

    print(f"wrote {scene_video}")
    print(f"wrote {demo_video} ({video_info['frames_written']} frames)")
    print(f"wrote {demo_json}")
    print(f"wrote {demo_html}")
    print(f"wrote {overhead_png}")
    print(f"connector {sim['connector_id']} blocked at frame {sim['block_frame']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
