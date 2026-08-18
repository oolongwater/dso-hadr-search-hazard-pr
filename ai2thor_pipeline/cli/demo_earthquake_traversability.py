#!/usr/bin/env python3
"""Corner waypoint path overlay on violent earthquake overhead video."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.video import finalize_mp4, probe_video
from hazard.utils import hazard_output_dir
from scene_graph.export import diff_scene_graph, replay_timeline
from scene_graph.schema import (
    Edge,
    EdgeType,
    FloorGT,
    FloorNode,
    FloorObs,
    HazardState,
    NodeSets,
    ObjectGT,
    ObjectNode,
    RegionGT,
    RegionNode,
    RegionObs,
    SceneGraph,
)
from scene_graph.validators import validate_scene_graph

SCENE_ID = "FloorPlan1"
WAYPOINT_IDS = [
    "obj_waypoint_nw",
    "obj_waypoint_ne",
    "obj_waypoint_sw",
    "obj_waypoint_se",
]
# Perimeter loop only: nw-ne-se-sw-nw
PERIMETER_PAIRS = [
    ("obj_waypoint_nw", "obj_waypoint_ne"),
    ("obj_waypoint_ne", "obj_waypoint_se"),
    ("obj_waypoint_se", "obj_waypoint_sw"),
    ("obj_waypoint_sw", "obj_waypoint_nw"),
]
PAIR_NW_NE = PERIMETER_PAIRS[0]
PAIR_NE_SE = PERIMETER_PAIRS[1]
PAIR_SE_SW = PERIMETER_PAIRS[2]
PAIR_NW_SW = PERIMETER_PAIRS[3]

IDENTITY_QUAT = [0.0, 0.0, 0.0, 1.0]
FLOOR_ID = "floor_0"
REGION_ID = "room_kitchen_0"

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30
PANEL_WIDTH = 220
PLOT_MARGIN = 24
MAP_CENTER_X = 0.015
MAP_CENTER_Z = -0.2
MAP_ORTHO_SIZE = 3.0

OVERHEAD_FILENAME = "demo_earthquake_traversability_overhead.png"
COMPARE_SOURCE = "compare_earthquake_FloorPlan1.mp4"

# Annotated targets on 613×516 overhead (see user mark-up screenshot).
MARK_TARGETS = {
    "obj_waypoint_nw": (280, 88),
    "obj_waypoint_ne": (508, 78),
    "obj_waypoint_sw": (225, 365),
    "obj_waypoint_se": (470, 458),
}
OBSTACLE_RECTS = [
    (200, 140, 410, 350),  # island
    (0, 0, 180, 280),  # left counter
    (0, 410, 400, 516),  # bottom counter (height filled at runtime)
]
OBSTACLE_PAD = 6
EAST_AISLE_BLOCK = (380, 120, 580, 480)
# NW↔SW: descend at x=280, turn west at y=133 into the aisle west of the island.
NW_SW_SOUTH_X = 280
NW_SW_TURN_Y = 133
NW_SW_NORTH_Y_MAX = 131
NW_SW_WEST_X_MIN = 192
NW_SW_Y132_EAST_X = 248  # east of this x, turn west at y=131 not y=132
STOOL_ROI = (410, 200, 520, 380)
PATH_GRID_SCALE = 2
STOOL_DIFF_THRESHOLD = 15.0


def _pose(x: float, y: float, z: float) -> list[float]:
    return [x, y, z, *IDENTITY_QUAT]


def _obj(node_id: str, category: str, x: float, y: float, z: float) -> ObjectNode:
    return ObjectNode(
        id=node_id,
        gt=ObjectGT(
            category=category,
            region=REGION_ID,
            pose=_pose(x, y, z),
            bbox_extent=[0.2, 0.2, 0.2],
            movable=False,
            support_parent=REGION_ID,
        ),
    )


def _reachable_edge(src: str, dst: str) -> Edge:
    return Edge(src=src, dst=dst, type=EdgeType.REACHABLE_FROM)


def _contains(src: str, dst: str) -> Edge:
    return Edge(src=src, dst=dst, type=EdgeType.CONTAINS)


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


# NW↔SW routes through the west aisle (left of the island).
WEST_AISLE_PAIRS = {_pair_key(*PAIR_NW_SW)}


def _display_label(node_id: str) -> str:
    return node_id.rsplit("_", 1)[-1]


def extract_violent_overhead_frames(compare_path: Path) -> list[np.ndarray]:
    """Extract per-frame overhead panels from the bottom row of the compare MP4."""
    cap = cv2.VideoCapture(str(compare_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {compare_path}")
    div = 4
    frames: list[np.ndarray] = []
    ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError(f"could not read frame from {compare_path}")
    h, w = first.shape[:2]
    panel_w = (w - 2 * div) // 3
    bottom = first[h // 2 :, :]
    frames.append(bottom[:, panel_w + div : 2 * panel_w + div].copy())
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        bottom = frame[h // 2 :, :]
        frames.append(bottom[:, panel_w + div : 2 * panel_w + div].copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames extracted from {compare_path}")
    return frames


def save_overhead_still(overhead: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overhead)


def map_projection_for_overhead(overhead: np.ndarray) -> dict[str, Any]:
    img_h, img_w = overhead.shape[:2]
    aspect = img_w / float(max(1, img_h))
    half_z = MAP_ORTHO_SIZE
    half_x = MAP_ORTHO_SIZE * aspect
    return {
        "center_x": MAP_CENTER_X,
        "center_z": MAP_CENTER_Z,
        "half_extent_x": half_x,
        "half_extent_z": half_z,
        "orthographic_size": MAP_ORTHO_SIZE,
        "image_width": img_w,
        "image_height": img_h,
    }


def map_px_to_world(u: float, v: float, proj: dict[str, Any]) -> tuple[float, float]:
    hx = float(proj["half_extent_x"])
    hz = float(proj["half_extent_z"])
    cx = float(proj["center_x"])
    cz = float(proj["center_z"])
    iw = float(proj["image_width"])
    ih = float(proj["image_height"])
    x = cx - hx + (u / iw) * 2.0 * hx
    z = cz + hz - (v / ih) * 2.0 * hz
    return x, z


def world_to_map_px(x: float, z: float, proj: dict[str, Any]) -> tuple[float, float]:
    hx = float(proj["half_extent_x"])
    hz = float(proj["half_extent_z"])
    cx = float(proj["center_x"])
    cz = float(proj["center_z"])
    iw = float(proj["image_width"])
    ih = float(proj["image_height"])
    u = (x - cx + hx) / (2.0 * hx) * iw
    v = (cz + hz - z) / (2.0 * hz) * ih
    return u, v


def _is_counter_pixel(overhead: np.ndarray, y: int, x: int) -> bool:
    """Dark island/counter pixels on the overhead still (not wood floor)."""
    b, g, r = (int(overhead[y, x, 0]), int(overhead[y, x, 1]), int(overhead[y, x, 2]))
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    if luma < 95:
        return True
    return r < 115 and g < 110 and b < 65


def _is_dark_surface(overhead: np.ndarray, y: int, x: int) -> bool:
    """Stricter non-floor check for path validation (granite / dark counter)."""
    b, g, r = (int(overhead[y, x, 0]), int(overhead[y, x, 1]), int(overhead[y, x, 2]))
    return 0.299 * r + 0.587 * g + 0.114 * b < 90


def _cull_counter_pixels(
    free: np.ndarray,
    overhead: np.ndarray,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> None:
    height, width = free.shape
    for y in range(max(0, y0), min(height, y1)):
        for x in range(max(0, x0), min(width, x1)):
            if free[y, x] and _is_counter_pixel(overhead, y, x):
                free[y, x] = False


def build_free_mask(
    height: int,
    width: int,
    overhead: np.ndarray | None = None,
    *,
    block_east: bool = False,
    erode: bool = True,
) -> np.ndarray:
    pad = OBSTACLE_PAD
    free = np.ones((height, width), dtype=bool)
    for x0, y0, x1, y1 in OBSTACLE_RECTS:
        y1_eff = height if y1 > height else y1
        if x0 == 200:  # island — no west pad; west aisle sits at x≈192–199
            free[
                max(0, y0 - pad) : min(height, y1_eff + pad),
                x0 : min(width, x1 + pad),
            ] = False
        else:
            free[
                max(0, y0 - pad) : min(height, y1_eff + pad),
                max(0, x0 - pad) : min(width, x1 + pad),
            ] = False
    if block_east:
        x0, y0, x1, y1 = EAST_AISLE_BLOCK
        y1_eff = min(y1, height)
        x1_eff = min(x1, width)
        free[y0:y1_eff, x0:x1_eff] = False
    free[:8, :] = False
    free[-8:, :] = False
    free[:, :8] = False
    free[:, -8:] = False
    if overhead is not None:
        # Island north face + west-aisle granite (skip y=133 turn row).
        _cull_counter_pixels(free, overhead, y0=88, y1=132, x0=200, x1=NW_SW_SOUTH_X)
        _cull_counter_pixels(free, overhead, y0=134, y1=360, x0=NW_SW_WEST_X_MIN, x1=200)
    if erode:
        eroded = cv2.erode(free.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1)
        return eroded > 0
    return free


def snap_waypoint(
    free: np.ndarray,
    tx: int,
    ty: int,
    *,
    y_min: int = 0,
    y_max: int | None = None,
    x_min: int = 0,
    x_max: int | None = None,
    floor: np.ndarray | None = None,
) -> tuple[int, int]:
    h, w = free.shape
    y_max = h if y_max is None else y_max
    x_hi = w if x_max is None else x_max
    best: tuple[int, int] | None = None
    best_d = float("inf")
    ys, xs = np.where(free)
    for x, y in zip(xs, ys, strict=False):
        xi, yi = int(x), int(y)
        if yi < y_min or yi >= y_max or xi < x_min or xi >= x_hi:
            continue
        if floor is not None:
            b, g, r = (int(floor[yi, xi, 0]), int(floor[yi, xi, 1]), int(floor[yi, xi, 2]))
            if r < 90 or g < 75:
                continue
        d = (xi - tx) ** 2 + (yi - ty) ** 2
        if d < best_d:
            best_d = d
            best = (xi, yi)
    if best is None:
        raise RuntimeError(f"no free cell near ({tx},{ty})")
    return best


def calibrate_corner_pixels(overhead: np.ndarray) -> dict[str, tuple[int, int]]:
    h, w = overhead.shape[:2]
    free = build_free_mask(h, w, overhead, block_east=False)
    return {
        "obj_waypoint_nw": snap_waypoint(free, *MARK_TARGETS["obj_waypoint_nw"], y_max=145, floor=overhead),
        "obj_waypoint_ne": snap_waypoint(free, *MARK_TARGETS["obj_waypoint_ne"], floor=overhead),
        "obj_waypoint_sw": snap_waypoint(
            free,
            *MARK_TARGETS["obj_waypoint_sw"],
            y_min=320,
            y_max=395,
            x_min=175,
            x_max=280,
            floor=overhead,
        ),
        "obj_waypoint_se": snap_waypoint(free, *MARK_TARGETS["obj_waypoint_se"], floor=overhead),
    }


def grid_shortest_path(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    scale: int = PATH_GRID_SCALE,
) -> list[tuple[int, int]] | None:
    h, w = free.shape
    gh, gw = h // scale, w // scale

    def to_cell(px: tuple[int, int]) -> tuple[int, int]:
        return px[1] // scale, px[0] // scale

    grid = np.zeros((gh, gw), dtype=bool)
    for r in range(gh):
        for c in range(gw):
            patch = free[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            grid[r, c] = patch.size > 0 and patch.all()

    sc, gc = to_cell(start), to_cell(goal)
    if not grid[sc[0], sc[1]] or not grid[gc[0], gc[1]]:
        return None

    q: deque[tuple[int, int]] = deque([sc])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {sc: None}
    while q:
        cur = q.popleft()
        if cur == gc:
            break
        r, c = cur
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < gh and 0 <= nc < gw and grid[nr, nc] and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if gc not in prev:
        return None

    pts: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = gc
    while cur is not None:
        r, c = cur
        pts.append((c * scale + scale // 2, r * scale + scale // 2))
        cur = prev[cur]
    pts.reverse()
    return pts


def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx // 2
        while x != x1:
            pts.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        pts.append((x1, y1))
    else:
        err = dy // 2
        while y != y1:
            pts.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        pts.append((x1, y1))
    return pts


def path_polyline_on_free(
    free: np.ndarray,
    poly: list[list[int]],
    overhead: np.ndarray | None = None,
    *,
    west_aisle_surface: bool = False,
) -> bool:
    """Every pixel along the polyline must lie on walkable floor."""
    h, w = free.shape
    for i in range(len(poly) - 1):
        x0, y0 = int(poly[i][0]), int(poly[i][1])
        x1, y1 = int(poly[i + 1][0]), int(poly[i + 1][1])
        for x, y in _bresenham(x0, y0, x1, y1):
            if not (0 <= x < w and 0 <= y < h and free[y, x]):
                return False
            if overhead is None:
                continue
            if west_aisle_surface:
                # Island north cap row: no east–west cut across the countertop at x≥200.
                if y == NW_SW_TURN_Y and x >= 200 and _is_counter_pixel(overhead, y, x):
                    return False
                if y >= 134 and x <= 200 and _is_dark_surface(overhead, y, x):
                    return False
            elif _is_counter_pixel(overhead, y, x):
                return False
    return True


def pixel_shortest_path(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = free.shape
    if not free[start[1], start[0]] or not free[goal[1], goal[0]]:
        return None
    q: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        x, y = cur
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            nxt = (nx, ny)
            if 0 <= nx < w and 0 <= ny < h and free[ny, nx] and nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if goal not in prev:
        return None
    pts: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        pts.append(cur)
        cur = prev[cur]
    pts.reverse()
    return pts


def _path_free_mask(
    free: np.ndarray,
    a: str,
    b: str,
    overhead: np.ndarray | None = None,
) -> np.ndarray:
    """Per-pair occupancy hints for perimeter routing."""
    if _pair_key(a, b) not in WEST_AISLE_PAIRS:
        return free
    blocked = free.copy()
    x0, y0, x1, y1 = EAST_AISLE_BLOCK
    blocked[y0:y1, x0:x1] = False
    # Descend at x=280, turn west at y≈131 — no north shortcuts above the turn row.
    blocked[83:NW_SW_NORTH_Y_MAX, :NW_SW_SOUTH_X] = False
    blocked[83:NW_SW_NORTH_Y_MAX, NW_SW_SOUTH_X + 1 : 380] = False
    # No east–west travel on the island north cap row (y≈133).
    blocked[133, 200 : NW_SW_SOUTH_X + 1] = False
    # Near NW, turn west at y=131 (open floor) instead of y=132 (island north edge).
    blocked[132, NW_SW_Y132_EAST_X : NW_SW_SOUTH_X + 1] = False
    return blocked


def perimeter_paths(
    free: np.ndarray,
    corners: dict[str, tuple[int, int]],
    *,
    overhead: np.ndarray | None = None,
) -> dict[str, list[list[int]]]:
    out: dict[str, list[list[int]]] = {}
    for a, b in PERIMETER_PAIRS:
        pair_key = _pair_key(a, b)
        route_free = _path_free_mask(free, a, b, overhead)
        if pair_key in WEST_AISLE_PAIRS:
            poly = pixel_shortest_path(route_free, corners[a], corners[b])
        else:
            poly = grid_shortest_path(route_free, corners[a], corners[b])
        if not poly:
            continue
        line = [[int(x), int(y)] for x, y in poly]
        check_west = overhead is not None and pair_key in WEST_AISLE_PAIRS
        if path_polyline_on_free(
            free,
            line,
            overhead if check_west else None,
            west_aisle_surface=check_west,
        ):
            out[pair_key] = line
    return out


def connected_pairs(pair_paths: dict[str, list[list[int]]]) -> list[list[str]]:
    pairs: list[list[str]] = []
    for key in pair_paths:
        a, b = key.split("|")
        pairs.append([a, b])
    return pairs


def detect_stool_tip_frame(frames: list[np.ndarray]) -> int:
    """Return first frame index where east stool ROI motion exceeds threshold."""
    if not frames:
        return 0
    x0, y0, x1, y1 = STOOL_ROI
    base = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    baseline = 0.0
    for idx in range(min(5, len(frames))):
        g = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY).astype(np.float32)
        baseline = max(baseline, float(np.abs(g[y0:y1, x0:x1] - base[y0:y1, x0:x1]).mean()))

    for idx, frame in enumerate(frames):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        score = float(np.abs(g[y0:y1, x0:x1] - base[y0:y1, x0:x1]).mean())
        if score > baseline + STOOL_DIFF_THRESHOLD:
            return idx
    return max(0, len(frames) // 3)


def build_graph_from_pairs(
    corners: dict[str, tuple[int, int]],
    pair_paths: dict[str, list[list[int]]],
    proj: dict[str, Any],
    *,
    hazard: HazardState = HazardState.NONE,
    accessible: bool = True,
) -> SceneGraph:
    bounds = [
        [-2.685, 0.0, -2.9],
        [2.715, 0.0, -2.9],
        [2.715, 0.0, 2.5],
        [-2.685, 0.0, 2.5],
    ]
    region = RegionNode(
        id=REGION_ID,
        gt=RegionGT(
            semantic_type="kitchen",
            floor=FLOOR_ID,
            bounds=bounds,
            centroid=[0.015, 0.0, -0.2],
        ),
        obs=RegionObs(explored=True, visible=True, traversable=accessible),
    )
    floor = FloorNode(
        id=FLOOR_ID,
        gt=FloorGT(floor_index=0),
        obs=FloorObs(explored=True, accessible=accessible, hazard_state=hazard),
    )

    objects: list[ObjectNode] = []
    for node_id, (px, py) in corners.items():
        wx, wz = map_px_to_world(px, py, proj)
        objects.append(_obj(node_id, "Waypoint", wx, 0.9, wz))

    edges: list[Edge] = [
        _contains(FLOOR_ID, REGION_ID),
        *(_contains(REGION_ID, obj.id) for obj in objects),
    ]
    for key in pair_paths:
        a, b = key.split("|")
        edges.append(_reachable_edge(a, b))
        edges.append(_reachable_edge(b, a))

    return SceneGraph(
        scene_id=SCENE_ID,
        nodes=NodeSets(floor=[floor], region=[region], object=objects),
        edges=edges,
    )


def build_initial_graph(
    corners: dict[str, tuple[int, int]],
    pair_paths: dict[str, list[list[int]]],
    proj: dict[str, Any],
) -> SceneGraph:
    return build_graph_from_pairs(corners, pair_paths, proj)


def build_final_graph_simple(
    initial: SceneGraph,
    open_pair_keys: set[str],
) -> SceneGraph:
    final = deepcopy(initial)
    final.edges = [
        e
        for e in final.edges
        if e.type != EdgeType.REACHABLE_FROM or _pair_key(e.src, e.dst) in open_pair_keys
    ]
    for floor in final.nodes.floor:
        floor.obs.hazard_state = HazardState.DEBRIS
        floor.obs.accessible = False
        floor.obs.last_updated = 20
    for region in final.nodes.region:
        region.obs.traversable = False
        region.obs.hazard_severity = 0.85
        region.obs.last_updated = 20
    return final


def build_timeline(initial: SceneGraph, final: SceneGraph, *, block_step: int) -> list[dict[str, Any]]:
    return [diff_scene_graph(initial, final, step=block_step)]


def build_payload(
    initial: SceneGraph,
    final: SceneGraph,
    timeline: list[dict[str, Any]],
    *,
    map_projection: dict[str, Any],
    overhead_image: str,
    corners_px: dict[str, tuple[int, int]],
    pair_paths_initial: dict[str, list[list[int]]],
    pair_paths_final: dict[str, list[list[int]]],
    block_frame: int,
    frame_count: int,
) -> dict[str, Any]:
    validation = validate_scene_graph(final)
    open_initial = connected_pairs(pair_paths_initial)
    open_final = connected_pairs(pair_paths_final)
    severed = [p for p in open_initial if p not in open_final]
    return {
        "schema_version": initial.schema_version,
        "scene_id": initial.scene_id,
        "initial": initial.model_dump(mode="json"),
        "final": final.model_dump(mode="json"),
        "timeline": timeline,
        "validation": validation.model_dump(mode="json"),
        "path_query": {
            "waypoint_ids": WAYPOINT_IDS,
            "waypoint_pixels": {k: list(v) for k, v in corners_px.items()},
            "pair_paths": {
                "initial": pair_paths_initial,
                "final": pair_paths_final,
            },
            "connected_pairs": {
                "initial": open_initial,
                "final": open_final,
            },
            "severed_pairs": severed,
            "block_frame": block_frame,
            "frame_count": frame_count,
            "map_projection": map_projection,
            "overhead_image": overhead_image,
            "stool_roi": list(STOOL_ROI),
        },
    }


def _self_check(payload: dict[str, Any], *, video_frame_count: int | None = None) -> None:
    initial = SceneGraph.model_validate(payload["initial"])
    final = SceneGraph.model_validate(payload["final"])
    replayed = replay_timeline(initial, payload["timeline"])
    assert replayed[-1].model_dump(mode="json") == final.model_dump(mode="json")

    pq = payload["path_query"]
    open_initial = {_pair_key(a, b) for a, b in pq["connected_pairs"]["initial"]}
    open_final = {_pair_key(a, b) for a, b in pq["connected_pairs"]["final"]}

    assert len(open_initial) == len(PERIMETER_PAIRS), f"expected {len(PERIMETER_PAIRS)} perimeter paths"
    assert _pair_key(*PAIR_NE_SE) in open_initial, "ne-se should be open initially"
    assert _pair_key(*PAIR_NE_SE) not in open_final, "ne-se should be blocked after tip"
    assert _pair_key(*PAIR_NW_SW) in open_initial, "nw-sw should be open initially"
    assert _pair_key(*PAIR_NW_SW) in open_final, "nw-sw should stay open after tip"
    assert payload["validation"]["ok"] is True

    before = {(e.src, e.dst) for e in initial.edges if e.type == EdgeType.REACHABLE_FROM}
    after = {(e.src, e.dst) for e in final.edges if e.type == EdgeType.REACHABLE_FROM}
    assert before - after, "expected reachable_from edges removed"

    if video_frame_count is not None:
        assert video_frame_count == pq["frame_count"], "video frames must match overhead stream"

    overhead_path = Path(payload["path_query"].get("overhead_image", OVERHEAD_FILENAME))
    if not overhead_path.is_file():
        overhead_path = hazard_output_dir() / "earthquake" / OVERHEAD_FILENAME
    overhead = cv2.imread(str(overhead_path)) if overhead_path.is_file() else None
    free_ref = build_free_mask(
        int(pq["map_projection"]["image_height"]),
        int(pq["map_projection"]["image_width"]),
        overhead,
        block_east=False,
        erode=False,
    )
    for phase_name, paths in pq["pair_paths"].items():
        for key, poly in paths.items():
            check_west = overhead is not None and key in WEST_AISLE_PAIRS
            assert path_polyline_on_free(
                free_ref,
                poly,
                overhead if check_west else None,
                west_aisle_surface=check_west,
            ), (
                f"{phase_name} {key} crosses non-floor"
            )


def _plot_rect(width: int, height: int) -> tuple[int, int, int, int]:
    plot_x = PLOT_MARGIN
    plot_y = PLOT_MARGIN
    plot_w = width - 2 * PLOT_MARGIN - PANEL_WIDTH
    plot_h = height - 2 * PLOT_MARGIN
    return plot_x, plot_y, plot_w, plot_h


def compose_overhead_background(
    overhead: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, int, int, int, int]:
    plot_x, plot_y, plot_w, plot_h = _plot_rect(width, height)
    bg = np.full((height, width, 3), (18, 21, 26), dtype=np.uint8)
    scaled = cv2.resize(overhead, (plot_w, plot_h), interpolation=cv2.INTER_AREA)
    bg[plot_y : plot_y + plot_h, plot_x : plot_x + plot_w] = scaled
    cv2.rectangle(bg, (plot_x, plot_y), (plot_x + plot_w, plot_y + plot_h), (71, 85, 105), 2)
    return bg, plot_x, plot_y, plot_w, plot_h


def _draw_text_box(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font_scale: float = 0.55,
    color: tuple[int, int, int] = (248, 250, 252),
    bg: tuple[int, int, int] = (15, 23, 42),
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + baseline + 2), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def _panel_px(
    px: tuple[int, int],
    plot_x: int,
    plot_y: int,
    plot_w: int,
    plot_h: int,
    src_w: int,
    src_h: int,
) -> tuple[int, int]:
    u, v = px
    return (
        plot_x + int(u / src_w * plot_w),
        plot_y + int(v / src_h * plot_h),
    )


def _scale_polyline(
    poly: list[list[int]],
    plot_x: int,
    plot_y: int,
    plot_w: int,
    plot_h: int,
    src_w: int,
    src_h: int,
) -> list[tuple[int, int]]:
    return [_panel_px((p[0], p[1]), plot_x, plot_y, plot_w, plot_h, src_w, src_h) for p in poly]


def render_overlay_frame(
    overhead: np.ndarray,
    *,
    corners_px: dict[str, tuple[int, int]],
    pair_paths: dict[str, list[list[int]]],
    severed_pairs: list[list[str]],
    block_active: bool,
    frame_idx: int,
    frame_count: int,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    title: str | None = None,
    panel_rows: list[tuple[str, str, tuple[int, int, int]]] | None = None,
    connector_id: str | None = None,
) -> np.ndarray:
    src_h, src_w = overhead.shape[:2]
    img, plot_x, plot_y, plot_w, plot_h = compose_overhead_background(overhead, width, height)
    severed_keys = {_pair_key(a, b) for a, b in severed_pairs}

    for key, poly in pair_paths.items():
        pts = _scale_polyline(poly, plot_x, plot_y, plot_w, plot_h, src_w, src_h)
        if len(pts) < 2:
            continue
        blocked = block_active and key in severed_keys
        color = (80, 80, 255) if blocked else (57, 211, 153)
        for i in range(len(pts) - 1):
            if blocked and i % 2 == 0:
                cv2.line(img, pts[i], pts[i + 1], color, 4, cv2.LINE_AA)
            elif not blocked:
                cv2.line(img, pts[i], pts[i + 1], color, 4, cv2.LINE_AA)
        if blocked:
            mid = pts[len(pts) // 2]
            cv2.line(img, (mid[0] - 8, mid[1] - 8), (mid[0] + 8, mid[1] + 8), color, 2, cv2.LINE_AA)
            cv2.line(img, (mid[0] - 8, mid[1] + 8), (mid[0] + 8, mid[1] - 8), color, 2, cv2.LINE_AA)
            _draw_text_box(img, "blocked", (mid[0] - 28, mid[1] - 22), font_scale=0.5, bg=(40, 20, 20))

    for node_id, px in corners_px.items():
        center = _panel_px(px, plot_x, plot_y, plot_w, plot_h, src_w, src_h)
        cv2.circle(img, center, 14, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, center, 11, (35, 147, 251), -1, cv2.LINE_AA)
        cv2.circle(img, center, 11, (255, 255, 255), 2, cv2.LINE_AA)
        _draw_text_box(img, _display_label(node_id), (center[0] - 18, center[1] - 24), font_scale=0.55)

    connected = len(pair_paths) - (len(severed_keys) if block_active else 0)
    phase = "aisle blocked" if block_active else "paths open"
    if connector_id:
        header = f"Multi-room paths — overhead f{frame_idx + 1}/{frame_count}"
    elif title:
        header = title.format(frame_idx=frame_idx, frame_count=frame_count, frame_no=frame_idx + 1)
    else:
        header = f"Corner paths — violent overhead f{frame_idx + 1}/{frame_count}"
    cv2.putText(
        img,
        header,
        (24, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (248, 250, 252),
        2,
        cv2.LINE_AA,
    )

    panel_x = width - PANEL_WIDTH
    cv2.rectangle(img, (panel_x - 16, 56), (width - 12, height - 24), (27, 32, 40), -1)
    cv2.rectangle(img, (panel_x - 16, 56), (width - 12, height - 24), (42, 51, 64), 1)
    if panel_rows is not None:
        lines = panel_rows
    elif connector_id:
        cross_state = "blocked" if block_active else "open"
        lines = [
            ("Phase", "hallway blocked" if block_active else "paths open", (248, 250, 252)),
            ("Connector", connector_id, (248, 250, 252)),
            (
                "Cross-room pairs",
                cross_state,
                (80, 80, 255) if block_active else (57, 211, 153),
            ),
            ("Frame", str(frame_idx + 1), (248, 250, 252)),
        ]
    else:
        lines = [
            ("Phase", phase, (248, 250, 252)),
            ("Connected pairs", str(connected), (57, 211, 153) if not block_active else (80, 80, 255)),
            ("Frame", str(frame_idx + 1), (248, 250, 252)),
            ("East aisle", "blocked" if block_active else "open", (80, 80, 255) if block_active else (57, 211, 153)),
        ]
    y = 90
    for label, value, color in lines:
        cv2.putText(img, label.upper(), (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (147, 160, 176), 1, cv2.LINE_AA)
        cv2.putText(img, value, (panel_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 52

    return img


def render_demo_video(
    frames: list[np.ndarray],
    *,
    corners_px: dict[str, tuple[int, int]],
    pair_paths_initial: dict[str, list[list[int]]],
    severed_pairs: list[list[str]],
    block_frame: int,
    out_path: Path,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    fps: int = VIDEO_FPS,
    title: str | None = None,
    panel_rows: list[tuple[str, str, tuple[int, int, int]]] | None = None,
    connector_id: str | None = None,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    written = 0

    for idx, overhead in enumerate(frames):
        block_active = idx >= block_frame
        frame = render_overlay_frame(
            overhead,
            corners_px=corners_px,
            pair_paths=pair_paths_initial,
            severed_pairs=severed_pairs,
            block_active=block_active,
            frame_idx=idx,
            frame_count=len(frames),
            width=width,
            height=height,
            title=title,
            panel_rows=panel_rows,
            connector_id=connector_id,
        )
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        writer.write(frame)
        written += 1

    assert writer is not None
    writer.release()
    finalize_mp4(out_path)
    probe = probe_video(out_path)
    return {"output_mp4": str(out_path), "frames_written": written, "fps": fps, "video": probe}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Earthquake corner paths — violent overhead</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #12151a;
      --panel: #1b2028;
      --text: #e8ecf1;
      --muted: #93a0b0;
      --path: #34d399;
      --blocked: #f87171;
      --waypoint: #60a5fa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(1200px 600px at 20% -10%, #1f2937 0%, var(--bg) 55%);
      color: var(--text);
      min-height: 100vh;
    }
    main { max-width: 1100px; margin: 0 auto; padding: 28px 20px 40px; }
    h1 { font-size: 1.55rem; font-weight: 600; margin: 0 0 6px; }
    .lede { color: var(--muted); margin: 0 0 22px; max-width: 72ch; line-height: 1.5; }
    .layout { display: grid; grid-template-columns: 1fr 280px; gap: 18px; }
    @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
    canvas {
      width: 100%; height: auto; display: block;
      background: #0f1318; border: 1px solid #2a3340; border-radius: 10px;
    }
    .side { background: var(--panel); border: 1px solid #2a3340; border-radius: 10px; padding: 16px; }
    .stat { margin-bottom: 14px; }
    .stat label {
      display: block; font-size: 0.72rem; text-transform: uppercase;
      letter-spacing: 0.08em; color: var(--muted); margin-bottom: 4px;
    }
    .stat .value { font-size: 1.05rem; font-weight: 600; }
    .stat .value.bad { color: var(--blocked); }
    .stat .value.good { color: var(--path); }
    input[type="range"] { width: 100%; margin-top: 8px; }
    button {
      width: 100%; margin-top: 10px; padding: 10px 12px;
      border: 1px solid #334155; border-radius: 8px;
      background: #243041; color: var(--text); cursor: pointer; font: inherit;
    }
    button:hover { background: #2c3a4f; }
    .legend { margin-top: 18px; font-size: 0.85rem; color: var(--muted); line-height: 1.7; }
    .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
  </style>
</head>
<body>
  <main>
    <h1>Corner waypoint paths</h1>
    <p class="lede">
      Four corner waypoints with perimeter paths (NW↔NE↔SE↔SW↔NW) on the
      violent earthquake overhead view. When stools tip into the east aisle, the NE↔SE corridor closes.
    </p>
    <div class="layout">
      <canvas id="view" width="760" height="520"></canvas>
      <div class="side">
        <div class="stat"><label>Frame</label><div class="value" id="frame">1</div></div>
        <div class="stat"><label>Connected pairs</label><div class="value good" id="pairs">—</div></div>
        <div class="stat"><label>East aisle</label><div class="value" id="aisle">open</div></div>
        <div class="stat"><label>NE↔SE</label><div class="value good" id="ne-se">open</div></div>
        <input id="scrub" type="range" min="0" max="0" step="1" value="0" />
        <button id="play" type="button">Play</button>
        <div class="legend">
          <div><span class="swatch" style="background:var(--waypoint)"></span>corner waypoint</div>
          <div><span class="swatch" style="background:var(--path)"></span>walkable pair path</div>
          <div><span class="swatch" style="background:var(--blocked)"></span>blocked pair path</div>
        </div>
      </div>
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
    const neSeKey = ["obj_waypoint_ne", "obj_waypoint_se"].sort().join("|");

    const scrub = document.getElementById("scrub");
    scrub.max = String(Math.max(0, frameCount - 1));

    const overheadImg = new Image();
    overheadImg.src = pq.overhead_image || "demo_earthquake_traversability_overhead.png";

    function pairPathsForFrame(fi) {
      return fi >= blockFrame ? (pq.pair_paths.final || {}) : (pq.pair_paths.initial || {});
    }

    function activePairKeys(fi) {
      const paths = pairPathsForFrame(fi);
      return Object.keys(paths);
    }

    function toScreen(u, v) {
      const proj = pq.map_projection;
      const margin = 36, panelW = 220;
      const plotW = canvas.width - 2 * margin - panelW;
      const plotH = canvas.height - 2 * margin;
      return [margin + u / proj.image_width * plotW, margin + v / proj.image_height * plotH];
    }

    function draw(fi) {
      const blocked = fi >= blockFrame;
      const paths = pq.pair_paths.initial || {};
      const margin = 36, panelW = 220;
      const plotW = canvas.width - 2 * margin - panelW;
      const plotH = canvas.height - 2 * margin;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#12151a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (overheadImg.complete && overheadImg.naturalWidth) {
        ctx.drawImage(overheadImg, margin, margin, plotW, plotH);
        ctx.strokeStyle = "#475569";
        ctx.strokeRect(margin, margin, plotW, plotH);
      }

      for (const [key, poly] of Object.entries(paths)) {
        if (poly.length < 2) continue;
        const isBlocked = blocked && severed.has(key);
        ctx.strokeStyle = isBlocked ? "#f87171" : "#34d399";
        ctx.lineWidth = isBlocked ? 3 : 4;
        ctx.setLineDash(isBlocked ? [8, 5] : []);
        ctx.beginPath();
        const [x0, y0] = toScreen(poly[0][0], poly[0][1]);
        ctx.moveTo(x0, y0);
        for (let i = 1; i < poly.length; i++) {
          const [x, y] = toScreen(poly[i][0], poly[i][1]);
          ctx.lineTo(x, y);
        }
        ctx.stroke();
        if (isBlocked) {
          const mid = poly[Math.floor(poly.length / 2)];
          const [mx, my] = toScreen(mid[0], mid[1]);
          ctx.strokeStyle = "#f87171";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(mx - 8, my - 8); ctx.lineTo(mx + 8, my + 8);
          ctx.moveTo(mx - 8, my + 8); ctx.lineTo(mx + 8, my - 8);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);

      for (const [id, px] of Object.entries(pq.waypoint_pixels || {})) {
        const [x, y] = toScreen(px[0], px[1]);
        ctx.fillStyle = "#60a5fa";
        ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(x - 18, y - 26, 36, 14);
        ctx.fillStyle = "#f8fafc";
        ctx.font = "11px IBM Plex Sans, sans-serif";
        ctx.fillText(id.split("_").pop(), x - 14, y - 16);
      }

      const openKeys = activePairKeys(fi);
      document.getElementById("frame").textContent = String(fi + 1) + " / " + frameCount;
      const pairsEl = document.getElementById("pairs");
      pairsEl.textContent = String(openKeys.length);
      pairsEl.className = "value " + (blocked ? "bad" : "good");
      const aisleEl = document.getElementById("aisle");
      aisleEl.textContent = blocked ? "blocked" : "open";
      aisleEl.className = "value " + (blocked ? "bad" : "good");
      const neSeEl = document.getElementById("ne-se");
      const neSeOpen = !blocked || !severed.has(neSeKey);
      neSeEl.textContent = neSeOpen ? "open" : "blocked";
      neSeEl.className = "value " + (neSeOpen ? "good" : "bad");
    }

    scrub.addEventListener("input", () => draw(Number(scrub.value)));
    let timer = null;
    document.getElementById("play").addEventListener("click", () => {
      if (timer) { clearInterval(timer); timer = null; return; }
      let i = Number(scrub.value);
      timer = setInterval(() => {
        i = (i + 1) % frameCount;
        scrub.value = String(i);
        draw(i);
      }, 1000 / 30);
    });

    overheadImg.onload = () => draw(0);
    draw(0);
  </script>
</body>
</html>
"""


def render_html(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__PAYLOAD__", blob)


def default_out_dir() -> Path:
    return hazard_output_dir() / "earthquake"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=default_out_dir())
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    compare_path = args.out_dir / COMPARE_SOURCE
    if not compare_path.is_file():
        compare_path = hazard_output_dir() / "earthquake" / COMPARE_SOURCE

    frames = extract_violent_overhead_frames(compare_path)
    overhead_path = args.out_dir / OVERHEAD_FILENAME
    save_overhead_still(frames[0], overhead_path)
    map_projection = map_projection_for_overhead(frames[0])

    h, w = frames[0].shape[:2]
    corners_px = calibrate_corner_pixels(frames[0])
    overhead = frames[0]
    free_open = build_free_mask(h, w, overhead, block_east=False, erode=False)
    free_blocked = build_free_mask(h, w, overhead, block_east=True, erode=False)
    pair_paths_initial = perimeter_paths(free_open, corners_px, overhead=overhead)
    pair_paths_final = perimeter_paths(free_blocked, corners_px, overhead=overhead)

    block_frame = detect_stool_tip_frame(frames)
    initial = build_initial_graph(corners_px, pair_paths_initial, map_projection)
    open_final_keys = set(pair_paths_final)
    final = build_final_graph_simple(initial, open_final_keys)
    timeline = build_timeline(initial, final, block_step=block_frame)

    payload = build_payload(
        initial,
        final,
        timeline,
        map_projection=map_projection,
        overhead_image=OVERHEAD_FILENAME,
        corners_px=corners_px,
        pair_paths_initial=pair_paths_initial,
        pair_paths_final=pair_paths_final,
        block_frame=block_frame,
        frame_count=len(frames),
    )

    json_path = args.out_dir / "demo_earthquake_traversability.scenegraph.json"
    html_path = args.out_dir / "demo_earthquake_traversability.html"
    video_path = args.out_dir / "demo_earthquake_traversability.mp4"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(render_html(payload), encoding="utf-8")

    video_info: dict[str, Any] | None = None
    if args.video:
        video_info = render_demo_video(
            frames,
            corners_px=corners_px,
            pair_paths_initial=pair_paths_initial,
            severed_pairs=payload["path_query"]["severed_pairs"],
            block_frame=block_frame,
            out_path=video_path,
        )
        _self_check(payload, video_frame_count=video_info["frames_written"])
    else:
        _self_check(payload)

    print(f"wrote {overhead_path}")
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    if video_info:
        print(f"wrote {video_path} ({video_info['frames_written']} frames @ {video_info['fps']} fps)")
    print(
        f"pairs: {len(payload['path_query']['connected_pairs']['initial'])} -> "
        f"{len(payload['path_query']['connected_pairs']['final'])} "
        f"(block frame {block_frame})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
