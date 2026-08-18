"""Scene Action Map (SAM) extraction and rendering for 2D occupancy maps.

Implements Algorithm 1 changepoint extraction and f_ep edge prediction from
Loo & Hsu (2024), with geometric scorers replacing trained phi_node/phi_edge.
"""

from __future__ import annotations

import heapq
import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from core.changepoint import Changepoint, ChangepointExit
from core.procthor_house import (
    TOPPLE_TYPE_RANK,
    _aabb_footprint,
    _aabb_height,
    door_world_frame,
    world_to_map_px,
)


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

BEHAVIOURS: tuple[str, ...] = ("turn-left", "go-forward", "turn-right")
DECISIONS: tuple[str, ...] = ("proceed", "backtrack", "reroute", "goal-unreachable")
NUM_BEHAVIOURS = len(BEHAVIOURS)
BEHAVIOUR_INDEX = {b: i for i, b in enumerate(BEHAVIOURS)}
CLUTTER_RADIUS_M = 1.8
MIN_CLUSTER_OBJECTS = 3
GRID_SIZE_M = 0.25

# Colors BGR for panel rendering
COLOR_FREE = (220, 220, 220)
COLOR_BLOCKED = (45, 45, 45)
COLOR_GRID = (180, 180, 180)
COLOR_CP = (0, 140, 255)  # orange
COLOR_DEST = (255, 80, 80)  # blue-ish
COLOR_AGENT = (0, 220, 0)
COLOR_SEVERED = (0, 0, 220)
COLOR_TRAVERSABLE = (0, 180, 0)
COLOR_ROUTE_REPLAN = (255, 220, 0)
BEHAVIOUR_COLORS: dict[str, tuple[int, int, int]] = {
    "turn-left": (200, 100, 0),
    "go-forward": (0, 180, 0),
    "turn-right": (180, 0, 180),
}

RAY_DIRS = 32
CORRIDOR_CLEAR_PX = 8
OPENING_MIN_DIRS = 3
EDGE_SCORE_FLOOR = 0.35
AGENT_RADIUS_M = 0.3


@dataclass
class SamNode:
    id: str
    px: tuple[int, int]
    world: tuple[float, float]
    kind: str  # changepoint | destination
    score: float = 0.0
    heading_deg: float = 0.0
    source: str = "geometric"  # geometric | doorway | cluster
    door_id: str | None = None
    world_y: float = 0.0
    local_frame: dict[str, Any] = field(default_factory=dict)
    passage_width_m: float = 0.0
    clutter_score: float = 0.0
    block_score: float = 0.0
    cluster_object_ids: list[str] = field(default_factory=list)
    cluster_object_types: list[str] = field(default_factory=list)
    cluster_type_summary: str = ""
    room_ids: list[str] = field(default_factory=list)
    connectivity: str = ""
    decision: str = ""
    decision_frame: str = ""
    blocked: bool = False
    clip: str = ""
    payload_png: str = ""
    views: list[str] = field(default_factory=list)


@dataclass
class SamEdge:
    src: str
    dst: str
    behaviour: str
    length_px: float
    alive: bool = True
    path_px: list[tuple[int, int]] = field(default_factory=list)
    block_reason: str = ""
    length_m: float = 0.0
    clearance_m: float = 0.0
    safety: float = 1.0
    safety_reason: str = ""
    visibility: float = 1.0
    connectivity: str = ""

    @property
    def traversable(self) -> bool:
        return self.alive

    @traversable.setter
    def traversable(self, value: bool) -> None:
        self.alive = bool(value)


@dataclass
class ControllerLabel:
    node_id: str
    behaviour: str
    status: str  # open | blocked | replanned | goal-unreachable
    decision: str = "proceed"
    blocked_edge: str | None = None
    blocked_door_id: str | None = None
    alternatives: list[str] = field(default_factory=list)
    message: str = ""
    decision_frame: str = ""


@dataclass
class DecisionDiff:
    node_id: str
    before: str
    after: str
    status: str
    door_id: str | None = None


@dataclass
class SceneActionMap:
    nodes: dict[str, SamNode] = field(default_factory=dict)
    edges: list[SamEdge] = field(default_factory=list)
    grid_points: list[tuple[int, int, float]] = field(default_factory=list)
    grid_res_px: float = 1.0

    def outgoing(self, node_id: str, *, alive_only: bool = True) -> list[SamEdge]:
        out: list[SamEdge] = []
        for e in self.edges:
            if e.src != node_id:
                continue
            if alive_only and not e.alive:
                continue
            out.append(e)
        return out

    def behaviours_out(self, node_id: str, *, alive_only: bool = True) -> set[str]:
        return {e.behaviour for e in self.outgoing(node_id, alive_only=alive_only)}


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
    mpp = _metres_per_px(proj)
    radius_px = max(1, int(math.ceil(GRID_SIZE_M / mpp)))
    k = radius_px * 2 + 1
    kernel = np.ones((k, k), np.uint8)
    dilated = cv2.dilate(free.astype(np.uint8), kernel, iterations=1)
    return dilated.astype(bool)


def _snap_px_to_free(free: np.ndarray, x: int, y: int, *, radius: int = 12) -> tuple[int, int] | None:
    h, w = free.shape
    best: tuple[int, int] | None = None
    best_d = float("inf")
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and free[ny, nx]:
                d = dx * dx + dy * dy
                if d < best_d:
                    best_d = d
                    best = (nx, ny)
    return best


def _distance_transform_peaks(free: np.ndarray, *, grid_res_px: float) -> list[tuple[int, int]]:
    """Local maxima of distance transform on a lattice — corridor/room centres."""
    if not free.any():
        return []
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    step = max(1, int(round(grid_res_px)))
    h, w = free.shape
    peaks: list[tuple[int, int, float]] = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if not free[y, x]:
                continue
            r = step // 2
            y0, y1 = max(0, y - r), min(h, y + r + 1)
            x0, x1 = max(0, x - r), min(w, x + r + 1)
            patch = dist[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            if dist[y, x] >= float(patch.max()) - 0.5:
                peaks.append((x, y, float(dist[y, x])))
    peaks.sort(key=lambda t: t[2], reverse=True)
    return [(x, y) for x, y, _ in peaks]


def _min_sep_px(proj: dict[str, Any], min_sep_m: float) -> float:
    mpp_x = (2.0 * float(proj["half_extent_x"])) / float(proj["image_width"])
    mpp_z = (2.0 * float(proj["half_extent_z"])) / float(proj["image_height"])
    return max(4.0, min_sep_m / min(mpp_x, mpp_z))


def _metres_per_px(proj: dict[str, Any]) -> float:
    mpp_x = (2.0 * float(proj["half_extent_x"])) / float(proj["image_width"])
    mpp_z = (2.0 * float(proj["half_extent_z"])) / float(proj["image_height"])
    return min(mpp_x, mpp_z)


def _make_local_frame(heading_deg: float) -> dict[str, Any]:
    yaw_rad = math.radians(heading_deg)
    return {
        "yaw_deg": heading_deg,
        "forward": {"x": math.sin(yaw_rad), "z": -math.cos(yaw_rad)},
        "right": {"x": math.cos(yaw_rad), "z": math.sin(yaw_rad)},
    }


def _movable_clutter_weight(obj: dict[str, Any]) -> float:
    foot_x, foot_z = _aabb_footprint(obj)
    area = foot_x * foot_z
    height = _aabb_height(obj)
    aspect = height / max(max(foot_x, foot_z), 0.01)
    otype = str(obj.get("objectType") or "")
    type_bonus = 1.5 if otype in TOPPLE_TYPE_RANK else 1.0
    rank = TOPPLE_TYPE_RANK.get(otype, 10)
    rank_factor = 1.0 / (1.0 + rank * 0.05)
    return area * min(1.0, aspect) * type_bonus * rank_factor


def _cluster_type_summary(types: list[str]) -> str:
    from collections import Counter

    parts: list[str] = []
    for otype, count in Counter(types).most_common():
        parts.append(f"{count}x {otype}" if count > 1 else otype)
    return ", ".join(parts[:6])


def _clutter_at_point(
    objects: list[dict[str, Any]],
    wx: float,
    wz: float,
    *,
    radius_m: float = CLUTTER_RADIUS_M,
) -> tuple[float, list[str], list[str]]:
    score = 0.0
    ids: list[str] = []
    types: list[str] = []
    for obj in objects:
        if not (obj.get("pickupable") or obj.get("moveable")):
            continue
        pos = obj.get("position") or {}
        ox = float(pos.get("x", 0.0))
        oz = float(pos.get("z", 0.0))
        if math.hypot(ox - wx, oz - wz) > radius_m:
            continue
        score += _movable_clutter_weight(obj)
        oid = str(obj.get("objectId") or obj.get("id") or "")
        if not oid:
            continue
        ids.append(oid)
        otype = str(obj.get("objectType") or "")
        if not otype:
            asset = str(obj.get("assetId") or "Movable")
            otype = asset.split("_")[0] if asset else "Movable"
        types.append(otype)
    return score, ids, types


def _cluster_centroid(
    objects: list[dict[str, Any]],
    object_ids: list[str],
) -> tuple[float, float] | None:
    id_set = set(object_ids)
    xs: list[float] = []
    zs: list[float] = []
    for obj in objects:
        oid = str(obj.get("objectId") or "")
        if oid not in id_set:
            continue
        pos = obj.get("position") or {}
        xs.append(float(pos.get("x", 0.0)))
        zs.append(float(pos.get("z", 0.0)))
    if not xs:
        return None
    return sum(xs) / len(xs), sum(zs) / len(zs)


def _choke_at_px(
    dist: np.ndarray,
    px: tuple[int, int],
    proj: dict[str, Any],
) -> tuple[float, float]:
    x, y = px
    h, w = dist.shape
    if not (0 <= x < w and 0 <= y < h):
        return 0.0, 0.0
    half_width_px = float(dist[y, x])
    mpp = _metres_per_px(proj)
    half_width_m = half_width_px * mpp
    passage_width_m = half_width_m * 2.0
    if half_width_m <= 0.0:
        return 0.0, passage_width_m
    ratio = min(half_width_m, AGENT_RADIUS_M) / max(half_width_m, AGENT_RADIUS_M)
    choke = ratio * (1.0 if half_width_m <= AGENT_RADIUS_M * 1.5 else 0.5)
    return choke, passage_width_m


def _narrow_passage_candidates(free: np.ndarray, dist: np.ndarray) -> list[tuple[int, int]]:
    skel = _skeletonize(free)
    h, w = free.shape
    candidates: list[tuple[int, int]] = []
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            if not skel[y, x]:
                continue
            val = float(dist[y, x])
            if val <= 0.0:
                continue
            local = dist[y - 2 : y + 3, x - 2 : x + 3]
            if val <= float(local.min()) + 0.5:
                candidates.append((x, y))
    return candidates


def _enforce_min_sep(nodes: list[SamNode], min_sep_px: float) -> list[SamNode]:
    ranked = sorted(nodes, key=lambda n: n.score, reverse=True)
    kept: list[SamNode] = []
    for node in ranked:
        if all(math.hypot(node.px[0] - k.px[0], node.px[1] - k.px[1]) >= min_sep_px for k in kept):
            kept.append(node)
    return kept


def clutter_choke_changepoints(
    house: dict[str, Any],
    proj: dict[str, Any],
    free: np.ndarray,
    objects: list[dict[str, Any]],
    *,
    world_y: float = 0.0,
    min_sep_m: float = 1.2,
) -> list[SamNode]:
    """Decision nodes at blockable chokes inside movable-object clusters."""
    if not free.any() or not objects:
        return []
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    min_sep_px = _min_sep_px(proj, min_sep_m)
    candidates: list[tuple[tuple[int, int], float, float, list[str], list[str], str | None, list[str], float]] = []

    for door in house.get("doors") or []:
        r0 = str(door.get("room0") or door.get("roomId0") or "")
        r1 = str(door.get("room1") or door.get("roomId1") or "")
        if not r0 or not r1 or r0 == r1:
            continue
        door_id = str(door.get("id") or f"{r0}|{r1}")
        try:
            frame = door_world_frame(door, house)
        except ValueError:
            continue
        cx = float(frame["center"]["x"])
        cz = float(frame["center"]["z"])
        nx = float(frame["normal"]["x"])
        nz = float(frame["normal"]["z"])
        heading = math.degrees(math.atan2(nx, -nz)) % 360.0
        u, v = world_to_map_px(cx, cz, proj)
        px = _snap_px_to_free(free, int(round(u)), int(round(v)))
        if px is None:
            continue
        clutter, c_ids, c_types = _clutter_at_point(objects, cx, cz)
        if len(c_ids) < MIN_CLUSTER_OBJECTS or clutter <= 0.0:
            continue
        choke, passage_w = _choke_at_px(dist, px, proj)
        if choke <= 0.0:
            continue
        block_score = choke * clutter
        candidates.append((px, block_score, passage_w, c_ids, c_types, door_id, [r0, r1], heading, clutter))

    for px in _narrow_passage_candidates(free, dist):
        wx, wz = map_px_to_world(float(px[0]), float(px[1]), proj)
        clutter, c_ids, c_types = _clutter_at_point(objects, wx, wz)
        if len(c_ids) < MIN_CLUSTER_OBJECTS or clutter <= 0.0:
            continue
        choke, passage_w = _choke_at_px(dist, px, proj)
        if choke <= 0.0:
            continue
        _, heading = openness_signature(free, px)
        block_score = choke * clutter
        candidates.append((px, block_score, passage_w, c_ids, c_types, None, [], heading, clutter))

    candidates.sort(key=lambda t: t[1], reverse=True)
    nodes: list[SamNode] = []
    for idx, (px, block_score, passage_w, c_ids, c_types, door_id, room_ids, heading, clutter) in enumerate(candidates):
        wx, wz = map_px_to_world(float(px[0]), float(px[1]), proj)
        centroid = _cluster_centroid(objects, c_ids)
        choke_px = px
        if centroid is not None:
            cx, cz = centroid
            cu, cv = world_to_map_px(cx, cz, proj)
            snapped = _snap_px_to_free(free, int(round(cu)), int(round(cv)))
            if snapped is not None:
                swx, swz = map_px_to_world(float(snapped[0]), float(snapped[1]), proj)
                cwx, cwz = map_px_to_world(float(choke_px[0]), float(choke_px[1]), proj)
                if (
                    math.hypot(swx - cwx, swz - cwz) <= CLUTTER_RADIUS_M
                    and _choke_at_px(dist, snapped, proj)[0] > 0.0
                ):
                    px = snapped
                    wx, wz = swx, swz
                else:
                    blend = 0.35
                    wx = wx * (1.0 - blend) + cx * blend
                    wz = wz * (1.0 - blend) + cz * blend
                    u, v = world_to_map_px(wx, wz, proj)
                    snapped2 = _snap_px_to_free(free, int(round(u)), int(round(v)))
                    if snapped2 is not None:
                        px = snapped2
                        wx, wz = map_px_to_world(float(px[0]), float(px[1]), proj)
            else:
                blend = 0.35
                wx = wx * (1.0 - blend) + cx * blend
                wz = wz * (1.0 - blend) + cz * blend
                u, v = world_to_map_px(wx, wz, proj)
                snapped2 = _snap_px_to_free(free, int(round(u)), int(round(v)))
                if snapped2 is not None:
                    px = snapped2
                    wx, wz = map_px_to_world(float(px[0]), float(px[1]), proj)
        final_clutter, final_ids, final_types = _clutter_at_point(objects, wx, wz)
        if len(final_ids) >= MIN_CLUSTER_OBJECTS and final_clutter > 0.0:
            clutter, c_ids, c_types = final_clutter, final_ids, final_types
            block_score = _choke_at_px(dist, px, proj)[0] * clutter
        source = "doorway" if door_id else "cluster"
        type_summary = _cluster_type_summary(c_types)
        nodes.append(
            SamNode(
                id=f"cp_{idx}",
                px=px,
                world=(wx, wz),
                kind="changepoint",
                score=block_score,
                heading_deg=heading,
                source=source,
                door_id=door_id,
                world_y=world_y,
                local_frame=_make_local_frame(heading),
                passage_width_m=passage_w,
                clutter_score=clutter,
                block_score=block_score,
                cluster_object_ids=list(c_ids),
                cluster_object_types=list(dict.fromkeys(c_types)),
                cluster_type_summary=type_summary,
                room_ids=list(room_ids),
            )
        )

    kept = _enforce_min_sep(nodes, min_sep_px)
    for i, node in enumerate(kept):
        node.id = f"cp_{i}"
    return kept


def doorway_changepoints(
    house: dict[str, Any],
    proj: dict[str, Any],
    free: np.ndarray,
    *,
    offset_m: float = 0.6,
    per_door: int = 1,
) -> list[SamNode]:
    """Changepoints at interior doors: one at centre (default) or two offset along normal."""
    nodes: list[SamNode] = []
    cp_idx = 0
    for door in house.get("doors") or []:
        r0 = str(door.get("room0") or door.get("roomId0") or "")
        r1 = str(door.get("room1") or door.get("roomId1") or "")
        if not r0 or not r1 or r0 == r1:
            continue
        door_id = str(door.get("id") or f"{r0}|{r1}")
        try:
            frame = door_world_frame(door, house)
        except ValueError:
            continue
        cx = float(frame["center"]["x"])
        cz = float(frame["center"]["z"])
        nx = float(frame["normal"]["x"])
        nz = float(frame["normal"]["z"])
        heading = math.degrees(math.atan2(nx, -nz)) % 360.0
        if per_door <= 1:
            u, v = world_to_map_px(cx, cz, proj)
            px = _snap_px_to_free(free, int(round(u)), int(round(v)))
            wx, wz = cx, cz
            if px is None:
                wx = cx + nx * offset_m
                wz = cz + nz * offset_m
                u, v = world_to_map_px(wx, wz, proj)
                px = _snap_px_to_free(free, int(round(u)), int(round(v)))
            if px is None:
                continue
            nodes.append(
                SamNode(
                    id=f"cp_door_{cp_idx}",
                    px=px,
                    world=(wx, wz),
                    kind="changepoint",
                    score=0.95,
                    heading_deg=heading,
                    source="doorway",
                    door_id=door_id,
                )
            )
        else:
            for sign, suffix in ((1.0, "a"), (-1.0, "b")):
                wx = cx + sign * nx * offset_m
                wz = cz + sign * nz * offset_m
                u, v = world_to_map_px(wx, wz, proj)
                px = _snap_px_to_free(free, int(round(u)), int(round(v)))
                if px is None:
                    continue
                nodes.append(
                    SamNode(
                        id=f"cp_door_{cp_idx}_{suffix}",
                        px=px,
                        world=(wx, wz),
                        kind="changepoint",
                        score=0.95,
                        heading_deg=heading,
                        source="doorway",
                        door_id=door_id,
                    )
                )
        cp_idx += 1
    return nodes


def grid_sample_points(free: np.ndarray, res_px: float) -> list[tuple[int, int]]:
    """Algorithm 1 line 1: grid-sampled points from map M with resolution r."""
    h, w = free.shape
    step = max(1, int(round(res_px)))
    pts: list[tuple[int, int]] = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if free[y, x]:
                pts.append((x, y))
    return pts


def _ray_clearance(free: np.ndarray, px: tuple[int, int], angle_deg: float, max_steps: int) -> float:
    x, y = px
    fh, fw = free.shape
    dx = math.cos(math.radians(angle_deg))
    dy = math.sin(math.radians(angle_deg))
    cx, cy = float(x), float(y)
    for step in range(1, max_steps + 1):
        cx += dx
        cy += dy
        ix, iy = int(round(cx)), int(round(cy))
        if ix < 0 or iy < 0 or ix >= fw or iy >= fh:
            return float(step - 1)
        if not free[iy, ix]:
            return float(step - 1)
    return float(max_steps)


def openness_signature(
    free: np.ndarray,
    px: tuple[int, int],
    *,
    max_ray: int | None = None,
) -> tuple[list[float], float]:
    """Return per-direction clearances and dominant corridor heading (deg)."""
    h, w = free.shape
    max_ray = max_ray or min(h, w) // 2
    clearances = [_ray_clearance(free, px, 360.0 * i / RAY_DIRS, max_ray) for i in range(RAY_DIRS)]
    best_i = int(np.argmax(clearances))
    heading = 360.0 * best_i / RAY_DIRS
    return clearances, heading


def changepoint_score(clearances: list[float], *, threshold: float = CORRIDOR_CLEAR_PX) -> float:
    """Geometric stand-in for phi_node(m_p): junction / corner / doorway likelihood."""
    n_sectors = 8
    sector_size = RAY_DIRS // n_sectors
    sector_max = [
        max(clearances[s * sector_size : (s + 1) * sector_size])
        for s in range(n_sectors)
    ]
    mx = max(sector_max)
    if mx < threshold * 0.5:
        return 0.1
    open_sectors = [i for i, v in enumerate(sector_max) if v >= mx * 0.55]
    n_open = len(open_sectors)
    if n_open >= 4:
        return 0.85 + 0.05 * min(n_open, 5)
    if n_open == 3:
        return 0.80
    if n_open == 2:
        a, b = open_sectors
        sep = abs(b - a)
        sep = min(sep, n_sectors - sep)
        if sep >= n_sectors // 2:
            return 0.12
        return 0.72
    if n_open == 1:
        if min(sector_max) < mx * 0.35:
            return 0.65
    return 0.1


def _bearing_deg(src: tuple[int, int], dst: tuple[int, int]) -> float:
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def _angle_diff(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def _edge_length_px(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _skeletonize(free: np.ndarray) -> np.ndarray:
    """Morphological skeleton of the free mask."""
    img = free.astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel.astype(bool)


def _node_heading(adj_indices: list[int], pts: list[tuple[int, int]], idx: int) -> float:
    if not adj_indices:
        return 0.0
    x0, y0 = pts[idx]
    bearings: list[float] = []
    for j in adj_indices:
        x1, y1 = pts[j]
        bearings.append(_bearing_deg((x0, y0), (x1, y1)))
    if len(bearings) == 1:
        return bearings[0]
    bearings.sort()
    return bearings[0]


def _graph_changepoint_score(
    idx: int,
    adj: dict[int, list[int]],
    pts: list[tuple[int, int]],
) -> float:
    neighbours = adj.get(idx, [])
    deg = len(neighbours)
    if deg >= 4:
        return 0.95
    if deg == 3:
        x0, y0 = pts[idx]
        bearings = [_bearing_deg((x0, y0), pts[n]) for n in neighbours]
        for i in range(len(bearings)):
            for j in range(i + 1, len(bearings)):
                if _angle_diff(bearings[i], bearings[j]) >= 150.0:
                    return 0.12
        return 0.88
    if deg == 2:
        x0, y0 = pts[idx]
        b0 = _bearing_deg((x0, y0), pts[neighbours[0]])
        b1 = _bearing_deg((x0, y0), pts[neighbours[1]])
        sep = _angle_diff(b0, b1)
        if 45.0 <= sep <= 135.0:
            n0_deg = len(adj.get(neighbours[0], []))
            n1_deg = len(adj.get(neighbours[1], []))
            if n0_deg == 1 and n1_deg == 1:
                return 0.75
            if n0_deg <= 1 or n1_deg <= 1:
                return 0.12
            return 0.75
        return 0.12
    if deg == 1:
        return 0.15
    return 0.08


def _build_grid_graph(
    free: np.ndarray,
    pts: list[tuple[int, int]],
    *,
    grid_res_px: float,
) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {i: [] for i in range(len(pts))}
    step = max(1, int(round(grid_res_px)))
    index = {p: i for i, p in enumerate(pts)}
    for i, (x, y) in enumerate(pts):
        for dx, dy in (
            (step, 0),
            (-step, 0),
            (0, step),
            (0, -step),
            (step, step),
            (step, -step),
            (-step, step),
            (-step, -step),
        ):
            j = index.get((x + dx, y + dy))
            if j is None or j <= i:
                continue
            if _line_traversable(free, (x, y), pts[j], clearance=1):
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _free_arm_count(free: np.ndarray, cx: int, cy: int, *, min_len: int = 8) -> int:
    arms = 0
    h, w = free.shape
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x, y = cx, cy
        length = 0
        for _ in range(min_len):
            x += dx
            y += dy
            if x < 0 or y < 0 or x >= w or y >= h or not free[y, x]:
                break
            length += 1
        if length >= min_len:
            arms += 1
    return arms


def _interior_junction(free: np.ndarray, cx: int, cy: int, *, margin: int = 5) -> bool:
    h, w = free.shape
    for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin)):
        x, y = cx + dx, cy + dy
        if x < 0 or y < 0 or x >= w or y >= h or not free[y, x]:
            return False
    return True


def _free_junctions(free: np.ndarray, *, min_arm: int = 15) -> list[tuple[int, int]]:
    h, w = free.shape
    mark = np.zeros((h, w), dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not free[y, x]:
                continue
            if _free_arm_count(free, x, y, min_len=min_arm) >= 3:
                mark[y, x] = 1
    if not mark.any():
        return []
    _, labels = cv2.connectedComponents(mark, connectivity=8)
    centroids: list[tuple[int, int]] = []
    for label in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == label)
        cx, cy = int(round(xs.mean())), int(round(ys.mean()))
        if _free_arm_count(free, cx, cy, min_len=min_arm) >= 3 and _interior_junction(free, cx, cy):
            centroids.append((cx, cy))
    return centroids


def _skeleton_arm_count(skel: np.ndarray, cx: int, cy: int, *, min_len: int = 4) -> int:
    arms = 0
    h, w = skel.shape
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        x, y = cx, cy
        length = 0
        for _ in range(min_len):
            x += dx
            y += dy
            if x < 0 or y < 0 or x >= w or y >= h or not skel[y, x]:
                break
            length += 1
        if length >= min_len:
            arms += 1
    return arms


def _skeleton_junctions(skel: np.ndarray) -> list[tuple[int, int]]:
    h, w = skel.shape
    mark = np.zeros((h, w), dtype=np.uint8)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not skel[y, x]:
                continue
            deg = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    if skel[y + dy, x + dx]:
                        deg += 1
            if deg >= 3:
                mark[y, x] = 1
    if not mark.any():
        return []
    _, labels = cv2.connectedComponents(mark, connectivity=8)
    centroids: list[tuple[int, int]] = []
    for label in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == label)
        cx, cy = int(round(xs.mean())), int(round(ys.mean()))
        if _skeleton_arm_count(skel, cx, cy) >= 3:
            centroids.append((cx, cy))
    return centroids


def extract_changepoints(
    free: np.ndarray,
    *,
    grid_res_px: float = 12.0,
    cp_threshold: float = 0.55,
    proj: dict[str, Any] | None = None,
    min_sep_m: float = 1.2,
    reserved_px: list[tuple[int, int]] | None = None,
) -> tuple[list[SamNode], list[tuple[int, int, float]]]:
    """Algorithm 1: grid sample, score, threshold, cluster, argmax."""
    skel = _skeletonize(free)
    peaks = _distance_transform_peaks(free, grid_res_px=grid_res_px)
    junctions = list(dict.fromkeys(_skeleton_junctions(skel) + _free_junctions(free, min_arm=12)))
    grid_pts = list(dict.fromkeys(peaks + junctions + grid_sample_points(free, grid_res_px)))
    if not grid_pts:
        return [], []

    skel_pts = [(x, y) for x, y in grid_pts if skel[y, x]]
    if not skel_pts:
        skel_pts = list(grid_pts)
    skel_index = {p: i for i, p in enumerate(skel_pts)}
    junction_set = set(junctions)

    min_sep = _min_sep_px(proj, min_sep_m) if proj else grid_res_px * 0.8
    reserved = list(reserved_px or [])

    def _too_close(x: int, y: int) -> bool:
        for rx, ry in reserved:
            if math.hypot(x - rx, y - ry) < min_sep:
                return True
        return False

    skel_adj = _build_grid_graph(skel, skel_pts, grid_res_px=grid_res_px)
    scored: list[tuple[int, int, float, float]] = []
    for x, y in grid_pts:
        if (x, y) in junction_set:
            _, heading = openness_signature(free, (x, y))
            scored.append((x, y, 0.95, heading))
            continue
        si = skel_index.get((x, y))
        if si is None:
            # snap scoring to nearest skeleton point
            best_si = None
            best_d = float("inf")
            for j, (sx, sy) in enumerate(skel_pts):
                d = (sx - x) ** 2 + (sy - y) ** 2
                if d < best_d:
                    best_d = d
                    best_si = j
            si = best_si
        graph_sp = _graph_changepoint_score(si or 0, skel_adj, skel_pts) if si is not None else 0.08
        clearances, heading = openness_signature(free, (x, y))
        ray_sp = changepoint_score(clearances)
        sp = max(graph_sp, ray_sp) if ray_sp >= 0.35 else ray_sp
        if si is not None:
            neighbours = skel_adj.get(si, [])
            if len(neighbours) <= 1:
                sp = min(sp, 0.15)
            elif len(neighbours) == 2:
                sx, sy = skel_pts[si]
                b0 = _bearing_deg((sx, sy), skel_pts[neighbours[0]])
                b1 = _bearing_deg((sx, sy), skel_pts[neighbours[1]])
                if _angle_diff(b0, b1) >= 150.0:
                    sp = min(sp, ray_sp)
            if heading == 0.0 and neighbours:
                heading = _node_heading(neighbours, skel_pts, si)
        scored.append((x, y, sp, heading))

    scored_map = {(x, y): s for x, y, s, _ in scored}
    all_grid = grid_sample_points(free, grid_res_px)
    grid_scored = [(x, y, scored_map.get((x, y), 0.08)) for x, y in all_grid]

    filtered = [(x, y, s, h) for x, y, s, h in scored if s > cp_threshold and not _too_close(x, y)]
    if not filtered:
        return [], grid_scored

    h, w = free.shape
    label_img = np.zeros((h, w), dtype=np.int32)
    for x, y, s, _ in filtered:
        label_img[y, x] = 1
    _, labels = cv2.connectedComponents(label_img.astype(np.uint8), connectivity=8)

    nodes: list[SamNode] = []
    for label in range(1, int(labels.max()) + 1):
        cluster = [(x, y, s, hd) for x, y, s, hd in filtered if labels[y, x] == label]
        if not cluster:
            continue
        x, y, s, hd = max(cluster, key=lambda t: t[2])
        if any(math.hypot(x - rx, y - ry) < min_sep for rx, ry in reserved):
            continue
        nodes.append(
            SamNode(
                id=f"cp_{len(nodes)}",
                px=(x, y),
                world=(0.0, 0.0),
                kind="changepoint",
                score=s,
                heading_deg=hd,
                source="geometric",
            )
        )
        reserved.append((x, y))
    return nodes, grid_scored


def _behaviour_for_bearing(heading_deg: float, bearing_deg: float) -> tuple[str, float]:
    """Map bearing relative to corridor axis to DECISION behaviour + confidence."""
    rel = (bearing_deg - heading_deg + 360.0) % 360.0
    if rel > 180.0:
        rel -= 360.0
    if abs(rel) <= 35.0:
        return "go-forward", 1.0 - abs(rel) / 35.0
    if 55.0 <= rel <= 125.0:
        return "turn-right", 1.0 - abs(rel - 90.0) / 35.0
    if -125.0 <= rel <= -55.0:
        return "turn-left", 1.0 - abs(rel + 90.0) / 35.0
    return "go-forward", 0.2


def _path_length_px(path: list[tuple[int, int]]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += _edge_length_px(path[i], path[i + 1])
    return total


def _simplify_path(
    path: list[tuple[int, int]],
    *,
    epsilon: float = 2.0,
    free: np.ndarray | None = None,
    clearance: int = 1,
) -> list[tuple[int, int]]:
    if len(path) < 3:
        return list(path)
    arr = np.array(path, dtype=np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(arr, epsilon, False)
    simplified = [(int(p[0][0]), int(p[0][1])) for p in approx]
    if free is None or len(simplified) < 2:
        return simplified

    def _nearest_index(pt: tuple[int, int]) -> int:
        best_i = 0
        best_d = float("inf")
        for i, p in enumerate(path):
            d = (p[0] - pt[0]) ** 2 + (p[1] - pt[1]) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    out = [simplified[0]]
    for seg in range(len(simplified) - 1):
        a, b = simplified[seg], simplified[seg + 1]
        if _line_traversable(free, a, b, clearance=clearance):
            out.append(b)
            continue
        ia, ib = _nearest_index(a), _nearest_index(b)
        if ia > ib:
            ia, ib = ib, ia
        sub = path[ia : ib + 1]
        if out[-1] == sub[0]:
            out.extend(sub[1:])
        else:
            out.extend(sub)
    deduped = [out[0]]
    for pt in out[1:]:
        if pt != deduped[-1]:
            deduped.append(pt)
    return deduped


def _bfs_path_on_mask(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    scale: int = 4,
) -> list[tuple[int, int]] | None:
    h, w = free.shape
    gh, gw = max(1, h // scale), max(1, w // scale)

    def to_cell(px: tuple[int, int]) -> tuple[int, int]:
        return px[1] // scale, px[0] // scale

    def to_px(cell: tuple[int, int]) -> tuple[int, int]:
        r, c = cell
        return (min(w - 1, c * scale + scale // 2), min(h - 1, r * scale + scale // 2))

    grid = np.zeros((gh, gw), dtype=bool)
    for r in range(gh):
        for c in range(gw):
            patch = free[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale]
            grid[r, c] = patch.size > 0 and patch.any()

    sc, gc = to_cell(start), to_cell(goal)
    if not (0 <= sc[0] < gh and 0 <= sc[1] < gw and grid[sc[0], sc[1]]):
        return None
    if not (0 <= gc[0] < gh and 0 <= gc[1] < gw and grid[gc[0], gc[1]]):
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

    cells: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = gc
    while cur is not None:
        cells.append(cur)
        cur = prev[cur]
    cells.reverse()
    return [to_px(c) for c in cells]


def mask_shortest_path(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    clearance_px: int = 2,
    scale: int = 4,
) -> list[tuple[int, int]] | None:
    """BFS polyline through free mask; eroded first, raw fallback."""
    if start == goal:
        return [start, goal]
    if not free[start[1], start[0]] or not free[goal[1], goal[0]]:
        return None

    eroded = free
    if clearance_px > 0:
        k = max(1, clearance_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(free.astype(np.uint8), kernel).astype(bool)

    path = _bfs_path_on_mask(eroded, start, goal, scale=scale)
    if path is None:
        # ponytail: eroded-BFS-then-raw-BFS fallback can return a path narrower than the
        # agent at doorways; upgrade to a distance-transform-weighted A* if it ever clips a frame.
        path = _bfs_path_on_mask(free, start, goal, scale=scale)
    if path is None:
        return None
    if path[0] != start:
        path = [start] + path
    if path[-1] != goal:
        path = path + [goal]
    validate_mask = eroded if clearance_px > 0 else free
    return _simplify_path(path, free=validate_mask, clearance=max(1, clearance_px))


def _line_traversable(
    free: np.ndarray,
    src: tuple[int, int],
    dst: tuple[int, int],
    *,
    clearance: int = 1,
) -> bool:
    x0, y0 = src
    x1, y1 = dst
    n = max(abs(x1 - x0), abs(y1 - y0), 1)
    h, w = free.shape
    for i in range(n + 1):
        t = i / n
        x = int(round(x0 + t * (x1 - x0)))
        y = int(round(y0 + t * (y1 - y0)))
        for dy in range(-clearance, clearance + 1):
            for dx in range(-clearance, clearance + 1):
                xx, yy = x + dx, y + dy
                if xx < 0 or yy < 0 or xx >= w or yy >= h:
                    return False
                if not free[yy, xx]:
                    return False
    return True


def _mst_edges(
    nodes: list[SamNode],
    pair_paths: dict[tuple[str, str], list[tuple[int, int]]],
) -> list[tuple[str, str, list[tuple[int, int]], float]]:
    """Minimum spanning tree over path-connected node pairs."""
    ids = {n.id for n in nodes}
    parent = {nid: nid for nid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    candidates: list[tuple[float, str, str, list[tuple[int, int]]]] = []
    for (a, b), path in pair_paths.items():
        if a not in ids or b not in ids:
            continue
        candidates.append((_path_length_px(path), a, b, path))
    candidates.sort(key=lambda t: t[0])

    mst: list[tuple[str, str, list[tuple[int, int]], float]] = []
    for length, a, b, path in candidates:
        if find(a) != find(b):
            union(a, b)
            mst.append((a, b, path, length))
    return mst


def predict_edges(
    free: np.ndarray,
    nodes: list[SamNode],
    *,
    agent_radius_px: int = 2,
) -> list[SamEdge]:
    """Build polyline edges via mask BFS; MST ensures connectivity."""
    if len(nodes) < 2:
        return []

    id_to_node = {n.id: n for n in nodes}
    pair_paths: dict[tuple[str, str], list[tuple[int, int]]] = {}
    pair_lengths: dict[tuple[str, str], float] = {}

    for i, ni in enumerate(nodes):
        for nj in nodes[i + 1 :]:
            path = mask_shortest_path(free, ni.px, nj.px, clearance_px=agent_radius_px)
            if path is None:
                path = mask_shortest_path(free, ni.px, nj.px, clearance_px=0)
            if path is None or len(path) < 2:
                continue
            key = (ni.id, nj.id) if ni.id < nj.id else (nj.id, ni.id)
            pair_paths[key] = path
            pair_lengths[key] = _path_length_px(path)

    cps = [n for n in nodes if n.kind == "changepoint"]
    mst_nodes = cps if len(cps) >= 2 else nodes
    mst_pairs = _mst_edges(mst_nodes, pair_paths)

    required: set[tuple[str, str]] = set()
    for a, b, _, _ in mst_pairs:
        required.add((a, b))
        required.add((b, a))

    edges: list[SamEdge] = []
    edge_keys: set[tuple[str, str]] = set()
    used_behaviours: dict[str, set[str]] = {n.id: set() for n in nodes}

    def _pick_behaviour(src_id: str, dst_id: str) -> str:
        src = id_to_node[src_id]
        dst = id_to_node[dst_id]
        bearing = _bearing_deg(src.px, dst.px)
        behaviour, _ = _behaviour_for_bearing(src.heading_deg, bearing)
        if behaviour not in used_behaviours[src_id]:
            return behaviour
        for alt in BEHAVIOURS:
            if alt not in used_behaviours[src_id]:
                return alt
        return behaviour

    def _add_edge(src_id: str, dst_id: str, path: list[tuple[int, int]], *, force: bool = False) -> None:
        if (src_id, dst_id) in edge_keys:
            return
        behaviour = _pick_behaviour(src_id, dst_id)
        if behaviour in used_behaviours[src_id] and not force:
            return
        used_behaviours[src_id].add(behaviour)
        edges.append(
            SamEdge(
                src=src_id,
                dst=dst_id,
                behaviour=behaviour,
                length_px=_path_length_px(path),
                path_px=list(path),
            )
        )
        edge_keys.add((src_id, dst_id))

    for a, b, path, _ in mst_pairs:
        _add_edge(a, b, path, force=True)
        _add_edge(b, a, list(reversed(path)), force=True)

    for vi in nodes:
        ranked: list[tuple[float, SamNode, list[tuple[int, int]]]] = []
        for vj in nodes:
            if vj.id == vi.id:
                continue
            key = (vi.id, vj.id) if vi.id < vj.id else (vj.id, vi.id)
            path = pair_paths.get(key)
            if path is None:
                continue
            ranked.append((pair_lengths[key], vj, path))
        ranked.sort(key=lambda t: t[0])

        for _, vj, path in ranked:
            if (vi.id, vj.id) in edge_keys:
                continue
            bearing = _bearing_deg(vi.px, vj.px)
            behaviour, conf = _behaviour_for_bearing(vi.heading_deg, bearing)
            if conf < EDGE_SCORE_FLOOR:
                continue
            if behaviour in used_behaviours[vi.id]:
                continue
            _add_edge(vi.id, vj.id, path)

    return edges


def attach_world_coords(nodes: list[SamNode], proj: dict[str, Any]) -> None:
    for node in nodes:
        wx, wz = map_px_to_world(float(node.px[0]), float(node.px[1]), proj)
        node.world = (wx, wz)


def _connect_destinations(
    free: np.ndarray,
    nodes: list[SamNode],
    edges: list[SamEdge],
    *,
    agent_radius_px: int,
) -> list[SamEdge]:
    """Link destination nodes to nearest changepoint so routes exist."""
    cps = [n for n in nodes if n.kind == "changepoint"]
    dests = [n for n in nodes if n.kind == "destination"]
    if not cps or not dests:
        return edges
    out = list(edges)
    existing = {(e.src, e.dst) for e in out}
    used_beh: dict[str, set[str]] = {}
    for e in out:
        used_beh.setdefault(e.src, set()).add(e.behaviour)

    def _append_edge(src: SamNode, dst: SamNode, *, allow_dup_behaviour: bool = False) -> None:
        if (src.id, dst.id) in existing:
            return
        path = mask_shortest_path(free, src.px, dst.px, clearance_px=agent_radius_px)
        if path is None:
            path = mask_shortest_path(free, src.px, dst.px, clearance_px=0)
        if path is None or len(path) < 2:
            return
        bearing = _bearing_deg(src.px, dst.px)
        behaviour, _ = _behaviour_for_bearing(src.heading_deg, bearing)
        used = used_beh.get(src.id, set())
        if behaviour in used:
            alt_beh = next((b for b in BEHAVIOURS if b not in used), None)
            if alt_beh is None:
                if not allow_dup_behaviour:
                    return
            else:
                behaviour = alt_beh
        used_beh.setdefault(src.id, set()).add(behaviour)
        out.append(
            SamEdge(
                src=src.id,
                dst=dst.id,
                behaviour=behaviour,
                length_px=_path_length_px(path),
                path_px=list(path),
            )
        )
        existing.add((src.id, dst.id))

    for dest in dests:
        ranked = sorted(cps, key=lambda cp: _edge_length_px(dest.px, cp.px))
        linked = 0
        for cp in ranked[:5]:
            before = len(out)
            _append_edge(dest, cp, allow_dup_behaviour=True)
            _append_edge(cp, dest, allow_dup_behaviour=True)
            if len(out) > before:
                linked += 1
            if linked >= 3:
                break
    return out


def _connect_doorway_pairs(
    free: np.ndarray,
    nodes: list[SamNode],
    edges: list[SamEdge],
) -> list[SamEdge]:
    """Ensure both sides of each doorway are linked."""
    by_door: dict[str, list[SamNode]] = {}
    for n in nodes:
        if n.door_id:
            by_door.setdefault(n.door_id, []).append(n)
    out = list(edges)
    existing = {(e.src, e.dst) for e in out}
    used_beh: dict[str, set[str]] = {}
    for e in out:
        used_beh.setdefault(e.src, set()).add(e.behaviour)

    for door_id, pair in by_door.items():
        if len(pair) != 2:
            continue
        a, b = pair
        path = mask_shortest_path(free, a.px, b.px, clearance_px=0)
        if path is None:
            continue
        for src, dst in ((a, b), (b, a)):
            if (src.id, dst.id) in existing:
                continue
            bearing = _bearing_deg(src.px, dst.px)
            behaviour, _ = _behaviour_for_bearing(src.heading_deg, bearing)
            if behaviour in used_beh.get(src.id, set()):
                continue
            used_beh.setdefault(src.id, set()).add(behaviour)
            out.append(
                SamEdge(
                    src=src.id,
                    dst=dst.id,
                    behaviour=behaviour,
                    length_px=_path_length_px(path),
                    path_px=list(path) if src.id == a.id else list(reversed(path)),
                )
            )
            existing.add((src.id, dst.id))
    return out


def _edge_connectivity_str(sam: SceneActionMap, edge: SamEdge) -> str:
    return (
        f"{_node_display_id(edge.src)} -> {_node_display_id(edge.dst)} "
        f"({edge.length_m:.1f}m, safety={edge.safety:.2f})"
    )


def compute_edge_attributes(
    sam: SceneActionMap,
    free: np.ndarray,
    proj: dict[str, Any],
    *,
    objects: list[dict[str, Any]] | None = None,
) -> None:
    """Fill edge length/clearance/safety/visibility/connectivity from the free mask."""
    if not free.any():
        return
    dist = cv2.distanceTransform(free.astype(np.uint8), cv2.DIST_L2, 5)
    mpp = _metres_per_px(proj)
    h, w = free.shape
    obj_list = objects or []
    for edge in sam.edges:
        path = edge.path_px if edge.path_px else [sam.nodes[edge.src].px, sam.nodes[edge.dst].px]
        edge.length_m = edge.length_px * mpp
        min_clear_px = float("inf")
        visible = 0
        src_px = sam.nodes[edge.src].px
        for px in path:
            x, y = px
            if 0 <= x < w and 0 <= y < h:
                min_clear_px = min(min_clear_px, float(dist[y, x]))
            if _line_traversable(free, src_px, px, clearance=1):
                visible += 1
        edge.clearance_m = min_clear_px * mpp if min_clear_px < float("inf") else 0.0
        edge.visibility = visible / max(len(path), 1)
        edge.safety = min(1.0, edge.clearance_m / max(AGENT_RADIUS_M, 0.01))
        flank_penalty = 0.0
        for obj in obj_list:
            if not (obj.get("pickupable") or obj.get("moveable")):
                continue
            mid_idx = len(path) // 2
            mx, my = path[mid_idx]
            mwx, mwz = map_px_to_world(float(mx), float(my), proj)
            pos = obj.get("position") or {}
            ox = float(pos.get("x", 0.0))
            oz = float(pos.get("z", 0.0))
            if math.hypot(ox - mwx, oz - mwz) <= 1.2:
                flank_penalty += 0.15
        edge.safety = max(0.0, edge.safety - flank_penalty)
        edge.safety_reason = f"clearance {edge.clearance_m:.2f}m"
        if flank_penalty > 0.0:
            edge.safety_reason += f"; flank clutter -{flank_penalty:.2f}"
        edge.connectivity = _edge_connectivity_str(sam, edge)


def update_node_connectivity(sam: SceneActionMap) -> None:
    for node in sam.nodes.values():
        if node.kind != "changepoint":
            continue
        exits = sam.outgoing(node.id, alive_only=True)
        n_trav = len(exits)
        rooms = node.room_ids
        if len(rooms) >= 2:
            room_str = " <-> ".join(rooms)
        elif rooms:
            room_str = rooms[0]
        else:
            room_str = "local passage"
        door_str = f" via {node.door_id}" if node.door_id else ""
        types = node.cluster_type_summary or ", ".join(node.cluster_object_types[:3]) or "movables"
        node.connectivity = (
            f"connects {room_str}{door_str}; {n_trav} traversable exit(s); "
            f"cluster: {types}"
        )


def recompute_edges_from_mask(
    sam: SceneActionMap,
    free: np.ndarray,
    proj: dict[str, Any],
    *,
    objects: list[dict[str, Any]] | None = None,
    agent_radius_px: int = 2,
) -> list[SamEdge]:
    """Refresh edge metrics from the current mask without re-severing debris edges."""
    del agent_radius_px  # severing is handled by sever_edges_by_objects in the demo
    compute_edge_attributes(sam, free, proj, objects=objects)
    update_node_connectivity(sam)
    return [e for e in sam.edges if not e.alive]


def build_scene_action_map(
    free: np.ndarray,
    proj: dict[str, Any],
    *,
    grid_res_px: float = 12.0,
    cp_threshold: float = 0.55,
    min_sep_m: float = 1.2,
    house: dict[str, Any] | None = None,
    destinations: list[tuple[str, tuple[float, float]]] | None = None,
    agent_radius_px: int = 2,
    use_geometric_cps: bool = False,
    per_door_cps: int = 1,
    objects: list[dict[str, Any]] | None = None,
    world_y: float = 0.0,
) -> SceneActionMap:
    """Full offline map-reading: nodes then edges."""
    cps: list[SamNode] = []
    if house is not None and objects:
        cps = clutter_choke_changepoints(
            house,
            proj,
            free,
            objects,
            world_y=world_y,
            min_sep_m=min_sep_m,
        )
        attach_world_coords(cps, proj)
        for node in cps:
            node.world_y = world_y
    elif house is not None:
        door_cps = doorway_changepoints(house, proj, free, per_door=per_door_cps)
        attach_world_coords(door_cps, proj)
        cps = door_cps

    reserved = [n.px for n in cps]
    geom_cps: list[SamNode] = []
    grid_scored: list[tuple[int, int, float]] = []
    if use_geometric_cps:
        geom_cps, grid_scored = extract_changepoints(
            free,
            grid_res_px=grid_res_px,
            cp_threshold=cp_threshold,
            proj=proj,
            min_sep_m=min_sep_m,
            reserved_px=reserved,
        )
        attach_world_coords(geom_cps, proj)
    elif free.any():
        grid_scored = [(x, y, 0.08) for x, y in grid_sample_points(free, grid_res_px)]

    dest_nodes: list[SamNode] = []
    for label, (wx, wz) in destinations or []:
        u, v = world_to_map_px(wx, wz, proj)
        px = (int(round(u)), int(round(v)))
        h, w = free.shape
        if not (0 <= px[0] < w and 0 <= px[1] < h):
            continue
        snap = _snap_px_to_free(free, px[0], px[1])
        if snap:
            px = snap
        _, heading = openness_signature(free, px)
        dest_nodes.append(
            SamNode(
                id=f"dest_{label}",
                px=px,
                world=(wx, wz),
                kind="destination",
                score=1.0,
                heading_deg=heading,
                source="destination",
                world_y=world_y,
                local_frame=_make_local_frame(heading),
            )
        )

    all_nodes = cps + geom_cps + dest_nodes
    edges = predict_edges(free, all_nodes, agent_radius_px=agent_radius_px)
    edges = _connect_doorway_pairs(free, all_nodes, edges)
    edges = _connect_destinations(free, all_nodes, edges, agent_radius_px=agent_radius_px)
    sam = SceneActionMap(
        nodes={n.id: n for n in all_nodes},
        edges=edges,
        grid_points=grid_scored,
        grid_res_px=grid_res_px,
    )
    compute_edge_attributes(sam, free, proj, objects=objects)
    update_node_connectivity(sam)
    return sam


def sever_edges_by_objects(
    sam: SceneActionMap,
    event,
    object_ids: list[str],
    proj: dict[str, Any],
    *,
    pad_px: int = 3,
) -> list[SamEdge]:
    if not object_ids:
        return []
    id_set = set(object_ids)
    severed: list[SamEdge] = []
    for edge in sam.edges:
        if not edge.alive:
            continue
        src = sam.nodes[edge.src].px
        dst = sam.nodes[edge.dst].px
        path = edge.path_px if edge.path_px else [src, dst]
        n = max(len(path) - 1, 1)
        line_pts: set[tuple[int, int]] = set()
        for i in range(n + 1):
            t = i / n
            idx = min(int(t * (len(path) - 1)), len(path) - 1)
            x, y = path[idx]
            for dy in range(-pad_px, pad_px + 1):
                for dx in range(-pad_px, pad_px + 1):
                    line_pts.add((x + dx, y + dy))
        blocked = False
        for obj in event.metadata.get("objects") or []:
            oid = obj.get("objectId")
            if oid not in id_set:
                continue
            bbox = obj.get("axisAlignedBoundingBox") or {}
            center = bbox.get("center") or obj.get("position") or {}
            size = bbox.get("size") or {}
            cx = float(center.get("x", 0.0))
            cz = float(center.get("z", 0.0))
            hx = float(size.get("x", 0.4)) / 2.0
            hz = float(size.get("z", 0.4)) / 2.0
            corners = [(cx - hx, cz - hz), (cx + hx, cz - hz), (cx + hx, cz + hz), (cx - hx, cz + hz)]
            us = [world_to_map_px(x, z, proj)[0] for x, z in corners]
            vs = [world_to_map_px(x, z, proj)[1] for x, z in corners]
            x0, x1 = int(min(us)), int(max(us))
            y0, y1 = int(min(vs)), int(max(vs))
            for px, py in line_pts:
                if x0 <= px <= x1 and y0 <= py <= y1:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            edge.alive = False
            edge.block_reason = "debris"
            severed.append(edge)
    return severed


def sever_nearest_edge(sam: SceneActionMap, px: tuple[int, int]) -> SamEdge | None:
    """ponytail: demo fallback when debris moved but missed edge tubes."""
    best: SamEdge | None = None
    best_d = float("inf")
    for edge in sam.edges:
        if not edge.alive:
            continue
        src = sam.nodes[edge.src].px
        dst = sam.nodes[edge.dst].px
        n = max(abs(dst[0] - src[0]), abs(dst[1] - src[1]), 1)
        for i in range(n + 1):
            t = i / n
            x = src[0] + t * (dst[0] - src[0])
            y = src[1] + t * (dst[1] - src[1])
            d = math.hypot(x - px[0], y - px[1])
            if d < best_d:
                best_d = d
                best = edge
    if best is not None:
        best.alive = False
        best.block_reason = "debris_nearest"
    return best


def sever_edges(
    sam: SceneActionMap,
    blocked: np.ndarray,
    *,
    agent_radius_px: int = 2,
) -> tuple[list[SamEdge], list[str]]:
    """Mark edges whose polyline path is broken on the blocked mask."""
    severed: list[SamEdge] = []
    starved: set[str] = set()
    for edge in sam.edges:
        src = sam.nodes[edge.src]
        dst = sam.nodes[edge.dst]
        path = edge.path_px if edge.path_px else [src.px, dst.px]
        ok = True
        for px in path:
            x, y = px
            h, w = blocked.shape
            if x < 0 or y < 0 or x >= w or y >= h:
                ok = False
                break
            for dy in range(-agent_radius_px, agent_radius_px + 1):
                for dx in range(-agent_radius_px, agent_radius_px + 1):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < w and 0 <= yy < h and not blocked[yy, xx]:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok and edge.alive:
            continue
        if edge.alive:
            edge.block_reason = "mask_blocked"
            severed.append(edge)
        edge.alive = False
        starved.add(edge.src)
    return severed, sorted(starved)


def plan_route(
    sam: SceneActionMap,
    start_id: str,
    goal_id: str,
) -> list[str]:
    """Dijkstra over alive edges (paper planner)."""
    if start_id not in sam.nodes or goal_id not in sam.nodes:
        return []
    dist: dict[str, float] = {start_id: 0.0}
    prev: dict[str, str | None] = {start_id: None}
    heap: list[tuple[float, str]] = [(0.0, start_id)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if u == goal_id:
            break
        for edge in sam.outgoing(u, alive_only=True):
            nd = d + edge.length_px
            if nd < dist.get(edge.dst, float("inf")):
                dist[edge.dst] = nd
                prev[edge.dst] = u
                heapq.heappush(heap, (nd, edge.dst))

    if goal_id not in prev and start_id != goal_id:
        return []
    path: list[str] = []
    cur: str | None = goal_id
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path


def pick_display_route(sam: SceneActionMap, start_id: str, goal_id: str) -> list[str]:
    route = plan_route(sam, start_id, goal_id)

    def _via_changepoint() -> list[str] | None:
        cps = [nid for nid, n in sam.nodes.items() if n.kind == "changepoint"]
        best: list[str] | None = None
        best_hops = float("inf")
        for cp in cps:
            leg1 = plan_route(sam, start_id, cp)
            leg2 = plan_route(sam, cp, goal_id)
            if len(leg1) >= 2 and len(leg2) >= 2:
                hops = len(leg1) + len(leg2) - 1
                if hops < best_hops:
                    best_hops = hops
                    best = leg1[:-1] + leg2
        return best

    if len(route) >= 2:
        has_cp = any(sam.nodes[n].kind == "changepoint" for n in route[1:-1])
        if has_cp or start_id == goal_id:
            return route
        via = _via_changepoint()
        if via is not None:
            return via
        return route
    via = _via_changepoint()
    return via if via is not None else []


def pick_designed_route(
    sam: SceneActionMap,
    start_id: str,
    goal_id: str,
    *,
    min_cps: int = 3,
) -> list[str]:
    """Pick a route that visits as many distinct changepoints as possible."""
    cps = [nid for nid, n in sam.nodes.items() if n.kind == "changepoint"]

    def _cp_count(route: list[str]) -> int:
        return sum(1 for nid in route[1:-1] if sam.nodes[nid].kind == "changepoint")

    def _consider(candidate: list[str] | None) -> None:
        nonlocal best, best_cp_count
        if candidate is None or len(candidate) < 2:
            return
        if len(set(candidate)) != len(candidate):
            return
        count = _cp_count(candidate)
        if count > best_cp_count:
            best_cp_count = count
            best = candidate

    best: list[str] | None = None
    best_cp_count = -1

    for cp in cps:
        leg1 = plan_route(sam, start_id, cp)
        leg2 = plan_route(sam, cp, goal_id)
        if len(leg1) >= 2 and len(leg2) >= 2:
            _consider(leg1[:-1] + leg2)

    for cp_a in cps:
        for cp_b in cps:
            if cp_a == cp_b:
                continue
            leg1 = plan_route(sam, start_id, cp_a)
            leg2 = plan_route(sam, cp_a, cp_b)
            leg3 = plan_route(sam, cp_b, goal_id)
            if len(leg1) >= 2 and len(leg2) >= 2 and len(leg3) >= 2:
                _consider(leg1[:-1] + leg2[:-1] + leg3)

    if best is not None and best_cp_count >= min_cps:
        return best
    return pick_display_route(sam, start_id, goal_id)


def _behaviour_tag(behaviour: str) -> str:
    return {"turn-left": "L", "go-forward": "F", "turn-right": "R"}.get(behaviour, "?")


def controller_labels(sam: SceneActionMap, route: list[str]) -> list[ControllerLabel]:
    """Per-route-node behaviour + edge status for the controller."""
    labels: list[ControllerLabel] = []
    for i, node_id in enumerate(route):
        node = sam.nodes.get(node_id)
        if node is None:
            continue
        if i >= len(route) - 1:
            labels.append(
                ControllerLabel(node_id=node_id, behaviour="", status="open", message="destination")
            )
            continue
        nxt_id = route[i + 1]
        edge = _edge_between(sam, node_id, nxt_id)
        alts = sorted(sam.behaviours_out(node_id, alive_only=True))
        if edge is None:
            leg = plan_route(sam, node_id, nxt_id)
            if len(leg) >= 2:
                labels.append(
                    ControllerLabel(
                        node_id=node_id,
                        behaviour="go-forward",
                        status="open",
                        alternatives=alts,
                        message=f"route -> {nxt_id}",
                    )
                )
                continue
            labels.append(
                ControllerLabel(
                    node_id=node_id,
                    behaviour="go-forward",
                    status="blocked",
                    blocked_edge=f"{node_id}->{nxt_id}",
                    alternatives=alts,
                    message=f"no edge to {nxt_id}",
                )
            )
            continue
        blocked_door = None
        if not edge.alive and node.door_id:
            blocked_door = node.door_id
        dst_node = sam.nodes.get(nxt_id)
        if not edge.alive and dst_node and dst_node.door_id:
            blocked_door = dst_node.door_id
        status = "open" if edge.alive else "blocked"
        msg = f"{edge.behaviour} -> {nxt_id}" if edge.alive else f"blocked: {edge.block_reason or 'severed'}"
        labels.append(
            ControllerLabel(
                node_id=node_id,
                behaviour=edge.behaviour,
                status=status,
                decision="proceed" if edge.alive else "backtrack",
                blocked_edge=f"{edge.src}->{edge.dst}" if not edge.alive else None,
                blocked_door_id=blocked_door,
                alternatives=alts,
                message=msg,
            )
        )
    return labels


def replan_from(
    sam: SceneActionMap,
    current_node_id: str,
    goal_id: str,
) -> tuple[list[str], list[ControllerLabel]]:
    route = plan_route(sam, current_node_id, goal_id)
    if len(route) >= 2:
        labels = controller_labels(sam, route)
        for lb in labels:
            if lb.status == "open" and lb.behaviour:
                lb.status = "replanned"
        return route, labels

    blocked_doors: list[str] = []
    for edge in sam.edges:
        if edge.alive:
            continue
        src = sam.nodes.get(edge.src)
        dst = sam.nodes.get(edge.dst)
        if src and src.door_id:
            blocked_doors.append(src.door_id)
        if dst and dst.door_id:
            blocked_doors.append(dst.door_id)
    door_msg = blocked_doors[0] if blocked_doors else "unknown"
    fail_label = ControllerLabel(
        node_id=current_node_id,
        behaviour="",
        status="goal-unreachable",
        blocked_door_id=door_msg,
        message=f"goal unreachable; blocked door {door_msg}",
    )
    return [], [fail_label]


def _edge_between(sam: SceneActionMap, a: str, b: str) -> SamEdge | None:
    for e in sam.edges:
        if e.src == a and e.dst == b and e.alive:
            return e
    return None


def _edge_path(sam: SceneActionMap, a: str, b: str) -> list[tuple[int, int]] | None:
    for e in sam.edges:
        if e.src == a and e.dst == b:
            if e.path_px:
                return list(e.path_px)
            return [sam.nodes[a].px, sam.nodes[b].px]
    return None


def _route_polyline(sam: SceneActionMap, route: list[str]) -> list[tuple[int, int]]:
    if len(route) < 2:
        return []
    pts: list[tuple[int, int]] = []
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        path = _edge_path(sam, a, b)
        if path is None:
            path = [sam.nodes[a].px, sam.nodes[b].px]
        for j, p in enumerate(path):
            if i > 0 and j == 0:
                continue
            pts.append(p)
    return pts


def _behaviour_display(behaviour: str) -> str:
    return {
        "turn-left": "TURN LEFT",
        "go-forward": "GO FORWARD",
        "turn-right": "TURN RIGHT",
    }.get(behaviour, behaviour.replace("-", " ").upper() or "-")


def _node_display_id(node_id: str) -> str:
    if node_id.startswith("cp_") and node_id[3:].isdigit():
        return f"D{node_id[3:]}"
    return (
        node_id.replace("cp_door_", "D")
        .replace("cp_", "CP")
        .replace("dest_", "")
    )


def decision_diff(
    labels_before: list[ControllerLabel],
    labels_after: list[ControllerLabel],
    sam: SceneActionMap | None = None,
) -> list[DecisionDiff]:
    before_map = {lb.node_id: lb for lb in labels_before}
    after_map = {lb.node_id: lb for lb in labels_after}
    diffs: list[DecisionDiff] = []
    for node_id in sorted(set(before_map) | set(after_map)):
        b = before_map.get(node_id)
        a = after_map.get(node_id)
        if b is not None and a is not None:
            b_beh = b.behaviour or "-"
            a_beh = a.behaviour or "-"
            if b_beh == a_beh and b.status == a.status:
                continue
            before_disp = _behaviour_display(b_beh) if b_beh != "-" else b.status
            after_disp = _behaviour_display(a_beh) if a_beh != "-" else a.status
            status = a.status
        elif a is not None:
            a_beh = a.behaviour or "-"
            before_disp = "-"
            after_disp = _behaviour_display(a_beh) if a_beh != "-" else a.status
            status = a.status
        elif b is not None:
            b_beh = b.behaviour or "-"
            before_disp = _behaviour_display(b_beh) if b_beh != "-" else b.status
            after_disp = "-"
            status = "removed"
        else:
            continue
        door_id = None
        if sam is not None:
            node = sam.nodes.get(node_id)
            if node is not None:
                door_id = node.door_id
        diffs.append(
            DecisionDiff(
                node_id=node_id,
                before=before_disp,
                after=after_disp,
                status=status,
                door_id=door_id,
            )
        )
    return diffs


def route_behaviours(sam: SceneActionMap, route: list[str]) -> list[str]:
    """Route-relative behaviour sequence for controller execution."""
    if len(route) < 2:
        return []
    behaviours: list[str] = []
    for i in range(len(route) - 1):
        edge = _edge_between(sam, route[i], route[i + 1])
        if edge is not None:
            behaviours.append(edge.behaviour)
            continue
        # fallback: compute from triple if middle node
        if i > 0:
            prev = sam.nodes[route[i - 1]]
            cur = sam.nodes[route[i]]
            nxt = sam.nodes[route[i + 1]]
            b_in = _bearing_deg(prev.px, cur.px)
            b_out = _bearing_deg(cur.px, nxt.px)
            beh, _ = _behaviour_for_bearing(b_in, b_out)
            behaviours.append(beh)
        else:
            behaviours.append("go-forward")
    return behaviours


def apply_route_relative_behaviours(
    sam: SceneActionMap,
    route: list[str],
    labels: list[ControllerLabel],
    *,
    entry_bearing_deg: float | None = None,
) -> list[ControllerLabel]:
    """Recompute per-node behaviours from approach/departure bearings on the route."""
    label_map = {lb.node_id: lb for lb in labels}
    out: list[ControllerLabel] = []
    for i, node_id in enumerate(route):
        lb = label_map.get(node_id)
        if lb is None:
            continue
        if i >= len(route) - 1:
            out.append(lb)
            continue
        nxt_id = route[i + 1]
        cur = sam.nodes[node_id]
        nxt = sam.nodes[nxt_id]
        if i > 0:
            prev = sam.nodes[route[i - 1]]
            b_in = _bearing_deg(prev.px, cur.px)
        elif entry_bearing_deg is not None:
            b_in = entry_bearing_deg
        else:
            b_in = _bearing_deg(cur.px, nxt.px)
        b_out = _bearing_deg(cur.px, nxt.px)
        beh, _ = _behaviour_for_bearing(b_in, b_out)
        out.append(
            ControllerLabel(
                node_id=lb.node_id,
                behaviour=beh,
                status=lb.status,
                blocked_edge=lb.blocked_edge,
                blocked_door_id=lb.blocked_door_id,
                alternatives=lb.alternatives,
                message=f"{beh} -> {nxt_id}",
            )
        )
    return out


def behaviour_display(behaviour: str) -> str:
    return _behaviour_display(behaviour)


def decision_display(decision: str, *, target: str = "") -> str:
    if decision == "proceed":
        return "PROCEED"
    if decision == "backtrack":
        return f"BACKTRACK to {target}" if target else "BACKTRACK"
    if decision == "reroute":
        return f"FIND NEW PATH via {target}" if target else "FIND NEW PATH"
    if decision == "goal-unreachable":
        return "GOAL UNREACHABLE"
    return decision.replace("-", " ").upper() or "-"


def decision_frame_text(
    decision: str,
    node: SamNode,
    *,
    world_yaw: float | None = None,
) -> str:
    yaw = node.local_frame.get("yaw_deg", node.heading_deg)
    if decision == "backtrack":
        return f"backtrack: -180 deg in node frame / world yaw {yaw:.0f}"
    if decision == "reroute":
        wy = world_yaw if world_yaw is not None else yaw
        return f"reroute: scan heading {wy:.0f} deg in world frame"
    if decision == "proceed":
        return f"proceed: forward +0 deg in node frame / world yaw {yaw:.0f}"
    return f"{decision} at node frame yaw {yaw:.0f}"


def changepoint_from_sam_node(
    node: SamNode,
    sam: SceneActionMap,
    *,
    label: ControllerLabel | None = None,
    phase: str = "",
    agent: dict[str, float] | None = None,
    agent_path_m: float = 0.0,
    quake_active: bool = False,
    shake_elapsed_s: float = 0.0,
    visit_index: int = 0,
) -> Changepoint:
    decision = label.decision if label and label.decision else node.decision or "proceed"
    decision_frame = (
        label.decision_frame if label and label.decision_frame else node.decision_frame
    )
    motion = _behaviour_display(label.behaviour) if label and label.behaviour else ""
    exits = [
        ChangepointExit(
            src=edge.src,
            dst=edge.dst,
            behaviour=edge.behaviour,
            traversable=edge.alive,
            clearance_m=edge.clearance_m,
            safety=edge.safety,
            visibility=edge.visibility,
        )
        for edge in sam.edges
        if node.id in (edge.src, edge.dst)
    ]
    return Changepoint(
        id=node.id,
        world={"x": node.world[0], "y": node.world_y, "z": node.world[1]},
        heading_deg=float(node.local_frame.get("yaw_deg", node.heading_deg)),
        source=node.source,
        door_id=node.door_id,
        room_ids=list(node.room_ids),
        passage_width_m=node.passage_width_m,
        clutter_score=node.clutter_score,
        block_score=node.block_score,
        cluster_object_ids=list(node.cluster_object_ids),
        cluster_object_types=list(node.cluster_object_types),
        cluster_type_summary=node.cluster_type_summary,
        connectivity=node.connectivity,
        decision=decision,
        decision_frame=decision_frame,
        blocked=node.blocked,
        exits=exits,
        visit_index=visit_index,
        phase=phase,
        agent=dict(agent or {}),
        agent_path_m=agent_path_m,
        quake_active=quake_active,
        shake_elapsed_s=shake_elapsed_s,
        motion=motion,
        clip=node.clip,
        payload_png=node.payload_png,
        views=list(node.views),
    )


def serialize_sam_graph(
    sam: SceneActionMap,
    *,
    route_before: list[str] | None = None,
    route_after: list[str] | None = None,
    decisions: list[dict[str, Any]] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    nodes_out: list[dict[str, Any]] = []
    for node in sam.nodes.values():
        if node.kind == "changepoint":
            cp = changepoint_from_sam_node(node, sam)
            payload = cp.to_dict()
            payload["kind"] = node.kind
            payload["local_frame"] = node.local_frame
            nodes_out.append(payload)
            continue
        nodes_out.append(
            {
                "id": node.id,
                "kind": node.kind,
                "world": {"x": node.world[0], "y": node.world_y, "z": node.world[1]},
                "local_frame": node.local_frame,
                "passage_width_m": round(node.passage_width_m, 3),
                "clutter_score": round(node.clutter_score, 3),
                "block_score": round(node.block_score, 3),
                "cluster_object_ids": list(node.cluster_object_ids),
                "cluster_object_types": list(node.cluster_object_types),
                "cluster_type_summary": node.cluster_type_summary,
                "room_ids": list(node.room_ids),
                "connectivity": node.connectivity,
                "decision": node.decision,
                "decision_frame": node.decision_frame,
                "blocked": node.blocked,
                "clip": node.clip,
                "payload_png": node.payload_png,
                "views": list(node.views),
                "door_id": node.door_id,
                "source": node.source,
                "heading_deg": round(node.heading_deg, 1),
            }
        )
    edges_out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for edge in sam.edges:
        key = (edge.src, edge.dst)
        if key in seen:
            continue
        seen.add(key)
        edges_out.append(
            {
                "nodes": [edge.src, edge.dst],
                "behaviour": edge.behaviour,
                "traversable": edge.traversable,
                "safety": round(edge.safety, 3),
                "safety_reason": edge.safety_reason,
                "visibility": round(edge.visibility, 3),
                "connectivity": edge.connectivity,
                "clearance_m": round(edge.clearance_m, 3),
                "length_m": round(edge.length_m, 3),
                "block_reason": edge.block_reason,
            }
        )
    return {
        "nodes": nodes_out,
        "edges": edges_out,
        "route_before": route_before or [],
        "route_after": route_after or [],
        "decisions": decisions or [],
        "artifacts": artifacts or {},
    }


def _wrap_cv2_text(
    text: str,
    *,
    font: int,
    scale: float,
    thick: int,
    max_width: int,
) -> list[str]:
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        (tw, _), _ = cv2.getTextSize(trial, font, scale, thick)
        if tw <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [text]


def render_decision_card(
    node: SamNode | None,
    label: ControllerLabel | None,
    *,
    width: int = 460,
    height: int = 800,
    cp: Changepoint | None = None,
    header_image: np.ndarray | None = None,
) -> np.ndarray:
    card = np.full((height, width, 3), (36, 36, 36), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if node is None:
        cv2.putText(card, "DECISION", (12, 40), font, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
        return card

    if cp is None:
        cp = changepoint_from_sam_node(node, SceneActionMap(), label=label)

    decision = cp.decision or "proceed"
    lf = node.local_frame or {}
    fwd = lf.get("forward") or {}
    right = lf.get("right") or {}
    local_frame_txt = (
        f"yaw {lf.get('yaw_deg', cp.heading_deg):.0f} deg | "
        f"forward ({float(fwd.get('x', 0)):.2f}, {float(fwd.get('z', 0)):.2f}) | "
        f"right ({float(right.get('x', 0)):.2f}, {float(right.get('z', 0)):.2f})"
    )
    edge_lines = [
        (
            f"  ({e.src or '?'}, {e.dst}) safety {e.safety:.2f} vis {e.visibility:.2f} "
            f"{'open' if e.traversable else 'blocked'} {e.clearance_m:.2f}m"
        )
        for e in cp.exits
    ] or ["  none"]
    metadata_lines = [
        f"door {cp.door_id or '-'}  rooms {', '.join(cp.room_ids) or '-'}",
        f"passage {cp.passage_width_m:.2f} m",
        f"clutter {cp.clutter_score:.2f}  block {cp.block_score:.2f}  n={cp.cluster_size}",
        cp.cluster_type_summary or ", ".join(cp.cluster_object_types[:6]) or "none",
        f"blocked={cp.blocked}",
    ]
    if cp.visit_index or cp.phase:
        metadata_lines.append(f"visit #{cp.visit_index}  phase={cp.phase}")
    if cp.agent:
        metadata_lines.append(
            f"agent ({cp.agent.get('x', 0):.2f}, {cp.agent.get('y', 0):.2f}, {cp.agent.get('z', 0):.2f}) "
            f"yaw {cp.agent.get('yaw', 0):.0f}"
        )
    if cp.motion:
        metadata_lines.append(f"motion: {cp.motion}")
    if cp.quake_active:
        metadata_lines.append(
            f"quake {cp.shake_elapsed_s:.1f}s  path {cp.agent_path_m:.1f}m"
        )
    localisation_lines = []
    if cp.clip:
        localisation_lines.append(f"4-view scan clip -> {cp.clip}")
    if cp.views:
        localisation_lines.append(f"views: {len(cp.views)} captured")

    header_color = (120, 200, 255)
    body_color = (230, 230, 230)
    meta_color = (210, 210, 210)
    edge_color = (180, 220, 180)
    decision_color = (0, 255, 255)

    sections: list[tuple[str, float, int, tuple[int, int, int]]] = [
        (f"NODE {_node_display_id(cp.id)}  ({node.kind}/{cp.source})", 0.78, 2, body_color),
    ]
    if localisation_lines or header_image is not None:
        sections.append(("LOCALISATION", 0.58, 2, header_color))
        sections.extend((line, 0.50, 1, meta_color) for line in localisation_lines)
    sections.extend(
        [
            ("WORLD COORDINATES", 0.58, 2, header_color),
            (
                f"x {cp.world.get('x', 0):.2f}  y {cp.world.get('y', 0):.2f}  z {cp.world.get('z', 0):.2f}",
                0.54,
                1,
                body_color,
            ),
            ("LOCAL FRAME", 0.58, 2, header_color),
            (local_frame_txt, 0.50, 1, body_color),
            ("DECISION WRT FRAME", 0.58, 2, header_color),
            (f"{decision_display(decision)}", 0.56, 1, decision_color),
            (cp.decision_frame, 0.50, 1, meta_color),
            ("METADATA", 0.58, 2, header_color),
            ("\n".join(metadata_lines), 0.50, 1, meta_color),
            ("CONNECTIVITY", 0.58, 2, header_color),
            (cp.connectivity or "none", 0.50, 1, meta_color),
            ("EDGES (src, dst, safety, vis)", 0.58, 2, header_color),
            ("\n".join(edge_lines), 0.48, 1, edge_color),
        ]
    )

    text_top = 34
    if header_image is not None:
        strip = header_image
        if strip.shape[1] != width:
            strip = cv2.resize(strip, (width, int(strip.shape[0] * width / strip.shape[1])))
        sh = min(strip.shape[0], height // 3)
        strip = strip[:sh, :width]
        card[0:sh, 0:width] = strip
        text_top = sh + 12

    margin_x = 10
    max_text_w = width - 2 * margin_x
    scale_factor = 1.0
    for _ in range(6):
        y = text_top
        fits = True
        for text, scale, thick, _color in sections:
            for line in _wrap_cv2_text(
                text.replace("\n", " | "),
                font=font,
                scale=scale * scale_factor,
                thick=thick,
                max_width=max_text_w,
            ):
                (_tw, th), _ = cv2.getTextSize(line, font, scale * scale_factor, thick)
                y += th + 12
                if y >= height - 8:
                    fits = False
                    break
            if not fits:
                break
        if fits:
            break
        scale_factor *= 0.88

    y = text_top
    for text, scale, thick, color in sections:
        for line in _wrap_cv2_text(
            text.replace("\n", " | "),
            font=font,
            scale=scale * scale_factor,
            thick=thick,
            max_width=max_text_w,
        ):
            (tw, th), _ = cv2.getTextSize(line, font, scale * scale_factor, thick)
            cv2.rectangle(card, (8, y - th - 6), (min(width - 4, 12 + tw), y + 6), (20, 20, 20), -1)
            cv2.putText(
                card,
                line,
                (margin_x, y),
                font,
                scale * scale_factor,
                color,
                thick,
                cv2.LINE_AA,
            )
            y += th + 12
            if y >= height - 8:
                return card
    return card


def render_node_payload_png(
    node: SamNode,
    label: ControllerLabel | None,
    views_bgr: list[np.ndarray],
    *,
    view_yaws: list[float] | None = None,
    width: int = 1280,
    height: int = 900,
) -> np.ndarray:
    """Full-size payload sheet: 2x2 labelled views + wrapped metadata."""
    canvas = np.full((height, width, 3), (28, 28, 28), dtype=np.uint8)
    cell_w, cell_h = 620, 360
    x0, y0 = 20, 20
    yaws = view_yaws or [0.0, 90.0, 180.0, 270.0]
    for i, view in enumerate(views_bgr[:4]):
        r, c = divmod(i, 2)
        px = x0 + c * (cell_w + 10)
        py = y0 + r * (cell_h + 10)
        resized = cv2.resize(view, (cell_w, cell_h))
        canvas[py : py + cell_h, px : px + cell_w] = resized
        cv2.rectangle(canvas, (px - 2, py - 2), (px + cell_w + 2, py + cell_h + 2), (0, 255, 255), 2)
        tag = f"{_node_display_id(node.id)}  yaw {yaws[i]:.0f} deg"
        cv2.putText(canvas, tag, (px + 8, py + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    meta_x = 20
    meta_y = y0 + 2 * (cell_h + 10) + 24
    font = cv2.FONT_HERSHEY_SIMPLEX
    decision = label.decision if label and label.decision else node.decision or "proceed"
    meta_lines = [
        f"Decision node {_node_display_id(node.id)}",
        f"World ({node.world[0]:.2f}, {node.world_y:.2f}, {node.world[1]:.2f})",
        f"Passage {node.passage_width_m:.2f} m | clutter {node.clutter_score:.2f} | block {node.block_score:.2f}",
        f"Cluster: {node.cluster_type_summary or ', '.join(node.cluster_object_types) or 'none'}",
        node.connectivity,
        f"Decision: {decision_display(decision)}",
        label.decision_frame if label and label.decision_frame else node.decision_frame,
    ]
    if label and label.behaviour:
        meta_lines.append(f"Motion: {_behaviour_display(label.behaviour)}")
    for text in meta_lines:
        for line in _wrap_cv2_text(text, font=font, scale=0.62, thick=1, max_width=width - 40):
            (tw, th), _ = cv2.getTextSize(line, font, 0.62, 1)
            cv2.putText(canvas, line, (meta_x, meta_y), font, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
            meta_y += th + 10
            if meta_y >= height - 8:
                return canvas
    return canvas


NODES_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SAM decision nodes</title>
  <style>
    body { margin: 0; font-family: "IBM Plex Sans", sans-serif; background: #12151a; color: #e8ecf1; }
    main { max-width: 1200px; margin: 0 auto; padding: 28px 20px; }
    .card { background: #1a1f27; border: 1px solid #2a3340; border-radius: 10px; padding: 16px; margin-bottom: 20px; }
    .card img { max-width: 100%; border-radius: 6px; }
    video { max-width: 100%; border-radius: 6px; background: #000; }
    table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    td { padding: 4px 8px; vertical-align: top; border-bottom: 1px solid #2a3340; }
    td:first-child { color: #94a3b8; width: 160px; }
    h2 { margin: 0 0 8px; font-size: 1.2rem; }
    .timeline li { margin: 4px 0; }
  </style>
</head>
<body>
  <main>
    <h1>SAM decision nodes</h1>
    <p id="summary"></p>
    <div id="cards"></div>
    <section class="card">
      <h2>Decision timeline</h2>
      <ul id="timeline" class="timeline"></ul>
    </section>
  </main>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const nodes = payload.nodes || [];
    const cps = nodes.filter(n => n.kind === "changepoint");
    document.getElementById("summary").textContent =
      `${cps.length} changepoint(s), ${(payload.edges || []).length} edge(s)`;
    const cards = document.getElementById("cards");
    for (const node of cps) {
      const card = document.createElement("section");
      card.className = "card";
      const payloadRel = node.payload_png ? node.payload_png.split("/").pop() : "";
      const clipRel = node.clip ? node.clip.split("/").pop() : "";
      card.innerHTML = `
        <h2>${node.id}</h2>
        ${payloadRel ? `<img src="nodes/${payloadRel}" alt="${node.id} payload" />` : ""}
        ${clipRel ? `<video controls src="nodes/${clipRel}"></video>` : ""}
        <table>
          <tr><td>World</td><td>${node.world.x.toFixed(2)}, ${node.world.y.toFixed(2)}, ${node.world.z.toFixed(2)}</td></tr>
          <tr><td>Passage</td><td>${node.passage_width_m} m</td></tr>
          <tr><td>Clutter / block</td><td>${node.clutter_score} / ${node.block_score}</td></tr>
          <tr><td>Cluster</td><td>${node.cluster_type_summary || (node.cluster_object_types || []).join(", ")}</td></tr>
          <tr><td>Connectivity</td><td>${node.connectivity || ""}</td></tr>
          <tr><td>Decision</td><td>${node.decision || ""}</td></tr>
        </table>`;
      cards.appendChild(card);
    }
    const tl = document.getElementById("timeline");
    for (const row of payload.decisions || []) {
      const li = document.createElement("li");
      li.textContent = `${row.node_id}: ${row.decision}${row.message ? " — " + row.message : ""}`;
      tl.appendChild(li);
    }
  </script>
</body>
</html>
"""


def render_nodes_html(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return NODES_HTML_TEMPLATE.replace("__PAYLOAD__", blob)


def bearing_between(sam: SceneActionMap, a_id: str, b_id: str) -> float:
    return _bearing_deg(sam.nodes[a_id].px, sam.nodes[b_id].px)


def _draw_dashed_line(
    img: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    gap: int = 6,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    length = int(math.hypot(x1 - x0, y1 - y0))
    if length == 0:
        return
    for i in range(0, length, gap * 2):
        t0 = i / length
        t1 = min(1.0, (i + gap) / length)
        a = (int(x0 + t0 * (x1 - x0)), int(y0 + t0 * (y1 - y0)))
        b = (int(x0 + t1 * (x1 - x0)), int(y0 + t1 * (y1 - y0)))
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def render_sam_panel(
    sam: SceneActionMap,
    free: np.ndarray,
    *,
    height: int = 720,
    route: list[str] | None = None,
    route_replan: list[str] | None = None,
    severed: list[SamEdge] | None = None,
    agent_px: tuple[int, int] | None = None,
    agent_yaw_deg: float = 0.0,
    title_lines: list[str] | None = None,
    controller_labels_list: list[ControllerLabel] | None = None,
    active_node_id: str | None = None,
    decision_diff_list: list[DecisionDiff] | None = None,
    show_graph: bool = True,
) -> np.ndarray:
    """Clean vector-style top-down SAM panel at fixed height."""
    fh, fw = free.shape
    scale = height / float(fh)
    width = max(1, int(round(fw * scale)))
    panel = np.full((height, width, 3), COLOR_BLOCKED, dtype=np.uint8)

    free_u8 = (free.astype(np.uint8) * 255)
    free_bgr = cv2.cvtColor(free_u8, cv2.COLOR_GRAY2BGR)
    free_bgr[free] = COLOR_FREE
    free_bgr[~free] = COLOR_BLOCKED
    panel = cv2.resize(free_bgr, (width, height), interpolation=cv2.INTER_NEAREST)

    def _scale_pt(px: tuple[int, int]) -> tuple[int, int]:
        return (int(round(px[0] * scale)), int(round(px[1] * scale)))

    def _draw_polyline(pts: list[tuple[int, int]], color: tuple[int, int, int], thickness: int, dashed: bool) -> None:
        if len(pts) < 2:
            return
        scaled = [_scale_pt(p) for p in pts]
        if dashed:
            for i in range(len(scaled) - 1):
                _draw_dashed_line(panel, scaled[i], scaled[i + 1], color, thickness=thickness)
        else:
            arr = np.array(scaled, dtype=np.int32)
            cv2.polylines(panel, [arr], False, color, thickness, cv2.LINE_AA)

    for edge in sam.edges:
        path = edge.path_px
        if not path:
            src = sam.nodes[edge.src].px
            dst = sam.nodes[edge.dst].px
            path = [src, dst]
        thickness = max(1, int(round(1 + edge.safety * 3)))
        if edge.alive:
            if not show_graph:
                continue
            green = int(80 + edge.safety * 140)
            color = (0, green, 0)
            _draw_polyline(path, color, thickness, dashed=False)
        else:
            _draw_polyline(path, COLOR_SEVERED, thickness, dashed=True)
            if len(path) >= 2:
                mid = path[len(path) // 2]
                mp = _scale_pt(mid)
                cv2.drawMarker(panel, mp, COLOR_SEVERED, cv2.MARKER_TILTED_CROSS, 10, 2)

        if show_graph and edge.alive and len(path) >= 2:
            mid = path[len(path) // 2]
            mp = _scale_pt(mid)
            tag = _behaviour_tag(edge.behaviour)
            tag_color = BEHAVIOUR_COLORS.get(edge.behaviour, (200, 200, 200))
            cv2.putText(panel, tag, (mp[0] - 4, mp[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, tag_color, 1, cv2.LINE_AA)

    if show_graph:
        step = max(1, int(round(sam.grid_res_px * scale)))
        for y in range(step // 2, height, step):
            for x in range(step // 2, width, step):
                cv2.circle(panel, (x, y), 1, COLOR_GRID, -1, cv2.LINE_AA)

        for pt in sam.grid_points:
            x, y, score = pt
            if score <= 0.55:
                continue
            sx, sy = _scale_pt((x, y))
            cv2.circle(panel, (sx, sy), 2, (160, 160, 255), -1, cv2.LINE_AA)

    if route_replan and len(route_replan) >= 2 and route_replan != route:
        replan_pts = _route_polyline(sam, route_replan)
        _draw_polyline(replan_pts, COLOR_ROUTE_REPLAN, 3, dashed=False)

    if route and len(route) >= 2:
        route_pts = _route_polyline(sam, route)
        _draw_polyline(route_pts, (255, 255, 255), 3, dashed=False)

    label_by_node = {lb.node_id: lb for lb in (controller_labels_list or [])}
    route_cp_ids = [
        nid
        for nid in (route or [])
        if nid in sam.nodes and sam.nodes[nid].kind == "changepoint"
    ]

    def _status_color(status: str) -> tuple[int, int, int]:
        if status in ("blocked", "goal-unreachable"):
            return (0, 0, 220)
        if status == "replanned":
            return (0, 140, 255)
        return (0, 160, 0)

    for node in sam.nodes.values():
        pt = _scale_pt(node.px)
        is_active = node.id == active_node_id
        if node.kind == "destination":
            sz = 7
            cv2.rectangle(panel, (pt[0] - sz, pt[1] - sz), (pt[0] + sz, pt[1] + sz), COLOR_DEST, 2)
            label = node.id.replace("dest_", "")
        else:
            color = (0, 200, 255) if node.source == "doorway" else COLOR_CP
            ring = max(6, min(16, int(round(6 + node.block_score * 4))))
            if is_active:
                cv2.circle(panel, pt, ring + 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(panel, pt, ring, color, 2, cv2.LINE_AA)
            label = _node_display_id(node.id)
        cv2.putText(
            panel,
            label,
            (pt[0] + 10, pt[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 if is_active else 0.35,
            (30, 30, 30),
            2 if is_active else 1,
            cv2.LINE_AA,
        )
        lb = label_by_node.get(node.id)
        if lb and lb.behaviour and node.kind == "changepoint" and node.id in route_cp_ids:
            dest = route[route.index(node.id) + 1] if route and node.id in route else "?"
            decision = f"{_node_display_id(node.id)}: {_behaviour_display(lb.behaviour)}"
            tag_color = _status_color(lb.status)
            cv2.putText(
                panel,
                decision,
                (pt[0] + 10, pt[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                tag_color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                f"-> {_node_display_id(dest)}",
                (pt[0] + 10, pt[1] + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                tag_color,
                1,
                cv2.LINE_AA,
            )
        elif lb and lb.behaviour and node.kind == "changepoint":
            beh_tag = _behaviour_tag(lb.behaviour)
            tag_color = BEHAVIOUR_COLORS.get(lb.behaviour, (200, 200, 200))
            cv2.putText(
                panel,
                beh_tag,
                (pt[0] - 6, pt[1] + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                tag_color,
                2,
                cv2.LINE_AA,
            )

    if route_cp_ids and controller_labels_list:
        table_y = 28
        for cp_id in route_cp_ids:
            lb = label_by_node.get(cp_id)
            if lb is None or not lb.behaviour:
                continue
            nxt = route[route.index(cp_id) + 1] if route and cp_id in route else "?"
            line = f"{_node_display_id(cp_id)}: {_behaviour_display(lb.behaviour)} -> {_node_display_id(nxt)}"
            color = _status_color(lb.status)
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            x = max(4, width - tw - 10)
            cv2.rectangle(panel, (x - 4, table_y - th - 2), (x + tw + 4, table_y + 4), (245, 245, 245), -1)
            cv2.putText(panel, line, (x, table_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            table_y += th + 6

    if agent_px is not None:
        ap = _scale_pt(agent_px)
        yaw_rad = math.radians(agent_yaw_deg)
        tip = (int(ap[0] + 14 * math.sin(yaw_rad)), int(ap[1] - 14 * math.cos(yaw_rad)))
        left = (int(ap[0] + 8 * math.sin(yaw_rad + 2.4)), int(ap[1] - 8 * math.cos(yaw_rad + 2.4)))
        right = (int(ap[0] + 8 * math.sin(yaw_rad - 2.4)), int(ap[1] - 8 * math.cos(yaw_rad - 2.4)))
        tri = np.array([tip, left, right], dtype=np.int32)
        cv2.fillConvexPoly(panel, tri, COLOR_AGENT)

    y = 22
    for line in title_lines or []:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(panel, (4, y - th - 4), (8 + tw, y + 4), (30, 30, 30), -1)
        cv2.putText(panel, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
        y += th + 8

    if controller_labels_list:
        active = None
        if active_node_id:
            active = label_by_node.get(active_node_id)
        if active is None:
            active = next(
                (lb for lb in controller_labels_list if lb.behaviour and lb.status != "open"),
                None,
            )
        if active is None and controller_labels_list:
            active = controller_labels_list[0]
        if active and active.behaviour:
            banner = f"{_node_display_id(active.node_id)} -> {_behaviour_display(active.behaviour)}"
            tag_color = BEHAVIOUR_COLORS.get(active.behaviour, (220, 220, 255))
            (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            bx = max(4, (width - tw) // 2)
            by = height - 48
            cv2.rectangle(panel, (bx - 6, by - th - 8), (bx + tw + 6, by + 8), (20, 20, 20), -1)
            cv2.putText(panel, banner, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.9, tag_color, 2, cv2.LINE_AA)

    if decision_diff_list:
        dy = height - 90
        for diff in decision_diff_list[:4]:
            line = f"{_node_display_id(diff.node_id)}: {diff.before} -> {diff.after}"
            color = (0, 0, 220) if diff.status in ("blocked", "goal-unreachable") else (0, 140, 255)
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            x = max(4, width - tw - 8)
            cv2.rectangle(panel, (x - 4, dy - th - 2), (x + tw + 4, dy + 4), (240, 240, 240), -1)
            cv2.putText(panel, line, (x, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            dy -= th + 8

    legend_y = height - 12
    cv2.line(panel, (10, legend_y), (30, legend_y), COLOR_TRAVERSABLE, 2)
    cv2.putText(panel, "traversable", (34, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 60), 1)
    cv2.line(panel, (110, legend_y), (130, legend_y), COLOR_SEVERED, 2)
    cv2.putText(panel, "blocked", (134, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 60), 1)
    cv2.putText(panel, "L/F/R", (210, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 60), 1)

    return panel


def agent_radius_px(proj: dict[str, Any]) -> int:
    mpp_x = (2.0 * float(proj["half_extent_x"])) / float(proj["image_width"])
    mpp_z = (2.0 * float(proj["half_extent_z"])) / float(proj["image_height"])
    return max(1, int(math.ceil(AGENT_RADIUS_M / min(mpp_x, mpp_z))))


def draw_agent_marker(
    panel: np.ndarray,
    agent_px: tuple[int, int],
    agent_yaw_deg: float,
    free_shape: tuple[int, int],
) -> None:
    """Draw agent triangle on an existing panel (mutates in place)."""
    fh, _fw = free_shape
    scale = panel.shape[0] / float(fh)
    ap = (int(round(agent_px[0] * scale)), int(round(agent_px[1] * scale)))
    yaw_rad = math.radians(agent_yaw_deg)
    tip = (int(ap[0] + 14 * math.sin(yaw_rad)), int(ap[1] - 14 * math.cos(yaw_rad)))
    left = (int(ap[0] + 8 * math.sin(yaw_rad + 2.4)), int(ap[1] - 8 * math.cos(yaw_rad + 2.4)))
    right = (int(ap[0] + 8 * math.sin(yaw_rad - 2.4)), int(ap[1] - 8 * math.cos(yaw_rad - 2.4)))
    tri = np.array([tip, left, right], dtype=np.int32)
    cv2.fillConvexPoly(panel, tri, COLOR_AGENT)


def _synthetic_corridor(w: int = 120, h: int = 40) -> np.ndarray:
    free = np.zeros((h, w), dtype=bool)
    free[h // 4 : 3 * h // 4, 10 : w - 10] = True
    return free


def _synthetic_l_corner(w: int = 80, h: int = 80) -> np.ndarray:
    free = np.zeros((h, w), dtype=bool)
    free[30:50, 10:70] = True
    free[50:70, 50:70] = True
    return free


def _synthetic_plus(w: int = 80, h: int = 80) -> np.ndarray:
    free = np.zeros((h, w), dtype=bool)
    cx, cy = w // 2, h // 2
    free[cy, 10 : w - 10] = True
    free[10 : h - 10, cx] = True
    return free


def _synthetic_two_room(w: int = 100, h: int = 60, door_x: int = 50) -> np.ndarray:
    free = np.zeros((h, w), dtype=bool)
    free[10:50, 10:door_x] = True
    free[10:50, door_x:90] = True
    free[28:32, door_x - 1 : door_x + 2] = True
    return free


def _self_check() -> None:
    proj = {
        "center_x": 0.0,
        "center_z": 0.0,
        "half_extent_x": 5.0,
        "half_extent_z": 5.0,
        "image_width": 100,
        "image_height": 100,
    }

    corridor = _synthetic_corridor()
    cps_c, _ = extract_changepoints(corridor, grid_res_px=8.0, cp_threshold=0.55, proj=proj)
    assert len(cps_c) == 0, f"corridor should have 0 changepoints, got {len(cps_c)}"

    l_mask = _synthetic_l_corner()
    cps_l, _ = extract_changepoints(l_mask, grid_res_px=8.0, cp_threshold=0.55, proj=proj)
    assert len(cps_l) >= 1, "L-corner should yield >=1 changepoint"
    corner_x, corner_y = 60, 50
    assert any(math.hypot(n.px[0] - corner_x, n.px[1] - corner_y) < 20 for n in cps_l)

    plus = _synthetic_plus()
    cps_p, _ = extract_changepoints(plus, grid_res_px=8.0, cp_threshold=0.55, proj=proj)
    assert len(cps_p) >= 1, "plus-junction should yield >=1 changepoint"
    cx, cy = plus.shape[1] // 2, plus.shape[0] // 2
    assert any(math.hypot(n.px[0] - cx, n.px[1] - cy) < 16 for n in cps_p)

    attach_world_coords(cps_p, proj)
    edges = predict_edges(plus, cps_p)
    for edge in edges:
        assert len(edge.path_px) >= 2, "polyline edges need >=2 points"
    for node in cps_p:
        beh = [e.behaviour for e in edges if e.src == node.id]
        assert len(beh) == len(set(beh)), f"duplicate behaviour at {node.id}: {beh}"

    sam = build_scene_action_map(plus, proj, grid_res_px=8.0, cp_threshold=0.55, use_geometric_cps=True)
    assert sam.nodes, "SAM should have nodes"
    panel = render_sam_panel(sam, plus, height=200, title_lines=["self-check"])
    assert panel.shape[0] == 200 and panel.shape[2] == 3
    panel_no_graph = render_sam_panel(sam, plus, height=200, title_lines=["self-check"], show_graph=False)
    assert not np.array_equal(panel, panel_no_graph), "show_graph=False should omit traversable edges"

    two_room = _synthetic_two_room()
    sam2 = build_scene_action_map(
        two_room,
        proj,
        grid_res_px=8.0,
        cp_threshold=0.50,
        destinations=[("start", (-2.0, 2.0)), ("goal", (2.0, 2.0))],
        agent_radius_px=1,
        use_geometric_cps=True,
    )
    route = plan_route(sam2, "dest_start", "dest_goal")
    assert route and route[0] == "dest_start" and route[-1] == "dest_goal", f"bad route {route}"
    cps_on_route = [n for n in route if sam2.nodes[n].kind == "changepoint"]
    assert len(cps_on_route) >= 1, "need changepoint between start and goal"

    designed = pick_designed_route(sam2, "dest_start", "dest_goal", min_cps=1)
    shortest = plan_route(sam2, "dest_start", "dest_goal")
    designed_cps = sum(1 for n in designed[1:-1] if sam2.nodes[n].kind == "changepoint")
    shortest_cps = sum(1 for n in shortest[1:-1] if sam2.nodes[n].kind == "changepoint")
    assert designed_cps >= shortest_cps, "designed route should visit at least as many CPs as shortest"

    labels_open = controller_labels(sam2, route)
    for edge in sam2.edges:
        if edge.src in route and route.index(edge.src) < len(route) - 1:
            nxt = route[route.index(edge.src) + 1]
            if edge.dst == nxt:
                edge.alive = False
                edge.block_reason = "test_block"
                break
    assert sum(1 for e in sam2.edges if not e.alive) < len(sam2.edges), "scoped sever should not kill entire graph"
    _, labels_blocked = replan_from(sam2, route[0], "dest_goal")
    diffs = decision_diff(labels_open, labels_blocked, sam2)
    assert diffs, "decision_diff should report label changes after block"

    for edge in sam2.edges:
        edge.alive = False
        edge.block_reason = "test_block"
    fail_route, fail_labels = replan_from(sam2, "dest_start", "dest_goal")
    assert fail_route == [], "blocked graph should fail replan"
    assert any(lb.status == "goal-unreachable" for lb in fail_labels)

    print("scene_action_map self-check passed")


if __name__ == "__main__":
    _self_check()
