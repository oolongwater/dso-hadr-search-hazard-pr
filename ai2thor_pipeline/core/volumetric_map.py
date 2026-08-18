"""Analytic 3D voxel occupancy map from ProcTHOR house geometry + AI2-THOR AABBs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.procthor_house import (
    _polygon_from_room,
    _wall_floor_endpoints,
    door_world_frame,
    doorway_corridor_rect,
    house_floors,
)

# Voxel labels
OUTSIDE = 0
FREE = 1
WALL = 2
OBJECT = 3
DEBRIS = 4  # reserved for post-hazard debris

# Per-column traversability classes (derived from floor-up clearance)
TRAV_BLOCKED = 0
TRAV_CRAWL = 1
TRAV_STOOP = 2
TRAV_WALK = 3

# Hardcoded clearance thresholds (metres) — no env vars
BLOCKED_CLEARANCE_M = 0.40
CRAWL_CLEARANCE_M = 1.00
STOOP_CLEARANCE_M = 1.75
WALK_CLEARANCE_M = 1.75

DEFAULT_RESOLUTION_M = 0.10
DEFAULT_PADDING_M = 0.5
WALL_THICKNESS_M = 0.10
DOOR_CORRIDOR_DEPTH_M = 0.45

# Skip structural / scene-spanning objects when stamping AABBs
STRUCTURAL_OBJECT_TYPES = frozenset(
    {
        "Floor",
        "Ceiling",
        "Wall",
        "Door",
        "Doorway",
        "Doorframe",
        "Window",
        "Room",
    }
)
STRUCTURAL_SPAN_FRAC = 0.80
MIN_VISIBLE_SEG_PX = 60
SEG_OVERLAY_ALPHA = 0.45

# Shell colors for semantic panel (BGR)
_COLOR_SHELL_FREE = (200, 200, 200)
_COLOR_SHELL_WALL = (120, 120, 160)
_COLOR_SHELL_OUTSIDE = (20, 20, 20)


@dataclass
class VolumeGrid:
    labels: np.ndarray  # uint8 (n_y, n_z, n_x)
    clearance: np.ndarray  # float32 (n_z, n_x), metres
    traversability: np.ndarray  # uint8 (n_z, n_x)
    footprint: np.ndarray  # bool (n_z, n_x)
    static_labels: np.ndarray  # uint8 (n_y, n_z, n_x) footprint/walls/doors before objects
    object_ids: np.ndarray  # uint16 (n_y, n_z, n_x), 0=none, else index into objects_table
    objects_table: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _world_to_col(x: float, meta: dict[str, Any]) -> int:
    return int((x - meta["origin_x"]) / meta["resolution"])


def _world_to_row(z: float, meta: dict[str, Any]) -> int:
    return int((z - meta["origin_z"]) / meta["resolution"])


def _world_to_iy(y: float, meta: dict[str, Any]) -> int:
    return int((y - meta["base_y"]) / meta["resolution"])


def world_to_voxel(x: float, y: float, z: float, meta: dict[str, Any]) -> tuple[int, int, int]:
    """Return (iy, row, col) indices, clamped to grid bounds."""
    iy = _world_to_iy(y, meta)
    row = _world_to_row(z, meta)
    col = _world_to_col(x, meta)
    iy = max(0, min(iy, meta["n_y"] - 1))
    row = max(0, min(row, meta["n_z"] - 1))
    col = max(0, min(col, meta["n_x"] - 1))
    return iy, row, col


def _polygon_y_range(wall: dict[str, Any]) -> tuple[float, float]:
    ys = [
        float(p.get("y", 0.0) if isinstance(p, dict) else p[1 if len(p) == 2 else 1])
        for p in (wall.get("polygon") or [])
    ]
    if not ys:
        return 0.0, 2.5
    return min(ys), max(ys)


def _door_y_range(door: dict[str, Any]) -> tuple[float, float]:
    hole = door.get("holePolygon") or door.get("hole_polygon") or []
    ys = [float(p.get("y", 0.0) if isinstance(p, dict) else p[1]) for p in hole]
    if not ys:
        return 0.0, 2.1
    return min(ys), max(ys)


def _house_footprint_bounds(house: dict[str, Any], *, padding: float) -> tuple[float, float, float, float]:
    xs: list[float] = []
    zs: list[float] = []
    for room in house.get("rooms") or []:
        for x, z in _polygon_from_room(room):
            xs.append(x)
            zs.append(z)
    if not xs:
        return -padding, padding, -padding, padding
    return min(xs) - padding, max(xs) + padding, min(zs) - padding, max(zs) + padding


def _ceiling_y(house: dict[str, Any]) -> float:
    ys: list[float] = []
    for wall in house.get("walls") or []:
        y0, y1 = _polygon_y_range(wall)
        ys.extend([y0, y1])
    floors = house_floors(house)
    ceiling = float(floors[0].get("ceilingY", 2.8)) if floors else 2.8
    if ys:
        ceiling = max(ceiling, max(ys))
    return ceiling


def _init_meta(
    house: dict[str, Any],
    *,
    resolution: float,
    padding: float,
    label: str,
) -> dict[str, Any]:
    min_x, max_x, min_z, max_z = _house_footprint_bounds(house, padding=padding)
    floors = house_floors(house)
    base_y = float(floors[0].get("baseY", 0.0))
    ceiling_y = _ceiling_y(house)
    n_x = max(1, int(math.ceil((max_x - min_x) / resolution)))
    n_z = max(1, int(math.ceil((max_z - min_z) / resolution)))
    n_y = max(1, int(math.ceil((ceiling_y - base_y) / resolution)))
    return {
        "label": label,
        "resolution": resolution,
        "origin_x": min_x,
        "origin_z": min_z,
        "base_y": base_y,
        "ceiling_y": ceiling_y,
        "n_x": n_x,
        "n_y": n_y,
        "n_z": n_z,
        "width_m": max_x - min_x,
        "depth_m": max_z - min_z,
    }


def _stamp_polygon_xz(
    labels: np.ndarray,
    polygon_xz: list[tuple[float, float]],
    value: int,
    meta: dict[str, Any],
    *,
    y0: float | None = None,
    y1: float | None = None,
) -> None:
    if len(polygon_xz) < 3:
        return
    res = meta["resolution"]
    pts = np.array(
        [[_world_to_col(x, meta), _world_to_row(z, meta)] for x, z in polygon_xz],
        dtype=np.int32,
    )
    mask = np.zeros((meta["n_z"], meta["n_x"]), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    iy0 = 0 if y0 is None else max(0, _world_to_iy(y0, meta))
    iy1 = meta["n_y"] if y1 is None else min(meta["n_y"], _world_to_iy(y1, meta) + 1)
    rows, cols = np.where(mask > 0)
    for iy in range(iy0, iy1):
        labels[iy, rows, cols] = value


def _stamp_line_xz(
    labels: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    value: int,
    meta: dict[str, Any],
    *,
    y0: float,
    y1: float,
    thickness_m: float = WALL_THICKNESS_M,
) -> None:
    res = meta["resolution"]
    thickness = max(1, int(round(thickness_m / res)))
    mask = np.zeros((meta["n_z"], meta["n_x"]), dtype=np.uint8)
    c0 = (_world_to_col(p0[0], meta), _world_to_row(p0[1], meta))
    c1 = (_world_to_col(p1[0], meta), _world_to_row(p1[1], meta))
    cv2.line(mask, c0, c1, 1, thickness=thickness)
    iy0 = max(0, _world_to_iy(y0, meta))
    iy1 = min(meta["n_y"], _world_to_iy(y1, meta) + 1)
    rows, cols = np.where(mask > 0)
    for iy in range(iy0, iy1):
        labels[iy, rows, cols] = value


def _hash_color_bgr(name: str) -> tuple[int, int, int]:
    h = abs(hash(name)) & 0xFFFFFF
    return (h & 255, (h >> 8) & 255, (h >> 16) & 255)


def _stamp_aabb(
    labels: np.ndarray,
    center: dict[str, float],
    size: dict[str, float],
    value: int,
    meta: dict[str, Any],
    *,
    object_ids: np.ndarray | None = None,
    obj_idx: int = 0,
) -> None:
    cx = float(center.get("x", 0.0))
    cy = float(center.get("y", 0.0))
    cz = float(center.get("z", 0.0))
    sx = float(size.get("x", 0.0))
    sy = float(size.get("y", 0.0))
    sz = float(size.get("z", 0.0))
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    z0, z1 = cz - sz / 2.0, cz + sz / 2.0
    iy0 = max(0, _world_to_iy(y0, meta))
    iy1 = min(meta["n_y"], _world_to_iy(y1, meta) + 1)
    r0 = max(0, _world_to_row(z0, meta))
    r1 = min(meta["n_z"], _world_to_row(z1, meta) + 1)
    c0 = max(0, _world_to_col(x0, meta))
    c1 = min(meta["n_x"], _world_to_col(x1, meta) + 1)
    labels[iy0:iy1, r0:r1, c0:c1] = value
    if object_ids is not None and obj_idx > 0:
        object_ids[iy0:iy1, r0:r1, c0:c1] = obj_idx


def _is_structural_object(obj: dict[str, Any], meta: dict[str, Any]) -> bool:
    obj_type = str(obj.get("objectType") or obj.get("category") or "")
    if obj_type in STRUCTURAL_OBJECT_TYPES:
        return True
    bbox = obj.get("axisAlignedBoundingBox") or {}
    size = bbox.get("size") or {}
    sx = float(size.get("x", 0.0))
    sz = float(size.get("z", 0.0))
    if sx >= meta["width_m"] * STRUCTURAL_SPAN_FRAC:
        return True
    if sz >= meta["depth_m"] * STRUCTURAL_SPAN_FRAC:
        return True
    return False


def _derive_clearance_and_traversability(
    labels: np.ndarray,
    footprint: np.ndarray,
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    free = labels == FREE
    occupied = ~free
    padded = np.concatenate([occupied, np.ones((1, meta["n_z"], meta["n_x"]), dtype=bool)], axis=0)
    first_occ = np.argmax(padded, axis=0)
    clearance = first_occ.astype(np.float32) * meta["resolution"]

    traversability = np.full((meta["n_z"], meta["n_x"]), TRAV_BLOCKED, dtype=np.uint8)
    traversability[clearance >= BLOCKED_CLEARANCE_M] = TRAV_CRAWL
    traversability[clearance >= CRAWL_CLEARANCE_M] = TRAV_STOOP
    traversability[clearance >= STOOP_CLEARANCE_M] = TRAV_WALK
    traversability[~footprint] = TRAV_BLOCKED
    return clearance, traversability


def _stamp_objects(
    labels: np.ndarray,
    object_ids: np.ndarray,
    objects: list[dict[str, Any]],
    meta: dict[str, Any],
    *,
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Stamp non-structural object AABBs; return (objects_table, stamped, skipped)."""
    # ponytail: solid AABBs block crawl-under space; upgrade via depth carving or OBB/mesh
    objects_table: list[dict[str, Any]] = []
    stamped = 0
    skipped = 0
    for obj in objects:
        bbox = obj.get("axisAlignedBoundingBox") or {}
        if not bbox:
            continue
        if _is_structural_object(obj, meta):
            skipped += 1
            continue
        center = bbox.get("center") or obj.get("position") or {}
        size = bbox.get("size") or {}
        if not size:
            continue
        obj_id = str(obj.get("objectId") or obj.get("id") or f"obj_{stamped}")
        obj_type = str(obj.get("objectType") or obj.get("category") or "Object")
        rgb = (color_map or {}).get(obj_id)
        if rgb is None:
            bgr = _hash_color_bgr(obj_id)
        else:
            bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        obj_idx = len(objects_table) + 1
        objects_table.append({"objectId": obj_id, "objectType": obj_type, "color": bgr})
        _stamp_aabb(
            labels,
            center,
            size,
            OBJECT,
            meta,
            object_ids=object_ids,
            obj_idx=obj_idx,
        )
        stamped += 1
    return objects_table, stamped, skipped


def build_volume(
    house: dict[str, Any],
    objects: list[dict[str, Any]] | None = None,
    *,
    resolution: float = DEFAULT_RESOLUTION_M,
    padding: float = DEFAULT_PADDING_M,
    label: str = "scene",
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> VolumeGrid:
    """Build analytic voxel grid: footprint -> walls -> door carve -> object AABBs."""
    meta = _init_meta(house, resolution=resolution, padding=padding, label=label)
    labels = np.full((meta["n_y"], meta["n_z"], meta["n_x"]), OUTSIDE, dtype=np.uint8)

    # Room footprints -> FREE columns
    footprint = np.zeros((meta["n_z"], meta["n_x"]), dtype=bool)
    for room in house.get("rooms") or []:
        polygon = _polygon_from_room(room)
        if not polygon:
            continue
        _stamp_polygon_xz(labels, polygon, FREE, meta)
        pts = np.array(
            [[_world_to_col(x, meta), _world_to_row(z, meta)] for x, z in polygon],
            dtype=np.int32,
        )
        mask = np.zeros((meta["n_z"], meta["n_x"]), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        footprint |= mask.astype(bool)

    # Walls
    for wall in house.get("walls") or []:
        try:
            p0, p1 = _wall_floor_endpoints(wall)
        except ValueError:
            continue
        y0, y1 = _polygon_y_range(wall)
        _stamp_line_xz(labels, p0, p1, WALL, meta, y0=y0, y1=y1)

    # Door openings carved back to FREE (windows stay solid)
    for door in house.get("doors") or []:
        try:
            frame = door_world_frame(door, house)
        except ValueError:
            continue
        rect = doorway_corridor_rect(frame, depth_m=DOOR_CORRIDOR_DEPTH_M)
        y0, y1 = _door_y_range(door)
        _stamp_polygon_xz(labels, rect, FREE, meta, y0=y0, y1=y1)

    static_labels = labels.copy()
    object_ids = np.zeros_like(labels, dtype=np.uint16)
    objects_table, stamped, skipped_structural = _stamp_objects(
        labels,
        object_ids,
        objects or [],
        meta,
        color_map=color_map,
    )

    meta["objects_stamped"] = stamped
    meta["objects_skipped_structural"] = skipped_structural

    clearance, traversability = _derive_clearance_and_traversability(labels, footprint, meta)
    return VolumeGrid(
        labels=labels,
        clearance=clearance,
        traversability=traversability,
        footprint=footprint,
        static_labels=static_labels,
        object_ids=object_ids,
        objects_table=objects_table,
        meta=meta,
    )


def restamp_objects(
    vol: VolumeGrid,
    objects: list[dict[str, Any]],
    *,
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> None:
    """Rebuild object layer from cached static shell and refresh traversability."""
    vol.labels = vol.static_labels.copy()
    vol.object_ids = np.zeros_like(vol.labels, dtype=np.uint16)
    vol.objects_table, stamped, skipped = _stamp_objects(
        vol.labels,
        vol.object_ids,
        objects,
        vol.meta,
        color_map=color_map,
    )
    vol.meta["objects_stamped"] = stamped
    vol.meta["objects_skipped_structural"] = skipped
    vol.clearance, vol.traversability = _derive_clearance_and_traversability(
        vol.labels, vol.footprint, vol.meta
    )


def clearance_at(vol: VolumeGrid, x: float, z: float) -> float:
    _, row, col = world_to_voxel(x, vol.meta["base_y"], z, vol.meta)
    if not vol.footprint[row, col]:
        return 0.0
    return float(vol.clearance[row, col])


def column_class(vol: VolumeGrid, x: float, z: float) -> int:
    _, row, col = world_to_voxel(x, vol.meta["base_y"], z, vol.meta)
    return int(vol.traversability[row, col])


def free_slice(vol: VolumeGrid, y: float) -> np.ndarray:
    """Horizontal FREE mask at world height y; shape (n_z, n_x), row=z, col=x."""
    iy = _world_to_iy(y, vol.meta)
    iy = max(0, min(iy, vol.meta["n_y"] - 1))
    return vol.labels[iy] == FREE


def color_map_from_event(event) -> dict[str, tuple[int, int, int]]:
    """Build objectId -> RGB from AI2-THOR instance segmentation metadata."""
    out: dict[str, tuple[int, int, int]] = {}
    for color, name in (getattr(event, "color_to_object_id", None) or {}).items():
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            out[str(name)] = (int(color[0]), int(color[1]), int(color[2]))
    oid_map = getattr(event, "object_id_to_color", None) or {}
    for name, color in oid_map.items():
        if isinstance(color, (list, tuple)) and len(color) >= 3:
            out[str(name)] = (int(color[0]), int(color[1]), int(color[2]))
    return out


def _object_type_by_id(objects: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(obj.get("objectId") or obj.get("id") or ""): str(
            obj.get("objectType") or obj.get("category") or "Object"
        )
        for obj in objects
        if obj.get("objectId") or obj.get("id")
    }


def visible_object_ids(
    event,
    *,
    min_pixels: int = MIN_VISIBLE_SEG_PX,
    structural_types: frozenset[str] = STRUCTURAL_OBJECT_TYPES,
) -> set[str]:
    """Return objectIds visible in the FPV instance segmentation frame."""
    seg = getattr(event, "instance_segmentation_frame", None)
    if seg is None:
        return set()
    color_map = getattr(event, "color_to_object_id", None) or {}
    if not color_map:
        return set()

    packed = seg.astype(np.uint32)
    keys = (
        packed[:, :, 0].astype(np.uint32)
        | (packed[:, :, 1].astype(np.uint32) << 8)
        | (packed[:, :, 2].astype(np.uint32) << 16)
    )
    unique, counts = np.unique(keys, return_counts=True)
    type_by_id = _object_type_by_id(event.metadata.get("objects") or [])

    visible: set[str] = set()
    for key, count in zip(unique, counts):
        if count < min_pixels:
            continue
        r = int(key & 255)
        g = int((key >> 8) & 255)
        b = int((key >> 16) & 255)
        obj_id = color_map.get((r, g, b))
        if not obj_id:
            continue
        obj_id = str(obj_id)
        obj_type = type_by_id.get(obj_id, "")
        if obj_type in structural_types:
            continue
        if obj_id.startswith("Floor") or obj_id.startswith("Wall"):
            continue
        visible.add(obj_id)
    return visible


def overlay_instance_seg(
    fpv_bgr: np.ndarray,
    event,
    *,
    alpha: float = SEG_OVERLAY_ALPHA,
) -> np.ndarray:
    """Alpha-blend instance segmentation colors over FPV, masking structural objects."""
    seg = getattr(event, "instance_segmentation_frame", None)
    if seg is None:
        return fpv_bgr
    if seg.shape[:2] != fpv_bgr.shape[:2]:
        seg = cv2.resize(seg, (fpv_bgr.shape[1], fpv_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    type_by_id = _object_type_by_id(event.metadata.get("objects") or [])
    color_map = getattr(event, "color_to_object_id", None) or {}
    overlay = fpv_bgr.copy()
    mask = np.zeros(fpv_bgr.shape[:2], dtype=bool)

    for color, obj_id in color_map.items():
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            continue
        obj_id = str(obj_id)
        obj_type = type_by_id.get(obj_id, "")
        if obj_type in STRUCTURAL_OBJECT_TYPES:
            continue
        if obj_id.startswith("Floor") or obj_id.startswith("Wall"):
            continue
        r, g, b = int(color[0]), int(color[1]), int(color[2])
        sel = (
            (seg[:, :, 0] == r) & (seg[:, :, 1] == g) & (seg[:, :, 2] == b)
        )
        if not sel.any():
            continue
        overlay[sel] = (b, g, r)  # RGB seg -> BGR overlay
        mask |= sel

    if not mask.any():
        return fpv_bgr
    out = fpv_bgr.copy()
    out[mask] = cv2.addWeighted(fpv_bgr, 1.0 - alpha, overlay, alpha, 0)[mask]
    return out


def label_counts(vol: VolumeGrid) -> dict[str, int]:
    names = {OUTSIDE: "outside", FREE: "free", WALL: "wall", OBJECT: "object", DEBRIS: "debris"}
    counts: dict[str, int] = {}
    for val, name in names.items():
        counts[name] = int((vol.labels == val).sum())
    return counts


def traversability_counts(vol: VolumeGrid) -> dict[str, int]:
    names = {
        TRAV_BLOCKED: "blocked",
        TRAV_CRAWL: "crawl",
        TRAV_STOOP: "stoop",
        TRAV_WALK: "walk",
    }
    counts: dict[str, int] = {}
    fp = vol.footprint
    for val, name in names.items():
        counts[name] = int((vol.traversability[fp] == val).sum())
    return counts


def crosscheck_reachable(
    vol: VolumeGrid,
    reachable: list[dict[str, float]],
) -> dict[str, Any]:
    """Compare GetReachablePositions against WALK-class columns."""
    total = len(reachable)
    if total == 0:
        return {"total": 0, "walk_match": 0, "non_walk_rate": 0.0, "mismatches": []}

    walk_match = 0
    mismatches: list[dict[str, Any]] = []
    for pt in reachable:
        x = float(pt["x"])
        z = float(pt["z"])
        cls = column_class(vol, x, z)
        clr = clearance_at(vol, x, z)
        if cls == TRAV_WALK:
            walk_match += 1
        elif len(mismatches) < 20:
            mismatches.append({"x": x, "z": z, "class": cls, "clearance_m": round(clr, 3)})

    non_walk = total - walk_match
    return {
        "total": total,
        "walk_match": walk_match,
        "non_walk": non_walk,
        "non_walk_rate": round(non_walk / total, 4),
        "mismatches_sample": mismatches,
    }


def save_volume(vol: VolumeGrid, npz_path: str | Path) -> None:
    path = Path(npz_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_json = json.dumps(vol.meta)
    np.savez_compressed(
        path,
        labels=vol.labels,
        clearance=vol.clearance,
        traversability=vol.traversability,
        footprint=vol.footprint.astype(np.uint8),
        static_labels=vol.static_labels,
        object_ids=vol.object_ids,
        objects_table_json=np.array(json.dumps(vol.objects_table)),
        meta_json=np.array(meta_json),
    )


def load_volume(npz_path: str | Path) -> VolumeGrid:
    data = np.load(Path(npz_path), allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    objects_table = json.loads(str(data["objects_table_json"])) if "objects_table_json" in data else []
    static_labels = data["static_labels"] if "static_labels" in data else data["labels"]
    object_ids = data["object_ids"] if "object_ids" in data else np.zeros_like(data["labels"], dtype=np.uint16)
    return VolumeGrid(
        labels=data["labels"],
        clearance=data["clearance"],
        traversability=data["traversability"],
        footprint=data["footprint"].astype(bool),
        static_labels=static_labels,
        object_ids=object_ids,
        objects_table=objects_table,
        meta=meta,
    )


# --- Rendering (cv2 only, no matplotlib) ---

_CLEARANCE_COLORMAP = np.array(
    [
        [40, 40, 40],
        [0, 0, 180],
        [0, 180, 180],
        [0, 180, 0],
        [0, 255, 0],
        [180, 255, 0],
        [255, 255, 0],
        [255, 180, 0],
        [255, 80, 0],
        [255, 0, 0],
    ],
    dtype=np.uint8,
)

_TRAV_COLORS = {
    TRAV_BLOCKED: (40, 40, 40),
    TRAV_CRAWL: (180, 120, 40),
    TRAV_STOOP: (40, 180, 220),
    TRAV_WALK: (60, 220, 80),
}


def _render_clearance_heatmap(vol: VolumeGrid, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    fp = vol.footprint
    max_clr = max(STOOP_CLEARANCE_M, vol.meta["ceiling_y"] - vol.meta["base_y"])
    norm = np.clip(vol.clearance / max_clr, 0.0, 1.0)
    idx = (norm * (_CLEARANCE_COLORMAP.shape[0] - 1)).astype(np.int32)
    panel = _CLEARANCE_COLORMAP[idx]
    panel[~fp] = (20, 20, 20)
    return cv2.resize(panel, (w, h), interpolation=cv2.INTER_NEAREST)


def _render_traversability_map(vol: VolumeGrid, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    fp = vol.footprint
    panel = np.zeros((vol.meta["n_z"], vol.meta["n_x"], 3), dtype=np.uint8)
    for cls, color in _TRAV_COLORS.items():
        panel[vol.traversability == cls] = color
    panel[~fp] = (20, 20, 20)
    return cv2.resize(panel, (w, h), interpolation=cv2.INTER_NEAREST)


def _render_slice_montage(vol: VolumeGrid, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    base = vol.meta["base_y"]
    ceiling = vol.meta["ceiling_y"]
    heights = [base + 0.05, base + 0.50, base + 1.00, base + 1.60]
    cols = 2
    rows = 2
    tile_h, tile_w = h // rows, w // cols
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    label_colors = {
        OUTSIDE: (20, 20, 20),
        FREE: (220, 220, 220),
        WALL: (80, 80, 200),
        OBJECT: (60, 160, 255),
    }
    for i, y in enumerate(heights):
        r, c = divmod(i, cols)
        sl = free_slice(vol, y)
        tile = np.zeros((vol.meta["n_z"], vol.meta["n_x"], 3), dtype=np.uint8)
        iy = max(0, min(_world_to_iy(y, vol.meta), vol.meta["n_y"] - 1))
        slice_labels = vol.labels[iy]
        for lbl, color in label_colors.items():
            tile[slice_labels == lbl] = color
        tile = cv2.resize(tile, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        y0, x0 = r * tile_h, c * tile_w
        panel[y0 : y0 + tile_h, x0 : x0 + tile_w] = tile
        cv2.putText(
            panel,
            f"y={y:.2f}m free={sl.sum()}",
            (x0 + 6, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return panel


def _render_isometric(vol: VolumeGrid, size: tuple[int, int]) -> np.ndarray:
    """Painter's-algorithm voxel preview — coarse subsample for speed."""
    h, w = size
    panel = np.full((h, w, 3), 30, dtype=np.uint8)
    res = vol.meta["resolution"]
    step = max(1, int(round(0.25 / res)))
    label_colors = {
        WALL: (100, 100, 220),
        OBJECT: (80, 180, 255),
        FREE: (180, 180, 180),
    }

    voxels: list[tuple[float, int, int, int]] = []
    for iy in range(0, vol.meta["n_y"], step):
        for row in range(0, vol.meta["n_z"], step):
            for col in range(0, vol.meta["n_x"], step):
                lbl = int(vol.labels[iy, row, col])
                if lbl not in label_colors:
                    continue
                x = vol.meta["origin_x"] + (col + 0.5) * res
                z = vol.meta["origin_z"] + (row + 0.5) * res
                y = vol.meta["base_y"] + (iy + 0.5) * res
                depth = x + z + y
                voxels.append((depth, lbl, row, col))

    voxels.sort(key=lambda t: t[0])
    scale = min(w, h) / max(vol.meta["width_m"], vol.meta["depth_m"], 1.0) * 0.35
    cx, cy = w // 2, int(h * 0.75)

    for _, lbl, row, col in voxels:
        x = vol.meta["origin_x"] + (col + 0.5) * res
        z = vol.meta["origin_z"] + (row + 0.5) * res
        sx = int(cx + (x - z) * scale)
        sy = int(cy - (x + z) * scale * 0.5)
        color = label_colors[lbl]
        cv2.circle(panel, (sx, sy), max(1, int(step * scale * 0.15)), color, -1, lineType=cv2.LINE_AA)

    cv2.putText(panel, "iso (subsampled)", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return panel


_TRAV_NAMES = {TRAV_BLOCKED: "blocked", TRAV_CRAWL: "crawl", TRAV_STOOP: "stoop", TRAV_WALK: "walk"}
TRAV_NAMES = _TRAV_NAMES
_COLOR_ROUTE = (0, 220, 255)
_COLOR_TRAIL = (255, 200, 80)
_COLOR_AGENT = (0, 80, 255)


def world_to_panel_px(x: float, z: float, vol: VolumeGrid, panel_w: int, panel_h: int) -> tuple[int, int]:
    col = (x - vol.meta["origin_x"]) / vol.meta["resolution"]
    row = (z - vol.meta["origin_z"]) / vol.meta["resolution"]
    px = int(round(col * panel_w / vol.meta["n_x"]))
    py = int(round(row * panel_h / vol.meta["n_z"]))
    return px, py


def _draw_agent_triangle(
    panel: np.ndarray,
    px: int,
    py: int,
    yaw_deg: float,
    *,
    size: int = 14,
) -> None:
    yaw_rad = math.radians(yaw_deg)
    tip = (int(px + size * math.sin(yaw_rad)), int(py - size * math.cos(yaw_rad)))
    left = (int(px + size * 0.6 * math.sin(yaw_rad + 2.4)), int(py - size * 0.6 * math.cos(yaw_rad + 2.4)))
    right = (int(px + size * 0.6 * math.sin(yaw_rad - 2.4)), int(py - size * 0.6 * math.cos(yaw_rad - 2.4)))
    tri = np.array([tip, left, right], dtype=np.int32)
    cv2.fillConvexPoly(panel, tri, _COLOR_AGENT)
    cv2.polylines(panel, [tri], True, (255, 255, 255), 1, cv2.LINE_AA)


def _render_agent_height_slice(vol: VolumeGrid, y: float, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    label_colors = {
        OUTSIDE: (20, 20, 20),
        FREE: (220, 220, 220),
        WALL: (80, 80, 200),
        OBJECT: (60, 160, 255),
    }
    iy = max(0, min(_world_to_iy(y, vol.meta), vol.meta["n_y"] - 1))
    tile = np.zeros((vol.meta["n_z"], vol.meta["n_x"], 3), dtype=np.uint8)
    slice_labels = vol.labels[iy]
    for lbl, color in label_colors.items():
        tile[slice_labels == lbl] = color
    tile[~vol.footprint] = (20, 20, 20)
    return cv2.resize(tile, (w, h), interpolation=cv2.INTER_NEAREST)


def _semantic_topdown(vol: VolumeGrid, discovered_object_ids: set[str]) -> np.ndarray:
    """Top-down semantic colors: shell known, objects only if discovered."""
    n_z, n_x = vol.meta["n_z"], vol.meta["n_x"]
    panel = np.zeros((n_z, n_x, 3), dtype=np.uint8)
    floor_static = vol.static_labels[0]
    for lbl, color in (
        (FREE, _COLOR_SHELL_FREE),
        (WALL, _COLOR_SHELL_WALL),
        (OUTSIDE, _COLOR_SHELL_OUTSIDE),
    ):
        panel[floor_static == lbl] = color
    panel[~vol.footprint] = _COLOR_SHELL_OUTSIDE

    id_to_idx = {entry["objectId"]: i + 1 for i, entry in enumerate(vol.objects_table)}
    for row in range(n_z):
        for col in range(n_x):
            for iy in range(vol.meta["n_y"] - 1, -1, -1):
                oid_idx = int(vol.object_ids[iy, row, col])
                if oid_idx <= 0:
                    continue
                entry = vol.objects_table[oid_idx - 1]
                if entry["objectId"] not in discovered_object_ids:
                    break
                panel[row, col] = tuple(entry["color"])
                break
    return panel


def render_semantic_panel(
    vol: VolumeGrid,
    discovered_object_ids: set[str],
    *,
    agent_x: float,
    agent_z: float,
    agent_yaw: float,
    route_world: list[tuple[float, float]] | None = None,
    trail_world: list[tuple[float, float]] | None = None,
    height: int = 360,
) -> np.ndarray:
    """Progressive semantic top-down map — only discovered objects colored."""
    aspect = vol.meta["n_x"] / max(vol.meta["n_z"], 1)
    panel_w = max(1, int(round(height * aspect)))
    sem = _semantic_topdown(vol, discovered_object_ids)
    panel = cv2.resize(sem, (panel_w, height), interpolation=cv2.INTER_NEAREST)

    if route_world and len(route_world) >= 2:
        pts = [world_to_panel_px(x, z, vol, panel_w, height) for x, z in route_world]
        for i in range(len(pts) - 1):
            cv2.line(panel, pts[i], pts[i + 1], _COLOR_ROUTE, 2, cv2.LINE_AA)
    if trail_world:
        for x, z in trail_world[-80:]:
            px, py = world_to_panel_px(x, z, vol, panel_w, height)
            cv2.circle(panel, (px, py), 2, _COLOR_TRAIL, -1, cv2.LINE_AA)
    apx, apy = world_to_panel_px(agent_x, agent_z, vol, panel_w, height)
    _draw_agent_triangle(panel, apx, apy, agent_yaw)

    total = len(vol.objects_table)
    discovered = sum(1 for e in vol.objects_table if e["objectId"] in discovered_object_ids)
    cov = (100.0 * discovered / total) if total else 0.0
    cv2.putText(
        panel,
        f"semantic  discovered={discovered}/{total} ({cov:.0f}%)",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_nav_panel(
    vol: VolumeGrid,
    *,
    agent_x: float,
    agent_z: float,
    agent_yaw: float,
    agent_y: float | None = None,
    route_world: list[tuple[float, float]] | None = None,
    trail_world: list[tuple[float, float]] | None = None,
    discovered_object_ids: set[str] | None = None,
    title_lines: list[str] | None = None,
    height: int = 720,
) -> np.ndarray:
    """Semantic top-down (discovered) + live traversability for navigation demo frames."""
    half_h = height // 2
    discovered = discovered_object_ids or set()
    semantic = render_semantic_panel(
        vol,
        discovered,
        agent_x=agent_x,
        agent_z=agent_z,
        agent_yaw=agent_yaw,
        route_world=route_world,
        trail_world=trail_world,
        height=half_h,
    )
    trav = _render_traversability_map(vol, (half_h, semantic.shape[1]))
    apx, apy = world_to_panel_px(agent_x, agent_z, vol, trav.shape[1], half_h)
    _draw_agent_triangle(trav, apx, apy, agent_yaw, size=10)

    panel = np.vstack([semantic, trav])
    cls = column_class(vol, agent_x, agent_z)
    clr = clearance_at(vol, agent_x, agent_z)
    lines = title_lines or [
        "Volumetric map",
        f"clearance={clr:.2f}m class={_TRAV_NAMES.get(cls, cls)}",
    ]
    y = half_h + 22
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(panel, (6, y - th - 4), (10 + tw, y + 4), (20, 20, 20), -1)
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        y += th + 8
    cv2.putText(
        panel,
        "traversability (live)",
        (8, half_h + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_diagnostic_panel(vol: VolumeGrid, *, panel_size: tuple[int, int] = (480, 360)) -> np.ndarray:
    """4-panel diagnostic: clearance | traversability / slices | isometric."""
    pw, ph = panel_size
    top_left = _render_clearance_heatmap(vol, (pw, ph))
    top_right = _render_traversability_map(vol, (pw, ph))
    bot_left = _render_slice_montage(vol, (pw, ph))
    bot_right = _render_isometric(vol, (pw, ph))

    cv2.putText(top_left, "clearance (m)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(top_right, "traversability", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    top = np.hstack([top_left, top_right])
    bot = np.hstack([bot_left, bot_right])
    return np.vstack([top, bot])


def _self_check() -> None:
    """Synthetic two-room house: door WALK, wall BLOCKED, interior FREE, exterior OUTSIDE."""
    house = {
        "rooms": [
            {
                "id": "room|a",
                "floorPolygon": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 4.0, "y": 0.0, "z": 0.0},
                    {"x": 4.0, "y": 0.0, "z": 3.0},
                    {"x": 0.0, "y": 0.0, "z": 3.0},
                ],
            },
            {
                "id": "room|b",
                "floorPolygon": [
                    {"x": 4.0, "y": 0.0, "z": 0.0},
                    {"x": 8.0, "y": 0.0, "z": 0.0},
                    {"x": 8.0, "y": 0.0, "z": 3.0},
                    {"x": 4.0, "y": 0.0, "z": 3.0},
                ],
            },
        ],
        "walls": [
            {
                "id": "wall|a|4|0|4|3",
                "roomId": "room|a",
                "polygon": [
                    {"x": 4.0, "y": 0.0, "z": 0.0},
                    {"x": 4.0, "y": 0.0, "z": 3.0},
                    {"x": 4.0, "y": 2.5, "z": 0.0},
                    {"x": 4.0, "y": 2.5, "z": 3.0},
                ],
            },
        ],
        "doors": [
            {
                "id": "door|a|b",
                "room0": "room|a",
                "room1": "room|b",
                "wall0": "wall|a|4|0|4|3",
                "wall1": "wall|a|4|0|4|3",
                "assetPosition": {"x": 1.5, "y": 1.0, "z": 0.0},
                "holePolygon": [
                    {"x": 1.0, "y": 0.0, "z": 0.0},
                    {"x": 2.0, "y": 2.0, "z": 0.0},
                ],
            },
        ],
        "metadata": {"schema": "1.0.0"},
    }

    vol = build_volume(house, objects=[], resolution=0.10, padding=0.5, label="selfcheck")

    # Interior room A
    assert column_class(vol, 2.0, 1.5) == TRAV_WALK, "room interior should be WALK"
    iy, row, col = world_to_voxel(2.0, 0.0, 1.5, vol.meta)
    assert vol.labels[iy, row, col] == FREE

    # Door column (~x=4, z=1.5) — carved FREE, should be WALK
    door_cls = column_class(vol, 4.0, 1.5)
    assert door_cls == TRAV_WALK, f"door should be WALK, got {door_cls}"

    # Wall column away from door (x=4, z=0.2 near wall line but not door)
    wall_cls = column_class(vol, 4.0, 0.2)
    assert wall_cls == TRAV_BLOCKED, f"wall should be BLOCKED, got {wall_cls}"

    # Exterior
    iy_e, row_e, col_e = world_to_voxel(-1.0, 0.0, -1.0, vol.meta)
    assert vol.labels[iy_e, row_e, col_e] == OUTSIDE

    # object_ids consistent with OBJECT labels
    assert np.all((vol.labels == OBJECT) == (vol.object_ids > 0))
    assert np.all((vol.labels != OBJECT) == (vol.object_ids == 0))

    # restamp_objects: box in doorway blocks, removal restores WALK
    blocker = {
        "objectId": "blocker|1",
        "objectType": "Box",
        "axisAlignedBoundingBox": {
            "center": {"x": 4.0, "y": 0.5, "z": 1.5},
            "size": {"x": 0.8, "y": 1.0, "z": 0.8},
        },
    }
    restamp_objects(vol, [blocker])
    assert column_class(vol, 4.0, 1.5) == TRAV_BLOCKED, "blocker should block doorway"
    restamp_objects(vol, [])
    assert column_class(vol, 4.0, 1.5) == TRAV_WALK, "doorway should reopen after blocker removed"

    print("volumetric_map _self_check passed")


if __name__ == "__main__":
    _self_check()
