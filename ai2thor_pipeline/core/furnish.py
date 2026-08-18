"""Place topple-prone furniture and falling clutter in ProcTHOR houses."""

from __future__ import annotations

import json
import math
import random
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from core.procthor_house import (
    _point_in_polygon,
    _polygon_from_room,
    door_world_frame,
    doorway_corridor_rect,
    house_floors,
    load_house_json,
    room_centroid,
)
from core.volumetric_map import DOOR_CORRIDOR_DEPTH_M

TOPPLE_ASSETS = (
    "Shelving_Unit_219_1",
    "Shelving_Unit_325_2",
    "Shelving_Unit_215_1",
    "Floor_Lamp_18",
    "Dresser_301_1",
)

CLUTTER_ASSETS = (
    "Book_1",
    "Bottle_1",
    "Bowl_1",
    "Cup_1",
    "Plate_1",
    "Vase_Decorative_1",
    "Kettle_1",
    "Pan_1",
    "Mug_1",
    "Soap_Bottle_1",
    "Dumbbell_1_1",
    "Boot_1",
)

RECEPTACLE_HOSTS = frozenset(
    {
        "Dresser",
        "CounterTop",
        "DiningTable",
        "CoffeeTable",
        "SideTable",
        "TVStand",
        "Desk",
        "ShelvingUnit",
        "Bed",
    }
)

DOOR_WALL_GAP_M = 0.10
DOOR_NORMAL_GAP_M = 0.05
FOOTPRINT_PAD_M = 0.05
CLUTTER_PAD_M = 0.02
MIN_ROOM_INSET_M = 0.15
INTERIOR_GRID_STEP_M = 0.45
RECEPTACLE_TOP_EPS_M = 0.02
RECEPTACLE_JITTER_FRAC = 0.60


def _asset_database_path() -> Path:
    spec = find_spec("procthor")
    if spec is None or spec.origin is None:
        raise RuntimeError("procthor package not installed")
    return Path(spec.origin).resolve().parent / "databases" / "asset-database.json"


def load_asset_dims(asset_id: str, db: dict[str, Any] | None = None) -> tuple[float, float, float]:
    if db is None:
        db = json.loads(_asset_database_path().read_text(encoding="utf-8"))
    for assets in db.values():
        for entry in assets:
            if str(entry.get("assetId")) == asset_id:
                bb = entry.get("boundingBox") or {}
                return (
                    float(bb.get("x", 0.3)),
                    float(bb.get("y", 0.3)),
                    float(bb.get("z", 0.3)),
                )
    raise KeyError(f"assetId not in procthor database: {asset_id}")


def _build_asset_index(db: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    for assets in db.values():
        for entry in assets:
            aid = entry.get("assetId")
            if not aid:
                continue
            bb = entry.get("boundingBox") or {}
            out[str(aid)] = (
                float(bb.get("x", 0.3)),
                float(bb.get("y", 0.3)),
                float(bb.get("z", 0.3)),
            )
    return out


def _build_asset_category(db: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for cat, assets in db.items():
        for entry in assets:
            aid = entry.get("assetId")
            if aid:
                out[str(aid)] = str(cat)
    return out


def _yaw_footprint(sx: float, sz: float, yaw_deg: float) -> tuple[float, float]:
    rad = math.radians(yaw_deg % 360.0)
    c, s = abs(math.cos(rad)), abs(math.sin(rad))
    return sx * c + sz * s, sx * s + sz * c


def _footprint_bounds(
    cx: float,
    cz: float,
    sx: float,
    sz: float,
    yaw_deg: float,
) -> tuple[float, float, float, float]:
    hx, hz = _yaw_footprint(sx, sz, yaw_deg)
    hx *= 0.5
    hz *= 0.5
    return cx - hx, cx + hx, cz - hz, cz + hz


def _aabb_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    *,
    pad: float = FOOTPRINT_PAD_M,
) -> bool:
    ax0, ax1, az0, az1 = a
    bx0, bx1, bz0, bz1 = b
    return not (
        ax1 + pad < bx0
        or bx1 + pad < ax0
        or az1 + pad < bz0
        or bz1 + pad < az0
    )


def _point_in_aabb(x: float, z: float, bounds: tuple[float, float, float, float]) -> bool:
    x0, x1, z0, z1 = bounds
    return x0 <= x <= x1 and z0 <= z <= z1


def _footprint_inside_polygon(
    cx: float,
    cz: float,
    sx: float,
    sz: float,
    yaw_deg: float,
    polygon: list[tuple[float, float]],
) -> bool:
    x0, x1, z0, z1 = _footprint_bounds(cx, cz, sx, sz, yaw_deg)
    for x, z in ((x0, z0), (x0, z1), (x1, z0), (x1, z1), (cx, cz)):
        if not _point_in_polygon(x, z, polygon):
            return False
    return True


def _intersects_corridor(
    cx: float,
    cz: float,
    sx: float,
    sz: float,
    yaw_deg: float,
    corridor: list[tuple[float, float]],
) -> bool:
    bounds = _footprint_bounds(cx, cz, sx, sz, yaw_deg)
    for x, z in corridor:
        if _point_in_aabb(x, z, bounds):
            return True
    cx_d = sum(p[0] for p in corridor) / 4.0
    cz_d = sum(p[1] for p in corridor) / 4.0
    if _point_in_aabb(cx_d, cz_d, bounds):
        return True
    x0, x1, z0, z1 = bounds
    for x, z in ((x0, z0), (x0, z1), (x1, z0), (x1, z1)):
        if _point_in_polygon(x, z, corridor):
            return True
    return False


def _intersects_any_corridor(
    cx: float,
    cz: float,
    sx: float,
    sz: float,
    yaw_deg: float,
    corridors: list[list[tuple[float, float]]],
) -> bool:
    return any(_intersects_corridor(cx, cz, sx, sz, yaw_deg, c) for c in corridors)


def _walk_house_objects(objects: list[dict[str, Any]]):
    for obj in objects:
        yield obj
        yield from _walk_house_objects(obj.get("children") or [])


def _existing_footprints(
    house: dict[str, Any],
    asset_index: dict[str, tuple[float, float, float]],
) -> list[tuple[float, float, float, float]]:
    footprints: list[tuple[float, float, float, float]] = []
    for obj in _walk_house_objects(house.get("objects") or []):
        aid = str(obj.get("assetId") or "")
        dims = asset_index.get(aid)
        if dims is None:
            continue
        pos = obj.get("position") or {}
        rot = obj.get("rotation") or {}
        cx = float(pos.get("x", 0.0))
        cz = float(pos.get("z", 0.0))
        yaw = float(rot.get("y", 0.0))
        footprints.append(_footprint_bounds(cx, cz, dims[0], dims[2], yaw))
    return footprints


def _interior_doors(house: dict[str, Any]) -> list[dict[str, Any]]:
    doors: list[dict[str, Any]] = []
    for door in house.get("doors") or []:
        r0 = str(door.get("room0") or "")
        r1 = str(door.get("room1") or "")
        if r0 and r1 and r0 != r1:
            doors.append(door)
    return doors


def _all_door_corridors(house: dict[str, Any]) -> list[list[tuple[float, float]]]:
    corridors: list[list[tuple[float, float]]] = []
    for door in _interior_doors(house):
        try:
            frame = door_world_frame(door, house)
        except ValueError:
            continue
        corridors.append(doorway_corridor_rect(frame, depth_m=DOOR_CORRIDOR_DEPTH_M))
    return corridors


def _polygon_edges(polygon: list[tuple[float, float]]):
    for i in range(len(polygon)):
        yield polygon[i], polygon[(i + 1) % len(polygon)]


def _edge_inward_normal(
    p0: tuple[float, float],
    p1: tuple[float, float],
    centroid: tuple[float, float],
) -> tuple[float, float, float, float]:
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dz) or 1.0
    ax, az = dx / length, dz / length
    nx, nz = -az, ax
    mid_x = (p0[0] + p1[0]) / 2.0
    mid_z = (p0[1] + p1[1]) / 2.0
    to_cx = centroid[0] - mid_x
    to_cz = centroid[1] - mid_z
    if nx * to_cx + nz * to_cz < 0:
        nx, nz = -nx, -nz
    return nx, nz, ax, az


def _pick_yaw(sx: float, sz: float) -> float:
    best_yaw = 0.0
    best_score = float("inf")
    for yaw in (0.0, 90.0, 180.0, 270.0):
        fx, fz = _yaw_footprint(sx, sz, yaw)
        score = abs(fx - max(sx, sz)) + abs(fz - min(sx, sz))
        if score < best_score:
            best_score = score
            best_yaw = yaw
    return best_yaw


def _placement_candidate(
    frame: dict[str, Any],
    room: dict[str, Any],
    *,
    lateral_sign: float,
    asset_id: str,
    dims: tuple[float, float, float],
    base_y: float,
) -> dict[str, Any] | None:
    sx, sy, sz = dims
    cx_d = float(frame["center"]["x"])
    cz_d = float(frame["center"]["z"])
    ax = float(frame["along"]["x"])
    az = float(frame["along"]["z"])
    nx = float(frame["normal"]["x"])
    nz = float(frame["normal"]["z"])
    door_w = float(frame["width"])

    centroid = room_centroid(room)
    to_room_x = float(centroid["x"]) - cx_d
    to_room_z = float(centroid["z"]) - cz_d
    if to_room_x * nx + to_room_z * nz < 0:
        nx, nz = -nx, -nz

    yaw = _pick_yaw(sx, sz)
    foot_along, foot_normal = _yaw_footprint(sx, sz, yaw)

    lateral = lateral_sign * (door_w / 2.0 + DOOR_WALL_GAP_M + foot_along / 2.0)
    normal = foot_normal / 2.0 + DOOR_NORMAL_GAP_M
    px = cx_d + ax * lateral + nx * normal
    pz = cz_d + az * lateral + nz * normal
    py = base_y + sy / 2.0

    return {
        "assetId": asset_id,
        "position": {"x": px, "y": py, "z": pz},
        "rotation": {"x": 0, "y": yaw, "z": 0},
        "kinematic": False,
        "layer": room.get("layer") or "Procedural0",
        "_footprint": _footprint_bounds(px, pz, sx, sz, yaw),
    }


def _report_objects(placed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": o["id"],
            "assetId": o["assetId"],
            "x": round(o["position"]["x"], 3),
            "z": round(o["position"]["z"], 3),
            "yaw": o["rotation"]["y"],
        }
        for o in placed
    ]


def place_topple_blockers(
    house: dict[str, Any],
    *,
    per_door: int = 2,
    seed: int = 0,
    doors: list[dict[str, Any]] | None = None,
    assets: tuple[str, ...] = TOPPLE_ASSETS,
    occupied: list[tuple[float, float, float, float]] | None = None,
    asset_index: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return new object entries flanking interior doorways on the room side."""
    db = json.loads(_asset_database_path().read_text(encoding="utf-8"))
    if asset_index is None:
        asset_index = _build_asset_index(db)
    floors = house_floors(house)
    base_y = float(floors[0].get("baseY", 0.0))
    rooms = {str(r.get("id")): r for r in (house.get("rooms") or []) if r.get("id")}

    rng = random.Random(seed)
    door_list = doors if doors is not None else _interior_doors(house)
    if occupied is None:
        occupied = _existing_footprints(house, asset_index)

    placed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    asset_cycle = list(assets)
    rng.shuffle(asset_cycle)

    for door in door_list:
        door_id = str(door.get("id") or "")
        try:
            frame = door_world_frame(door, house)
        except ValueError as exc:
            skipped.append({"door": door_id, "reason": str(exc)})
            continue
        corridor = doorway_corridor_rect(frame, depth_m=DOOR_CORRIDOR_DEPTH_M)

        for room_id in (str(door.get("room0") or ""), str(door.get("room1") or "")):
            room = rooms.get(room_id)
            if room is None:
                continue
            polygon = _polygon_from_room(room)
            if len(polygon) < 3:
                continue

            for side_idx, lateral_sign in enumerate((-1.0, 1.0)):
                if side_idx >= per_door:
                    break
                asset_id = asset_cycle[(len(placed) + side_idx) % len(asset_cycle)]
                dims = asset_index.get(asset_id)
                if dims is None:
                    skipped.append({"door": door_id, "room": room_id, "reason": f"unknown asset {asset_id}"})
                    continue

                cand = _placement_candidate(
                    frame,
                    room,
                    lateral_sign=lateral_sign,
                    asset_id=asset_id,
                    dims=dims,
                    base_y=base_y,
                )
                if cand is None:
                    continue

                px = float(cand["position"]["x"])
                pz = float(cand["position"]["z"])
                yaw = float(cand["rotation"]["y"])
                fp = cand["_footprint"]

                if not _footprint_inside_polygon(px, pz, dims[0], dims[2], yaw, polygon):
                    skipped.append({"door": door_id, "room": room_id, "reason": "outside room polygon"})
                    continue
                if _intersects_corridor(px, pz, dims[0], dims[2], yaw, corridor):
                    skipped.append({"door": door_id, "room": room_id, "reason": "intersects doorway corridor"})
                    continue
                if any(_aabb_overlap(fp, occ) for occ in occupied):
                    skipped.append({"door": door_id, "room": room_id, "reason": "overlaps existing object"})
                    continue

                obj_id = f"topple|{door_id.replace('|', '_')}|{room_id.replace('|', '_')}|{side_idx}"
                entry = {
                    "id": obj_id,
                    "assetId": asset_id,
                    "position": cand["position"],
                    "rotation": cand["rotation"],
                    "kinematic": False,
                    "layer": cand["layer"],
                }
                placed.append(entry)
                occupied.append(fp)

    report = {
        "placed": len(placed),
        "skipped": len(skipped),
        "skipped_detail": skipped[:40],
        "objects": _report_objects(placed),
    }
    return placed, report


def place_interior_topple(
    house: dict[str, Any],
    *,
    per_room: int = 3,
    seed: int = 0,
    occupied: list[tuple[float, float, float, float]] | None = None,
    asset_index: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Place tall Moveable furniture against interior walls."""
    db = json.loads(_asset_database_path().read_text(encoding="utf-8"))
    if asset_index is None:
        asset_index = _build_asset_index(db)
    floors = house_floors(house)
    base_y = float(floors[0].get("baseY", 0.0))
    if occupied is None:
        occupied = _existing_footprints(house, asset_index)
    corridors = _all_door_corridors(house)

    rng = random.Random(seed + 17)
    assets = list(TOPPLE_ASSETS)
    rng.shuffle(assets)

    placed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for room in house.get("rooms") or []:
        room_id = str(room.get("id") or "")
        polygon = _polygon_from_room(room)
        if len(polygon) < 3:
            continue
        centroid_pt = room_centroid(room)
        centroid = (float(centroid_pt["x"]), float(centroid_pt["z"]))

        edges = sorted(
            list(_polygon_edges(polygon)),
            key=lambda e: math.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]),
            reverse=True,
        )
        attempts = 0
        room_placed = 0
        for p0, p1 in edges:
            if room_placed >= per_room:
                break
            edge_len = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if edge_len < 0.8:
                continue
            nx, nz, ax, az = _edge_inward_normal(p0, p1, centroid)
            for t in (0.35, 0.65, 0.5):
                if room_placed >= per_room:
                    break
                attempts += 1
                asset_id = assets[(len(placed) + attempts) % len(assets)]
                dims = asset_index[asset_id]
                sx, sy, sz = dims
                yaw = _pick_yaw(sx, sz)
                foot_along, foot_normal = _yaw_footprint(sx, sz, yaw)

                px = p0[0] + (p1[0] - p0[0]) * t + nx * (foot_normal / 2.0 + DOOR_NORMAL_GAP_M)
                pz = p0[1] + (p1[1] - p0[1]) * t + nz * (foot_normal / 2.0 + DOOR_NORMAL_GAP_M)
                py = base_y + sy / 2.0
                fp = _footprint_bounds(px, pz, sx, sz, yaw)

                if not _footprint_inside_polygon(px, pz, sx, sz, yaw, polygon):
                    skipped.append({"room": room_id, "reason": "outside room polygon"})
                    continue
                if _intersects_any_corridor(px, pz, sx, sz, yaw, corridors):
                    skipped.append({"room": room_id, "reason": "intersects doorway corridor"})
                    continue
                if any(_aabb_overlap(fp, occ) for occ in occupied):
                    skipped.append({"room": room_id, "reason": "overlaps existing object"})
                    continue

                obj_id = f"interior|{room_id.replace('|', '_')}|{room_placed}"
                placed.append(
                    {
                        "id": obj_id,
                        "assetId": asset_id,
                        "position": {"x": px, "y": py, "z": pz},
                        "rotation": {"x": 0, "y": yaw, "z": 0},
                        "kinematic": False,
                        "layer": room.get("layer") or "Procedural0",
                    }
                )
                occupied.append(fp)
                room_placed += 1

    report = {
        "placed": len(placed),
        "skipped": len(skipped),
        "skipped_detail": skipped[:40],
        "objects": _report_objects(placed),
    }
    return placed, report


def _room_grid_points(polygon: list[tuple[float, float]], step: float) -> list[tuple[float, float]]:
    xs = [p[0] for p in polygon]
    zs = [p[1] for p in polygon]
    x0, x1 = min(xs) + MIN_ROOM_INSET_M, max(xs) - MIN_ROOM_INSET_M
    z0, z1 = min(zs) + MIN_ROOM_INSET_M, max(zs) - MIN_ROOM_INSET_M
    pts: list[tuple[float, float]] = []
    x = x0
    while x <= x1:
        z = z0
        while z <= z1:
            if _point_in_polygon(x, z, polygon):
                pts.append((x, z))
            z += step
        x += step
    return pts


def place_floor_clutter(
    house: dict[str, Any],
    *,
    per_room: int = 6,
    seed: int = 0,
    occupied: list[tuple[float, float, float, float]] | None = None,
    asset_index: dict[str, tuple[float, float, float]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scatter small CanPickup clutter on room floors."""
    db = json.loads(_asset_database_path().read_text(encoding="utf-8"))
    if asset_index is None:
        asset_index = _build_asset_index(db)
    floors = house_floors(house)
    base_y = float(floors[0].get("baseY", 0.0))
    if occupied is None:
        occupied = _existing_footprints(house, asset_index)
    corridors = _all_door_corridors(house)

    rng = random.Random(seed + 31)
    assets = list(CLUTTER_ASSETS)
    rng.shuffle(assets)

    placed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for room in house.get("rooms") or []:
        room_id = str(room.get("id") or "")
        polygon = _polygon_from_room(room)
        if len(polygon) < 3:
            continue
        candidates = _room_grid_points(polygon, INTERIOR_GRID_STEP_M)
        rng.shuffle(candidates)
        room_placed = 0
        for px, pz in candidates:
            if room_placed >= per_room:
                break
            asset_id = assets[(len(placed) + room_placed) % len(assets)]
            sx, sy, sz = asset_index[asset_id]
            yaw = rng.choice((0.0, 90.0, 180.0, 270.0))
            py = base_y + sy / 2.0
            fp = _footprint_bounds(px, pz, sx, sz, yaw)

            if not _footprint_inside_polygon(px, pz, sx, sz, yaw, polygon):
                continue
            if _intersects_any_corridor(px, pz, sx, sz, yaw, corridors):
                skipped.append({"room": room_id, "reason": "intersects doorway corridor"})
                continue
            if any(_aabb_overlap(fp, occ, pad=CLUTTER_PAD_M) for occ in occupied):
                continue

            obj_id = f"floor|{room_id.replace('|', '_')}|{room_placed}"
            placed.append(
                {
                    "id": obj_id,
                    "assetId": asset_id,
                    "position": {"x": px, "y": py, "z": pz},
                    "rotation": {"x": 0, "y": yaw, "z": 0},
                    "kinematic": False,
                    "layer": room.get("layer") or "Procedural0",
                }
            )
            occupied.append(fp)
            room_placed += 1

    report = {
        "placed": len(placed),
        "skipped": len(skipped),
        "skipped_detail": skipped[:40],
        "objects": _report_objects(placed),
    }
    return placed, report


def place_receptacle_clutter(
    house: dict[str, Any],
    *,
    per_host: int = 2,
    seed: int = 0,
    asset_index: dict[str, tuple[float, float, float]] | None = None,
    asset_category: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stack small clutter on top of existing furniture receptacles."""
    db = json.loads(_asset_database_path().read_text(encoding="utf-8"))
    if asset_index is None:
        asset_index = _build_asset_index(db)
    if asset_category is None:
        asset_category = _build_asset_category(db)

    rng = random.Random(seed + 47)
    assets = list(CLUTTER_ASSETS)
    rng.shuffle(assets)

    placed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    stack_fps: list[tuple[float, float, float, float]] = []

    for host in _walk_house_objects(house.get("objects") or []):
        host_id = str(host.get("id") or host.get("objectId") or "")
        aid = str(host.get("assetId") or "")
        cat = asset_category.get(aid, "")
        if cat not in RECEPTACLE_HOSTS:
            continue
        host_dims = asset_index.get(aid)
        if host_dims is None:
            continue
        hsx, hsy, hsz = host_dims
        pos = host.get("position") or {}
        rot = host.get("rotation") or {}
        hx = float(pos.get("x", 0.0))
        hy = float(pos.get("y", 0.0))
        hz = float(pos.get("z", 0.0))
        host_yaw = float(rot.get("y", 0.0))
        layer = host.get("layer") or "Procedural0"

        for n in range(per_host):
            clutter_id = assets[(len(placed) + n) % len(assets)]
            csx, csy, csz = asset_index[clutter_id]
            yaw = rng.choice((0.0, 90.0, 180.0, 270.0))
            jx = rng.uniform(-RECEPTACLE_JITTER_FRAC, RECEPTACLE_JITTER_FRAC) * hsx * 0.5
            jz = rng.uniform(-RECEPTACLE_JITTER_FRAC, RECEPTACLE_JITTER_FRAC) * hsz * 0.5
            px = hx + jx
            pz = hz + jz
            py = hy + hsy / 2.0 + csy / 2.0 + RECEPTACLE_TOP_EPS_M
            fp = _footprint_bounds(px, pz, csx, csz, yaw)

            if any(_aabb_overlap(fp, sfp, pad=CLUTTER_PAD_M) for sfp in stack_fps):
                skipped.append({"host": host_id, "reason": "overlaps stacked clutter"})
                continue

            obj_id = f"stack|{host_id.replace('|', '_')}|{n}"
            placed.append(
                {
                    "id": obj_id,
                    "assetId": clutter_id,
                    "position": {"x": px, "y": py, "z": pz},
                    "rotation": {"x": 0, "y": yaw, "z": 0},
                    "kinematic": False,
                    "layer": layer,
                }
            )
            stack_fps.append(fp)

    report = {
        "placed": len(placed),
        "skipped": len(skipped),
        "skipped_detail": skipped[:40],
        "objects": _report_objects(placed),
    }
    return placed, report


def furnish_house(
    house: dict[str, Any],
    *,
    per_door: int = 2,
    interior_per_room: int = 3,
    floor_clutter_per_room: int = 6,
    receptacle_clutter: int = 2,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copy of house with doorway, interior, floor, and stacked clutter."""
    db = json.loads(_asset_database_path().read_text(encoding="utf-8"))
    asset_index = _build_asset_index(db)
    asset_category = _build_asset_category(db)

    out = json.loads(json.dumps(house))
    occupied = _existing_footprints(out, asset_index)
    all_new: list[dict[str, Any]] = []

    doorway, r_door = place_topple_blockers(
        out,
        per_door=per_door,
        seed=seed,
        occupied=occupied,
        asset_index=asset_index,
    )
    all_new.extend(doorway)

    interior, r_int = place_interior_topple(
        out,
        per_room=interior_per_room,
        seed=seed,
        occupied=occupied,
        asset_index=asset_index,
    )
    all_new.extend(interior)

    floor_c, r_floor = place_floor_clutter(
        out,
        per_room=floor_clutter_per_room,
        seed=seed,
        occupied=occupied,
        asset_index=asset_index,
    )
    all_new.extend(floor_c)

    stacked, r_stack = place_receptacle_clutter(
        out,
        per_host=receptacle_clutter,
        seed=seed,
        asset_index=asset_index,
        asset_category=asset_category,
    )
    all_new.extend(stacked)

    out.setdefault("objects", []).extend(all_new)

    combined = {
        "total_placed": len(all_new),
        "doorway": r_door,
        "interior": r_int,
        "floor_clutter": r_floor,
        "receptacle_clutter": r_stack,
    }
    return out, combined


def _self_check() -> None:
    batch = Path(__file__).resolve().parents[1] / "assets" / "houses" / "batch"
    house_path = batch / "four_room_ring_1f.json"
    house = load_house_json(house_path)
    db = json.loads(_asset_database_path().read_text(encoding="utf-8"))
    asset_index = _build_asset_index(db)
    corridors = _all_door_corridors(house)
    rooms = {str(r.get("id")): r for r in (house.get("rooms") or []) if r.get("id")}

    furnished, report = furnish_house(house, seed=42)
    assert report["total_placed"] > 20, f"expected >20 placements, got {report['total_placed']}: {report}"

    occupied = _existing_footprints(house, asset_index)
    new_objects = [
        o
        for o in furnished.get("objects") or []
        if str(o.get("id", "")).startswith(("topple|", "interior|", "floor|", "stack|"))
    ]
    assert len(new_objects) == report["total_placed"]

    for obj in new_objects:
        assert obj.get("kinematic") is False
        aid = str(obj["assetId"])
        dims = asset_index[aid]
        pos = obj["position"]
        yaw = float(obj["rotation"]["y"])
        fp = _footprint_bounds(float(pos["x"]), float(pos["z"]), dims[0], dims[2], yaw)

        if str(obj["id"]).startswith(("topple|", "interior|", "floor|")):
            pad = CLUTTER_PAD_M if str(obj["id"]).startswith("floor|") else FOOTPRINT_PAD_M
            assert not any(_aabb_overlap(fp, occ, pad=pad) for occ in occupied), obj["id"]
            occupied.append(fp)
            in_any = False
            for room in rooms.values():
                poly = _polygon_from_room(room)
                if _footprint_inside_polygon(
                    float(pos["x"]), float(pos["z"]), dims[0], dims[2], yaw, poly
                ):
                    in_any = True
                    break
            assert in_any, obj["id"]
            assert not _intersects_any_corridor(
                float(pos["x"]), float(pos["z"]), dims[0], dims[2], yaw, corridors
            ), obj["id"]

    print(
        f"furnish _self_check passed ({report['total_placed']} total: "
        f"door={report['doorway']['placed']} interior={report['interior']['placed']} "
        f"floor={report['floor_clutter']['placed']} stack={report['receptacle_clutter']['placed']})"
    )


if __name__ == "__main__":
    _self_check()
