"""ProcTHOR house load/create helpers for multi-room AI2-THOR scenes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from core.thor import CONTROLLER_GRID, ROTATE_STEP_DEG, distance_xz

_DEFAULT_HOUSE = (
    Path(__file__).resolve().parents[1] / "assets" / "houses" / "house_multifloor_seed1234.json"
)
_LEGACY_HOUSE = (
    Path(__file__).resolve().parents[1] / "assets" / "houses" / "house_kitchen_living_seed42.json"
)
_FLOOR_Y_TOLERANCE = 0.35
_FLOOR_STANDING_OFFSET = 0.9  # ponytail: navmesh y sits above slab baseY, not at baseY


def default_house_path() -> Path:
    if _DEFAULT_HOUSE.is_file():
        return _DEFAULT_HOUSE
    return _LEGACY_HOUSE


def load_house_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def house_scene_id(house: dict[str, Any], *, fallback: str = "procthor_multiroom") -> str:
    for key in ("id", "house_id", "scene_id"):
        if house.get(key):
            return str(house[key])
    rooms = house.get("rooms") or []
    if rooms and rooms[0].get("id"):
        return f"procthor_{rooms[0]['id'].replace('|', '_')}"
    return fallback


def default_local_executable() -> Path:
    return (
        Path.home()
        / "ai2thor-src-full/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR"
    )


def make_procedural_controller(
    house: dict[str, Any] | None = None,
    *,
    headless: bool = False,
    width: int = 640,
    height: int = 480,
    render_depth: bool = True,
    render_instance_segmentation: bool = False,
    local_executable_path: str | Path | None = None,
):
    """Create a Controller on the Procedural scene and load ``house``."""
    import sys

    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    kwargs: dict[str, Any] = {
        "scene": "Procedural",
        "gridSize": CONTROLLER_GRID,
        "rotateStepDegrees": ROTATE_STEP_DEG,
        "snapToGrid": False,
        "width": width,
        "height": height,
        "renderDepthImage": render_depth,
        "renderInstanceSegmentation": render_instance_segmentation,
    }
    if local_executable_path:
        kwargs["local_executable_path"] = str(local_executable_path)
    if headless and sys.platform.startswith("linux"):
        kwargs["platform"] = CloudRendering
    controller = Controller(**kwargs)
    if house is None:
        controller.step(action="Initialize", visibilityDistance=6.0)
    if house is not None:
        event = controller.reset(house, visibilityDistance=6.0)
        if not event.metadata.get("lastActionSuccess", False):
            raise RuntimeError("CreateHouse failed for procedural scene")
        agent = (house.get("metadata") or {}).get("agent") or {}
        position = dict(agent.get("position") or {})
        rotation = agent.get("rotation") or {"x": 0, "y": 0, "z": 0}
        if house_schema(house) == "2.0.0":
            ground = next(
                (floor for floor in house_floors(house) if int(floor.get("index", 0)) == 0),
                house_floors(house)[0],
            )
            floor_id = str(ground.get("id"))
            ground_rooms = rooms_on_floor(house, floor_id)
            if ground_rooms:
                preferred = next(
                    (
                        room
                        for room in ground_rooms
                        if str(room.get("roomType")) in {"Kitchen", "LivingRoom"}
                    ),
                    ground_rooms[0],
                )
                position = room_centroid(preferred)
                position["y"] = float(ground.get("baseY", 0.0)) + 0.95
        if position:
            teleport_event = controller.step(
                action="TeleportFull",
                position=position,
                rotation=rotation,
                horizon=float(agent.get("horizon", 0)),
                standing=bool(agent.get("standing", True)),
            )
            if not teleport_event.metadata.get("lastActionSuccess", False):
                raise RuntimeError(
                    "TeleportFull failed after CreateHouse: "
                    f"{teleport_event.metadata.get('errorMessage')}"
                )
    return controller


def _polygon_from_room(room: dict[str, Any]) -> list[tuple[float, float]]:
    raw = room.get("floorPolygon") or room.get("floor_polygon") or []
    polygon: list[tuple[float, float]] = []
    for pt in raw:
        if isinstance(pt, dict):
            polygon.append((float(pt.get("x", 0.0)), float(pt.get("z", pt.get("y", 0.0)))))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            polygon.append((float(pt[0]), float(pt[2] if len(pt) >= 3 else pt[1])))
    return polygon


def room_centroid(room: dict[str, Any]) -> dict[str, float]:
    polygon = _polygon_from_room(room)
    if not polygon:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    xs = [p[0] for p in polygon]
    zs = [p[1] for p in polygon]
    return {"x": sum(xs) / len(xs), "y": 0.0, "z": sum(zs) / len(zs)}


def room_containing(house: dict[str, Any], x: float, z: float) -> dict[str, Any] | None:
    """Return the room whose floor polygon contains (x, z), or None."""
    for room in house.get("rooms") or []:
        polygon = _polygon_from_room(room)
        if polygon and _point_in_polygon(x, z, polygon):
            return room
    return None


def room_adjacency(house: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return {room_id: {neighbour_id: door_id}} for interior doors."""
    adj: dict[str, dict[str, str]] = {}
    for door in house.get("doors") or []:
        r0 = str(door.get("room0") or door.get("roomId0") or "")
        r1 = str(door.get("room1") or door.get("roomId1") or "")
        if not r0 or not r1 or r0 == r1:
            continue
        door_id = str(door.get("id") or f"{r0}|{r1}")
        adj.setdefault(r0, {})[r1] = door_id
        adj.setdefault(r1, {})[r0] = door_id
    return adj


def _wall_floor_endpoints(wall: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return floor-level (x, z) endpoints from a ProcTHOR wall polygon."""
    floor_pts: list[tuple[float, float]] = []
    for pt in wall.get("polygon") or []:
        if isinstance(pt, dict):
            y = float(pt.get("y", 0.0))
            if abs(y) < 0.01:
                floor_pts.append((float(pt.get("x", 0.0)), float(pt.get("z", 0.0))))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            floor_pts.append((float(pt[0]), float(pt[2] if len(pt) >= 3 else pt[1])))
    if len(floor_pts) < 2:
        raise ValueError(f"wall {wall.get('id')} has fewer than 2 floor points")
    return floor_pts[0], floor_pts[1]


def door_world_frame(door: dict[str, Any], house: dict[str, Any]) -> dict[str, Any]:
    """Convert wall-local assetPosition to world door frame (center, along, normal, width)."""
    wall_id = str(door.get("wall0") or "")
    walls = {str(w.get("id")): w for w in (house.get("walls") or []) if w.get("id")}
    wall = walls.get(wall_id)
    if wall is None:
        raise ValueError(f"door {door.get('id')} references missing wall0={wall_id}")

    p0, p1 = _wall_floor_endpoints(wall)
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dz) or 1.0
    along_x, along_z = dx / length, dz / length
    normal_x, normal_z = -along_z, along_x

    asset_pos = door.get("assetPosition") or {}
    along_dist = float(asset_pos.get("x", 0.0))
    center_x = p0[0] + along_x * along_dist
    center_z = p0[1] + along_z * along_dist
    center_y = float(asset_pos.get("y", 0.0))

    hole = door.get("holePolygon") or door.get("hole_polygon") or []
    hole_xs = [
        float(p.get("x", 0.0) if isinstance(p, dict) else p[0])
        for p in hole
    ]
    width = max(hole_xs) - min(hole_xs) if hole_xs else 1.0

    return {
        "center": {"x": center_x, "y": center_y, "z": center_z},
        "along": {"x": along_x, "z": along_z},
        "normal": {"x": normal_x, "z": normal_z},
        "width": width,
        "wall0": wall_id,
    }


def door_world_center(door: dict[str, Any], house: dict[str, Any]) -> dict[str, float]:
    return dict(door_world_frame(door, house)["center"])


def doorway_corridor_rect(
    frame: dict[str, Any],
    *,
    depth_m: float = 0.45,
) -> list[tuple[float, float]]:
    """Four world (x, z) corners spanning the door opening."""
    cx = float(frame["center"]["x"])
    cz = float(frame["center"]["z"])
    ax = float(frame["along"]["x"])
    az = float(frame["along"]["z"])
    nx = float(frame["normal"]["x"])
    nz = float(frame["normal"]["z"])
    half_w = float(frame["width"]) / 2.0
    return [
        (cx - ax * half_w + nx * depth_m, cz - az * half_w + nz * depth_m),
        (cx + ax * half_w + nx * depth_m, cz + az * half_w + nz * depth_m),
        (cx + ax * half_w - nx * depth_m, cz + az * half_w - nz * depth_m),
        (cx - ax * half_w - nx * depth_m, cz - az * half_w - nz * depth_m),
    ]


def _aabb_footprint(obj: dict[str, Any]) -> tuple[float, float]:
    bbox = obj.get("axisAlignedBoundingBox") or {}
    size = bbox.get("size") or {}
    return float(size.get("x", 0.3)), float(size.get("z", 0.3))


def _aabb_height(obj: dict[str, Any]) -> float:
    bbox = obj.get("axisAlignedBoundingBox") or {}
    size = bbox.get("size") or {}
    return float(size.get("y", 0.3))


TOPPLE_TYPE_RANK = {
    "FloorLamp": 0,
    "Chair": 1,
    "ArmChair": 2,
    "Statue": 3,
    "Vase": 4,
}
DOORWAY_STAGE_COUNT = 3
DOORWAY_STAGE_OFFSET_M = 0.55
DOORWAY_STAGE_SPREAD = (-0.35, 0.0, 0.35)
MIN_FOOTPRINT_M = 0.25


def stage_doorway_blockers(
    controller,
    frame: dict[str, Any],
    room: dict[str, Any],
    *,
    count: int = DOORWAY_STAGE_COUNT,
    door_id: str | None = None,
    offset_m: float = DOORWAY_STAGE_OFFSET_M,
    spread_absolute: bool = False,
    skip_preplaced: bool = False,
    mass_tiebreak: bool = False,
) -> list[str]:
    """Return topple-prone movable object ids adjacent to the doorway."""
    from hazard.utils import find_objects

    door_width = float(frame["width"])
    cx = float(frame["center"]["x"])
    cz = float(frame["center"]["z"])
    cy = float(frame["center"].get("y", 0.0))
    nx = float(frame["normal"]["x"])
    nz = float(frame["normal"]["z"])
    ax = float(frame["along"]["x"])
    az = float(frame["along"]["z"])
    centroid = room_centroid(room)
    to_centroid_x = float(centroid["x"]) - cx
    to_centroid_z = float(centroid["z"]) - cz
    if to_centroid_x * nx + to_centroid_z * nz < 0:
        nx, nz = -nx, -nz

    event = controller.last_event
    candidates = find_objects(
        event,
        predicate=lambda o: (o.get("pickupable") or o.get("moveable")) and not o.get("isBroken"),
    )

    def _score(obj: dict[str, Any]) -> tuple[int, float]:
        foot_x, foot_z = _aabb_footprint(obj)
        if max(foot_x, foot_z) >= door_width:
            return (999, 0.0)
        if mass_tiebreak and max(foot_x, foot_z) < MIN_FOOTPRINT_M:
            return (997, 0.0)
        otype = str(obj.get("objectType") or "")
        rank = TOPPLE_TYPE_RANK.get(otype, 50)
        if mass_tiebreak:
            mass = float(obj.get("mass", 1.0)) or 1.0
            return (rank, -mass)
        return (rank, -_aabb_height(obj))

    if not skip_preplaced:
        door_prefix = f"topple|{door_id.replace('|', '_')}|" if door_id else None
        near_radius = offset_m + 0.75
        preplaced: list[dict[str, Any]] = []
        for obj in candidates:
            oid = str(obj.get("objectId") or obj.get("id") or "")
            pos = obj.get("position") or {}
            dist = math.hypot(float(pos.get("x", 0.0)) - cx, float(pos.get("z", 0.0)) - cz)
            if door_prefix and oid.startswith(door_prefix):
                preplaced.append(obj)
            elif dist <= near_radius:
                preplaced.append(obj)
        if preplaced:
            ranked = sorted(preplaced, key=_score)
            ranked = [o for o in ranked if _score(o)[0] < 50][:count]
            if ranked:
                return [str(o["objectId"]) for o in ranked]

    ranked = sorted(candidates, key=_score)
    ranked = [o for o in ranked if _score(o)[0] < 50][:count]
    if not ranked:
        return []

    spreads = DOORWAY_STAGE_SPREAD[: len(ranked)]
    if len(spreads) < len(ranked):
        spreads = list(spreads) + [0.0] * (len(ranked) - len(spreads))

    staged: list[str] = []
    for obj, spread in zip(ranked, spreads, strict=False):
        lateral = spread if spread_absolute else spread * door_width
        place_x = cx + nx * offset_m + ax * lateral
        place_z = cz + nz * offset_m + az * lateral
        floor_y = cy + 0.02 if mass_tiebreak else float((obj.get("position") or {}).get("y", 0.0))
        res = controller.step(
            action="TeleportObject",
            objectId=obj["objectId"],
            position={"x": place_x, "y": floor_y, "z": place_z},
            rotation=obj.get("rotation") or {"x": 0.0, "y": 0.0, "z": 0.0},
            forceAction=True,
        )
        if res.metadata.get("lastActionSuccess", False):
            staged.append(str(obj["objectId"]))
    return staged


def house_schema(house: dict[str, Any]) -> str:
    return str(house.get("schema") or house.get("metadata", {}).get("schema") or "1.0.0")


def house_floors(house: dict[str, Any]) -> list[dict[str, Any]]:
    floors = house.get("floors") or []
    if floors:
        return sorted(floors, key=lambda f: int(f.get("index", 0)))
    return [{"id": "floor|0", "index": 0, "baseY": 0.0, "ceilingY": 2.8}]


def floor_base_y(house: dict[str, Any], floor_id: str) -> float:
    for floor in house_floors(house):
        if str(floor.get("id")) == floor_id:
            return float(floor.get("baseY", 0.0))
    return 0.0


def rooms_on_floor(house: dict[str, Any], floor_id: str) -> list[dict[str, Any]]:
    return [
        room
        for room in (house.get("rooms") or [])
        if str(room.get("floorId") or "floor|0") == floor_id
    ]


_FALLABLE_MIN_Y_M = 0.35
_VIEW_MIN_DIST_M = 1.5
_VIEW_MAX_DIST_M = 4.0
_VIEW_FOV_DEG = 60.0
_VIEW_YAW_STEP_DEG = 30
_VIEW_GRID_STEP_M = 0.5
_CONE_WEIGHT = 0.6
_GROUND_FALLABLE_WEIGHT = 0.4


def _walk_house_objects(objects: list[dict[str, Any]]):
    for obj in objects:
        yield obj
        yield from _walk_house_objects(obj.get("children") or [])


def _ground_floor(house: dict[str, Any]) -> dict[str, Any]:
    floors = house_floors(house)
    return next((f for f in floors if int(f.get("index", 0)) == 0), floors[0])


def _fallable_on_floor(
    house: dict[str, Any],
    floor_id: str,
    base_y: float,
) -> list[tuple[float, float]]:
    """Return (x, z) of elevated non-kinematic objects on ``floor_id``."""
    pts: list[tuple[float, float]] = []
    for obj in _walk_house_objects(house.get("objects") or []):
        if obj.get("kinematic") is not False:
            continue
        if str(obj.get("floorId") or "floor|0") != floor_id:
            continue
        pos = obj.get("position") or {}
        if float(pos.get("y", 0.0)) - base_y <= _FALLABLE_MIN_Y_M:
            continue
        pts.append((float(pos.get("x", 0.0)), float(pos.get("z", 0.0))))
    return pts


def _best_viewpoint_cone(fallable_xz: list[tuple[float, float]], rooms: list[dict[str, Any]]) -> int:
    """Mirror EarthquakeHazard._score_pose: count fallables in FOV from best grid pose."""
    if not fallable_xz or not rooms:
        return 0
    best = 0
    for room in rooms:
        polygon = _polygon_from_room(room)
        if len(polygon) < 3:
            continue
        xs = [p[0] for p in polygon]
        zs = [p[1] for p in polygon]
        x0, z0 = min(xs), min(zs)
        nx = int((max(xs) - x0) / _VIEW_GRID_STEP_M) + 1
        nz = int((max(zs) - z0) / _VIEW_GRID_STEP_M) + 1
        for i in range(nx):
            for k in range(nz):
                px = x0 + _VIEW_GRID_STEP_M * i
                pz = z0 + _VIEW_GRID_STEP_M * k
                if not _point_in_polygon(px, pz, polygon):
                    continue
                for yaw in range(0, 360, _VIEW_YAW_STEP_DEG):
                    count = 0
                    for ox, oz in fallable_xz:
                        dx, dz = ox - px, oz - pz
                        dist = math.hypot(dx, dz)
                        if dist > _VIEW_MAX_DIST_M or dist < _VIEW_MIN_DIST_M:
                            continue
                        bearing = math.degrees(math.atan2(dx, dz))
                        rel = abs((bearing - yaw + 180.0) % 360.0 - 180.0)
                        if rel > _VIEW_FOV_DEG / 2.0:
                            continue
                        count += 1
                    best = max(best, count)
    return best


def ground_floor_fallable_score(house: dict[str, Any]) -> dict[str, int]:
    """Static proxy for runtime visible-fallable and fallen-object counts."""
    ground = _ground_floor(house)
    floor_id = str(ground.get("id", "floor|0"))
    base_y = float(ground.get("baseY", 0.0))
    g_pts = _fallable_on_floor(house, floor_id, base_y)
    all_pts: list[tuple[float, float]] = []
    for floor in house_floors(house):
        fid = str(floor.get("id", "floor|0"))
        all_pts.extend(_fallable_on_floor(house, fid, float(floor.get("baseY", 0.0))))
    g_rooms = rooms_on_floor(house, floor_id)
    return {
        "cone": _best_viewpoint_cone(g_pts, g_rooms),
        "ground_fallable": len(g_pts),
        "all_fallable": len(all_pts),
    }


def rank_house_json_paths(paths: list[Path], *, top: int = 10) -> list[tuple[Path, dict[str, Any]]]:
    """Score house JSON paths; return top-N sorted by composite fallable score."""
    scored: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        house = load_house_json(path)
        metrics = ground_floor_fallable_score(house)
        scored.append((path, metrics))
    if not scored:
        return []
    cmax = max(m["cone"] for _, m in scored) or 1
    gmax = max(m["ground_fallable"] for _, m in scored) or 1
    ranked: list[tuple[Path, dict[str, Any]]] = []
    for path, metrics in scored:
        score = (
            _CONE_WEIGHT * (metrics["cone"] / cmax)
            + _GROUND_FALLABLE_WEIGHT * (metrics["ground_fallable"] / gmax)
        )
        ranked.append((path, {**metrics, "score": round(score, 4)}))
    ranked.sort(key=lambda item: item[1]["score"], reverse=True)
    return ranked[:top]


def vertical_connectors(house: dict[str, Any]) -> list[dict[str, Any]]:
    return list(house.get("verticalConnectors") or [])


def reachable_on_floor(
    reachable: list[dict[str, float]],
    base_y: float,
    *,
    tolerance: float = _FLOOR_Y_TOLERANCE,
) -> list[dict[str, float]]:
    surface_y = float(base_y) + _FLOOR_STANDING_OFFSET
    return [
        pt
        for pt in reachable
        if abs(float(pt.get("y", 0.0)) - surface_y) <= tolerance
    ]


def multifloor_house_spec():
    from procthor.generation import FloorSpec, HouseSpec
    from procthor.generation.room_specs import RoomSpec
    from procthor.utils.types import LeafRoom

    ground = RoomSpec(
        room_spec_id="custom-ground",
        sampling_weight=1,
        spec=[
            LeafRoom(room_id=2, ratio=3, room_type="Kitchen"),
            LeafRoom(room_id=3, ratio=4, room_type="LivingRoom"),
        ],
    )
    upper = RoomSpec(
        room_spec_id="custom-upper",
        sampling_weight=1,
        spec=[
            LeafRoom(room_id=2, ratio=3, room_type="LivingRoom"),
            LeafRoom(room_id=3, ratio=2, room_type="Bedroom"),
            LeafRoom(room_id=4, ratio=1, room_type="Bathroom"),
        ],
    )
    return HouseSpec(
        house_spec_id="custom-two-floor",
        dims=(14, 10),
        floors=[
            FloorSpec(room_spec=ground, stair_host_room_id=3),
            FloorSpec(room_spec=upper, stair_host_room_id=2),
        ],
    )


def generate_multifloor_house(
    out_path: Path | str,
    *,
    seed: int = 1234,
    local_executable_path: str | Path | None = None,
) -> Path:
    """Generate a deterministic two-floor schema-2 house via hadr-nav/procthor."""
    from procthor.generation import HouseGenerator

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exe = Path(local_executable_path) if local_executable_path else default_local_executable()
    controller = make_procedural_controller(
        None,
        headless=False,
        local_executable_path=str(exe),
    )
    generator = HouseGenerator(
        split="train",
        seed=seed,
        house_spec=multifloor_house_spec(),
        controller=controller,
    )
    house, _ = generator.sample()
    out.write_text(json.dumps(house.data, indent=2), encoding="utf-8")
    controller.stop()
    return out


def pick_primary_connector(house: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (door, room_a, room_b) for the demo hallway pair."""
    rooms = {str(r.get("id")): r for r in (house.get("rooms") or []) if r.get("id")}
    preferred_types = {"Kitchen", "LivingRoom", "Bedroom", "Bathroom"}

    valid_doors: list[dict[str, Any]] = []
    for door in house.get("doors") or []:
        r0 = str(door.get("room0") or door.get("roomId0") or "")
        r1 = str(door.get("room1") or door.get("roomId1") or "")
        if not r0 or not r1 or r0 == r1:
            continue
        valid_doors.append(door)
    if not valid_doors:
        raise ValueError("house has no interior doors/connectors")

    def _score(door: dict[str, Any]) -> tuple[int, int, float]:
        r0 = str(door.get("room0") or door.get("roomId0") or "")
        r1 = str(door.get("room1") or door.get("roomId1") or "")
        t0 = str((rooms.get(r0) or {}).get("roomType") or "")
        t1 = str((rooms.get(r1) or {}).get("roomType") or "")
        type_hit = int(t0 in preferred_types and t1 in preferred_types)
        kitchen_living = int({t0, t1} == {"Kitchen", "LivingRoom"})
        width = 0.0
        hole = door.get("holePolygon") or []
        if hole:
            xs = [float(p.get("x", 0.0) if isinstance(p, dict) else p[0]) for p in hole]
            if xs:
                width = max(xs) - min(xs)
        ground_floor = 0
        if house_schema(house) == "2.0.0":
            fa = str((rooms.get(r0) or {}).get("floorId") or "")
            fb_id = str((rooms.get(r1) or {}).get("floorId") or "")
            ground_floor = int(fa == "floor|0" and fb_id == "floor|0")
        return (ground_floor, kitchen_living, type_hit, width)

    door = max(valid_doors, key=_score)
    r0_id = str(door.get("room0") or door.get("roomId0") or "")
    r1_id = str(door.get("room1") or door.get("roomId1") or "")
    room_a = rooms.get(r0_id) or next(iter(rooms.values()))
    room_b = rooms.get(r1_id) or next(reversed(rooms.values()))
    return door, room_a, room_b


def map_projection_from_props(map_props: dict[str, Any], image_width: int, image_height: int) -> dict[str, Any]:
    pos = map_props.get("position") or {}
    cx = float(pos.get("x", 0.0))
    cz = float(pos.get("z", 0.0))
    half_z = float(map_props.get("orthographicSize", 3.0))
    aspect = image_width / float(max(1, image_height))
    half_x = half_z * aspect
    return {
        "center_x": cx,
        "center_z": cz,
        "half_extent_x": half_x,
        "half_extent_z": half_z,
        "orthographic_size": half_z,
        "image_width": image_width,
        "image_height": image_height,
    }


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


def path_corners_to_pixels(path: list[dict[str, float]], proj: dict[str, Any]) -> list[list[int]]:
    poly: list[list[int]] = []
    for pt in path:
        u, v = world_to_map_px(float(pt["x"]), float(pt["z"]), proj)
        poly.append([int(round(u)), int(round(v))])
    return poly


def generate_and_cache_house(out_path: Path | str, *, seed: int = 42) -> Path:
    """Write a reproducible small multi-room house JSON (procthor-10k train index)."""
    import prior

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset = prior.load_dataset("procthor-10k")
    train = dataset["train"]
    # ponytail: linear scan for first 2-room house with one interior door; upgrade to indexed lookup if catalog grows
    for offset in range(500):
        idx = (seed + offset) % len(train)
        data = train[idx]
        rooms = data.get("rooms") or []
        interior_doors = [
            d for d in (data.get("doors") or [])
            if str(d.get("room0")) != str(d.get("room1"))
        ]
        if len(rooms) == 2 and interior_doors:
            out.write_text(json.dumps(data, indent=2), encoding="utf-8")
            return out
    data = train[seed % len(train)]
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def objects_near_point(
    event,
    center: dict[str, float],
    *,
    radius: float = 1.5,
    movable_only: bool = True,
) -> list[str]:
    ids: list[str] = []
    cx = float(center["x"])
    cz = float(center["z"])
    for obj in event.metadata.get("objects") or []:
        oid = obj.get("objectId")
        if not oid:
            continue
        if movable_only and not (obj.get("pickupable") or obj.get("moveable")):
            continue
        pos = obj.get("position") or {}
        if distance_xz(float(pos.get("x", 0.0)), float(pos.get("z", 0.0)), cx, cz) <= radius:
            ids.append(str(oid))
    return ids


def _point_in_polygon(x: float, z: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, zi = polygon[i]
        xj, zj = polygon[j]
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def stage_doorway_blocker(
    controller,
    doorway: dict[str, float],
    room: dict[str, Any],
    *,
    offset_m: float = 1.4,
) -> str | None:
    """Legacy single-object staging; prefer stage_doorway_blockers."""
    frame = {
        "center": doorway,
        "along": {"x": 0.0, "z": 1.0},
        "normal": {"x": 1.0, "z": 0.0},
        "width": 1.5,
    }
    ids = stage_doorway_blockers(controller, frame, room, count=1)
    return ids[0] if ids else None


def push_toward_point(
    controller,
    target: dict[str, float],
    *,
    magnitude: float,
    radius: float = 2.5,
    object_ids: set[str] | list[str] | None = None,
) -> int:
    from hazard.utils import find_objects

    event = controller.last_event
    tx = float(target["x"])
    tz = float(target["z"])
    id_filter = set(object_ids) if object_ids else None
    pushed = 0
    for obj in find_objects(
        event,
        predicate=lambda o: (o.get("pickupable") or o.get("moveable")) and not o.get("isBroken"),
    ):
        oid = str(obj.get("objectId") or "")
        if id_filter is not None and oid not in id_filter:
            continue
        pos = obj.get("position") or {}
        px = float(pos.get("x", 0.0))
        pz = float(pos.get("z", 0.0))
        if distance_xz(px, pz, tx, tz) > radius:
            continue
        angle = math.degrees(math.atan2(tx - px, tz - pz))
        res = controller.step(
            action="DirectionalPush",
            objectId=obj["objectId"],
            moveMagnitude=magnitude,
            pushAngle=angle,
            forceAction=True,
        )
        if res.metadata.get("lastActionSuccess", False):
            pushed += 1
    return pushed
