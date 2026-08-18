"""AI2-THOR helpers for placing and loading changepoints (requires ai2thor)."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from changepoint_kit.changepoint import Changepoint, ChangepointLog

MANUAL_CLUSTER_RADIUS_M = 1.5


def _distance_xz(ax: float, az: float, bx: float, bz: float) -> float:
    return math.hypot(ax - bx, az - bz)


def get_reachable_positions(controller) -> list[dict[str, float]]:
    ev = controller.step(action="GetReachablePositions")
    if not ev.metadata.get("lastActionSuccess", False):
        return []
    return list(ev.metadata.get("actionReturn") or [])


def nearest_reachable(
    controller,
    x: float,
    z: float,
    *,
    max_dist: float = 1.0,
) -> dict[str, float] | None:
    """Nearest navmesh position to (x, z), or None if farther than max_dist."""
    best: dict[str, float] | None = None
    best_d = float("inf")
    for p in get_reachable_positions(controller):
        px, pz = float(p["x"]), float(p["z"])
        d = _distance_xz(x, z, px, pz)
        if d < best_d:
            best_d = d
            best = {"x": px, "y": float(p["y"]), "z": pz}
    if best is None or best_d > max_dist:
        return None
    return best


def objects_near(
    event,
    x: float,
    z: float,
    radius_m: float,
) -> tuple[list[str], list[str]]:
    """Pickupable/moveable non-broken objects within radius_m of (x, z)."""
    hits: list[tuple[str, str]] = []
    for obj in event.metadata.get("objects") or []:
        if not (obj.get("pickupable") or obj.get("moveable")):
            continue
        if obj.get("isBroken"):
            continue
        pos = obj.get("position") or {}
        ox, oz = float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
        if _distance_xz(x, z, ox, oz) > radius_m:
            continue
        oid = str(obj.get("objectId") or "")
        otype = str(obj.get("objectType") or "")
        if oid:
            hits.append((oid, otype))
    hits.sort(key=lambda t: t[0])
    ids = [h[0] for h in hits]
    types: list[str] = []
    seen: set[str] = set()
    for _, otype in hits:
        if otype and otype not in seen:
            seen.add(otype)
            types.append(otype)
    return ids, types


def _cluster_type_summary(types: list[str]) -> str:
    counts = Counter(types)
    parts = [f"{c}x {t}" if c > 1 else t for t, c in counts.most_common()]
    return ", ".join(parts)


def _next_manual_id(log: ChangepointLog | None) -> str:
    n = 0
    if log is not None:
        for cp in log.records:
            if cp.id.startswith("cp_manual_"):
                try:
                    n = max(n, int(cp.id.split("_")[-1]) + 1)
                except ValueError:
                    pass
    return f"cp_manual_{n}"


def _resolve_heading(
    controller,
    x: float,
    z: float,
    *,
    heading_deg: float | None,
    face: tuple[float, float] | None,
) -> float:
    if heading_deg is not None:
        return float(heading_deg)
    if face is not None:
        tx, tz = face
        return math.degrees(math.atan2(tx - x, tz - z))
    agent = controller.last_event.metadata.get("agent") or {}
    rot = agent.get("rotation") or {}
    return float(rot.get("y", 0.0))


def make_changepoint_at(
    controller,
    x: float,
    z: float,
    *,
    cp_id: str = "",
    heading_deg: float | None = None,
    face: tuple[float, float] | None = None,
    snap: bool = True,
    max_snap_dist: float = 1.0,
    cluster_radius_m: float = MANUAL_CLUSTER_RADIUS_M,
    room_ids: list[str] | None = None,
    door_id: str | None = None,
    log: ChangepointLog | None = None,
) -> Changepoint:
    """Drop a changepoint record at (x, z) on the live AI2-THOR navmesh."""
    if snap:
        snapped = nearest_reachable(controller, x, z, max_dist=max_snap_dist)
        if snapped is None:
            nearest = nearest_reachable(controller, x, z, max_dist=1e9)
            dist = (
                _distance_xz(x, z, nearest["x"], nearest["z"])
                if nearest
                else float("inf")
            )
            raise ValueError(
                f"no reachable position within {max_snap_dist:.2f}m of ({x}, {z}); "
                f"nearest navmesh point is {dist:.2f}m away"
            )
        wx, wy, wz = snapped["x"], snapped["y"], snapped["z"]
    else:
        agent = controller.last_event.metadata.get("agent") or {}
        pos = agent.get("position") or {}
        wx, wz = float(x), float(z)
        wy = float(pos.get("y", 0.9))

    heading = _resolve_heading(controller, wx, wz, heading_deg=heading_deg, face=face)
    event = controller.last_event
    c_ids, c_types = objects_near(event, wx, wz, cluster_radius_m)
    type_summary = _cluster_type_summary(c_types)
    node_id = cp_id or _next_manual_id(log)
    cluster_txt = type_summary or "none"
    connectivity = f"manual drop @ ({wx:.1f}, {wz:.1f}); cluster: {cluster_txt}"

    cp = Changepoint(
        id=node_id,
        world={"x": wx, "y": wy, "z": wz},
        heading_deg=heading,
        source="manual",
        door_id=door_id,
        room_ids=list(room_ids or []),
        cluster_object_ids=c_ids,
        cluster_object_types=c_types,
        cluster_type_summary=type_summary,
        connectivity=connectivity,
        decision="proceed",
        decision_frame=f"proceed: manual drop / world yaw {heading:.0f}",
        blocked=False,
        exits=[],
    )
    if log is not None:
        log.append(cp)
    return cp
