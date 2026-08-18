"""Shared AI2-THOR helpers: controller factory, poses, reachability, path queries."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Repo root (parent of ai2thor_pipeline/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCENES_FILE = _REPO_ROOT / "ai2thor_pipeline" / "selected_scenes.txt"
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "output_ai2thor"

# Curated iTHOR floor plans (~10 scenes, all four room types)
CURATED_ITHOR_SCENES: list[tuple[str, str]] = [
    ("FloorPlan1", "kitchen"),
    ("FloorPlan5", "kitchen"),
    ("FloorPlan10", "kitchen"),
    ("FloorPlan201", "living_room"),
    ("FloorPlan205", "living_room"),
    ("FloorPlan301", "bedroom"),
    ("FloorPlan305", "bedroom"),
    ("FloorPlan401", "bathroom"),
    ("FloorPlan405", "bathroom"),
    ("FloorPlan15", "kitchen"),
]

GRID_SIZE = 0.25
ROTATE_STEP_DEG = 15
CONTROLLER_GRID = 0.10
GOAL_REACH_M = 0.65


def repo_root() -> Path:
    return _REPO_ROOT


def output_root() -> Path:
    return _DEFAULT_OUTPUT_ROOT


def scene_graph_dir(scene: str) -> Path:
    return output_root() / "scene_graphs" / scene


def load_scene_list(scenes_file: Path | None = None) -> list[str]:
    path = scenes_file or _DEFAULT_SCENES_FILE
    if not path.is_file():
        return [s for s, _ in CURATED_ITHOR_SCENES]
    scenes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scenes.append(line.split()[0])
    return scenes


def write_scene_list(scenes_file: Path, scenes: list[str]) -> None:
    scenes_file.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Curated iTHOR floor plans (kitchen / living / bedroom / bathroom)\n"]
    type_map = {s: t for s, t in CURATED_ITHOR_SCENES}
    for name in scenes:
        room = type_map.get(name, "unknown")
        lines.append(f"{name}  # {room}\n")
    scenes_file.write_text("".join(lines), encoding="utf-8")


def make_controller(
    scene: str,
    *,
    headless: bool = False,
    width: int = 640,
    height: int = 480,
    render_depth: bool = True,
    local_executable_path: str | None = None,
):
    """Create an AI2-THOR Controller for iTHOR (default agent)."""
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    kwargs: dict[str, Any] = {
        "scene": scene,
        "gridSize": CONTROLLER_GRID,
        "rotateStepDegrees": ROTATE_STEP_DEG,
        "snapToGrid": False,
        "width": width,
        "height": height,
        "renderDepthImage": render_depth,
    }
    if local_executable_path:
        kwargs["local_executable_path"] = str(local_executable_path)
    if headless and sys.platform.startswith("linux"):
        kwargs["platform"] = CloudRendering
    return Controller(**kwargs)


def agent_pose(event) -> tuple[float, float, float]:
    """Return (x, z, yaw_deg) from metadata agent."""
    pos = event.metadata["agent"]["position"]
    rot = event.metadata["agent"]["rotation"]
    return float(pos["x"]), float(pos["z"]), float(rot["y"])


def distance_xz(ax: float, az: float, bx: float, bz: float) -> float:
    return math.hypot(ax - bx, az - bz)


def wrap_angle_deg(a: float) -> float:
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def yaw_toward(ax: float, az: float, bx: float, bz: float) -> float:
    """Yaw in degrees (Unity y-rotation) from (ax,az) toward (bx,bz)."""
    dx = bx - ax
    dz = bz - az
    return math.degrees(math.atan2(dx, dz))


def get_reachable_positions(controller) -> list[dict[str, float]]:
    ev = controller.step(action="GetReachablePositions")
    if not ev.metadata.get("lastActionSuccess", False):
        return []
    return list(ev.metadata.get("actionReturn") or [])


def pos_dict(x: float, y: float, z: float) -> dict[str, float]:
    return {"x": float(x), "y": float(y), "z": float(z)}


def teleport(
    controller,
    x: float,
    y: float,
    z: float,
    yaw: float = 0.0,
    *,
    horizon: float = 0.0,
):
    return controller.step(
        action="Teleport",
        position=pos_dict(x, y, z),
        rotation={"x": 0.0, "y": float(yaw), "z": 0.0},
        horizon=float(horizon),
        standing=True,
    )


def get_shortest_path(
    controller,
    x: float,
    z: float,
    *,
    y: float | None = None,
) -> tuple[list[dict[str, float]] | None, float | None]:
    """Query GetShortestPathToPoint from current agent pose to (x,z)."""
    agent = controller.last_event.metadata["agent"]["position"]
    target_y = float(y if y is not None else agent["y"])
    init_pos = pos_dict(float(agent["x"]), float(agent["y"]), float(agent["z"]))
    target = pos_dict(float(x), target_y, float(z))
    ev = controller.step(
        action="GetShortestPathToPoint",
        position=init_pos,
        target=target,
    )
    if not ev.metadata.get("lastActionSuccess", False):
        return None, None
    ret = ev.metadata.get("actionReturn") or {}
    path = ret.get("corners") or ret.get("path") or []
    if not path:
        return None, None
    dist = float(ret.get("distance", 0.0))
    if dist <= 0.0 and len(path) > 1:
        dist = 0.0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            dist += distance_xz(float(a["x"]), float(a["z"]), float(b["x"]), float(b["z"]))
    return path, dist


def farthest_point_sample(
    points: list[tuple[float, float, float]],
    k: int,
    seed_idx: int = 0,
) -> list[tuple[float, float, float]]:
    """Greedy farthest-point sampling on (x,y,z) tuples."""
    if not points or k <= 0:
        return []
    if k >= len(points):
        return list(points)
    selected = [points[seed_idx]]
    remaining = [p for i, p in enumerate(points) if i != seed_idx]
    while len(selected) < k and remaining:
        best_i = 0
        best_d = -1.0
        for i, p in enumerate(remaining):
            d_min = min(distance_xz(p[0], p[2], s[0], s[2]) for s in selected)
            if d_min > best_d:
                best_d = d_min
                best_i = i
        selected.append(remaining.pop(best_i))
    return selected


def snap_anchors_to_reachable(
    anchors: list[dict],
    reachable: list[dict[str, float]],
) -> list[dict]:
    """Replace each anchor with nearest reachable position (keep label)."""
    out: list[dict] = []
    for a in anchors:
        snap = snap_to_reachable(float(a["x"]), float(a["z"]), reachable, max_dist=3.0)
        if snap:
            out.append({**a, "x": snap[0], "y": snap[1], "z": snap[2], "snapped": True})
        else:
            out.append({**a, "snapped": False})
    return out


def snap_to_reachable(
    target_x: float,
    target_z: float,
    reachable: list[dict[str, float]],
    max_dist: float = 1.5,
) -> tuple[float, float, float] | None:
    """Return nearest reachable (x,y,z) to target within max_dist."""
    if not reachable:
        return None
    best = None
    best_d = float("inf")
    for p in reachable:
        d = distance_xz(target_x, target_z, float(p["x"]), float(p["z"]))
        if d < best_d:
            best_d = d
            best = (float(p["x"]), float(p["y"]), float(p["z"]))
    if best is None or best_d > max_dist:
        return None
    return best


def rasterize_reachable(
    reachable: list[dict[str, float]],
    resolution: float = 0.10,
    padding: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Rasterize reachable positions to a binary occupancy grid.
    Returns (grid bool HxW, meta with origin_x, origin_z, resolution, height).
    Grid axes: row = z, col = x (image coordinates).
    """
    if not reachable:
        return np.zeros((1, 1), dtype=bool), {
            "origin_x": 0.0,
            "origin_z": 0.0,
            "resolution": resolution,
            "height": 0.0,
        }
    xs = [p["x"] for p in reachable]
    zs = [p["z"] for p in reachable]
    ys = [p["y"] for p in reachable]
    min_x, max_x = min(xs) - padding, max(xs) + padding
    min_z, max_z = min(zs) - padding, max(zs) + padding
    w = max(1, int(math.ceil((max_x - min_x) / resolution)))
    h = max(1, int(math.ceil((max_z - min_z) / resolution)))
    grid = np.zeros((h, w), dtype=bool)
    for p in reachable:
        c = int((p["x"] - min_x) / resolution)
        r = int((p["z"] - min_z) / resolution)
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = True
    meta = {
        "origin_x": min_x,
        "origin_z": min_z,
        "resolution": resolution,
        "height": float(np.median(ys)),
        "width_cells": w,
        "height_cells": h,
    }
    return grid, meta


def world_to_grid(x: float, z: float, meta: dict[str, float]) -> tuple[int, int]:
    res = meta["resolution"]
    c = int((x - meta["origin_x"]) / res)
    r = int((z - meta["origin_z"]) / res)
    return r, c


def grid_to_world(r: int, c: int, meta: dict[str, float]) -> tuple[float, float]:
    res = meta["resolution"]
    x = meta["origin_x"] + (c + 0.5) * res
    z = meta["origin_z"] + (r + 0.5) * res
    return x, z


def rotate_toward(
    controller,
    target_yaw: float,
    tolerance_deg: float | None = None,
    *,
    rotate_deg: float | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> tuple[Any, int]:
    """Rotate left/right until within tolerance. Returns (last_event, num_steps)."""
    step_deg = ROTATE_STEP_DEG if rotate_deg is None else float(rotate_deg)
    tol = step_deg if tolerance_deg is None else float(tolerance_deg)
    rot_kw = {} if rotate_deg is None else {"degrees": float(rotate_deg)}
    steps = 0
    event = controller.last_event
    while True:
        _, _, yaw = agent_pose(event)
        err = wrap_angle_deg(target_yaw - yaw)
        if abs(err) <= tol:
            break
        if err > 0:
            event = controller.step(action="RotateRight", **rot_kw)
        else:
            event = controller.step(action="RotateLeft", **rot_kw)
        steps += 1
        if on_event is not None:
            on_event(event)
        if steps > int(math.ceil(360.0 / max(step_deg, 1.0))) + 4:
            break
    return event, steps


def follow_path_discrete(
    controller,
    path: list[dict[str, float]],
    *,
    corner_idx_start: int = 1,
    max_failures: int = 3,
    max_consecutive_skips: int = 3,
    max_steps: int = 5000,
    step_m: float | None = None,
    rotate_deg: float | None = None,
    on_event: Callable[[Any], None] | None = None,
) -> tuple[Any, int, int, int]:
    """
    Follow path corners with rotate-then-move. Returns (event, steps, failures, corners_reached).
    """
    reach_m = GRID_SIZE * 0.6 if step_m is None else max(step_m * 1.2, 0.05)
    step_deg = ROTATE_STEP_DEG if rotate_deg is None else float(rotate_deg)
    rot_trigger = step_deg * 1.5
    rot_tol = step_deg
    move_kw = {} if step_m is None else {"moveMagnitude": float(step_m)}
    event = controller.last_event
    total_steps = 0
    failures = 0
    corners = 0
    consecutive_skips = 0
    for i in range(corner_idx_start, len(path)):
        wp = path[i]
        tx, tz = float(wp["x"]), float(wp["z"])
        waypoint_failures = 0
        while True:
            if total_steps >= max_steps:
                return event, total_steps, failures, corners
            ax, az, yaw = agent_pose(event)
            if distance_xz(ax, az, tx, tz) < reach_m:
                corners += 1
                consecutive_skips = 0
                break
            target_yaw = yaw_toward(ax, az, tx, tz)
            err = abs(wrap_angle_deg(target_yaw - yaw))
            if err > rot_trigger:
                event, n = rotate_toward(
                    controller,
                    target_yaw,
                    tolerance_deg=rot_tol,
                    rotate_deg=rotate_deg,
                    on_event=on_event,
                )
                total_steps += max(n, 1)
            else:
                event = controller.step(action="MoveAhead", **move_kw)
                total_steps += 1
                if on_event is not None:
                    on_event(event)
                if not event.metadata.get("lastActionSuccess", True):
                    failures += 1
                    waypoint_failures += 1
                    dodge = rotate_deg if rotate_deg is not None else ROTATE_STEP_DEG
                    event, n = rotate_toward(
                        controller,
                        yaw + dodge,
                        tolerance_deg=rot_tol,
                        rotate_deg=rotate_deg,
                        on_event=on_event,
                    )
                    total_steps += max(n, 1)
                    if waypoint_failures >= max_failures:
                        consecutive_skips += 1
                        if consecutive_skips >= max_consecutive_skips:
                            return event, total_steps, failures, corners
                        break
    return event, total_steps, failures, corners
