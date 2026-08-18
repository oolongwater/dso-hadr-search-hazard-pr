"""Shared helpers for AI2-THOR Unity PhysX hazard demos."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from core.thor import agent_pose, output_root
from core.video import rgb_to_bgr_uint8

PHYSICS_TIMESTEP = 0.05
FPS = 10.0
FOG_COLOR_BGR = (180, 180, 190)

HAZARD_FAMILY_DIRS = {
    "smoke": "fire_smoke",
    "scene_smoke": "fire_smoke",
    "falling": "earthquake",
    "scene_earthquake": "earthquake",
    "blocked": "obstruction",
    "scene_obstruction": "obstruction",
    "compare_smoke": "fire_smoke",
    "compare_earthquake": "earthquake",
    "compare_obstruction": "obstruction",
}

# Longest scene_<family> prefix wins for variant names like scene_smoke_smolder.
_SCENE_FAMILY_PREFIXES = sorted(
    (k for k in HAZARD_FAMILY_DIRS if k.startswith("scene_")),
    key=len,
    reverse=True,
)


def hazard_family_dir(demo_name: str) -> str:
    if demo_name in HAZARD_FAMILY_DIRS:
        return HAZARD_FAMILY_DIRS[demo_name]
    for prefix in _SCENE_FAMILY_PREFIXES:
        if demo_name == prefix or demo_name.startswith(prefix + "_"):
            return HAZARD_FAMILY_DIRS[prefix]
    raise ValueError(f"No hazard family folder mapping for demo '{demo_name}'")


def hazard_output_dir() -> Path:
    path = output_root() / "hazards"
    path.mkdir(parents=True, exist_ok=True)
    return path


def hazard_artifact_paths(scene: str, demo_name: str) -> dict[str, Path]:
    root = hazard_output_dir() / hazard_family_dir(demo_name)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "video": root / f"{demo_name}_{scene}.mp4",
        "summary": root / f"{demo_name}_{scene}.summary.json",
        "frames_dir": root / f"{demo_name}_{scene}_frames",
    }


def make_video_writer(path: Path, frame) -> cv2.VideoWriter:
    bgr = rgb_to_bgr_uint8(frame)
    height, width = bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, FPS, (width, height))


def write_frame(writer: cv2.VideoWriter, frame) -> None:
    writer.write(rgb_to_bgr_uint8(frame))


def capture_event_frame(
    writer: cv2.VideoWriter | None,
    event,
    *,
    postprocess: Callable[[Any], Any] | None = None,
) -> tuple[cv2.VideoWriter | None, int]:
    if event.frame is None:
        return writer, 0
    frame = event.frame
    if postprocess is not None:
        frame = postprocess(frame)
    if writer is None:
        writer = make_video_writer(_pending_video_path, frame)
    write_frame(writer, frame)
    return writer, 1


_pending_video_path: Path = Path("hazard_demo.mp4")


def set_pending_video_path(path: Path) -> None:
    global _pending_video_path
    _pending_video_path = path


def pause_physics(controller):
    return controller.step(action="PausePhysicsAutoSim")


def unpause_physics(controller):
    return controller.step(action="UnpausePhysicsAutoSim")


def advance_physics(controller, time_step: float = PHYSICS_TIMESTEP):
    return controller.step(action="AdvancePhysicsStep", timeStep=float(time_step))


def stepped_physics_capture(
    controller,
    n_steps: int,
    *,
    time_step: float = PHYSICS_TIMESTEP,
    writer: cv2.VideoWriter | None = None,
    postprocess: Callable[[Any], Any] | None = None,
    ensure_paused: bool = True,
):
    """Advance Unity PhysX manually and capture one frame per step."""
    event = controller.last_event
    if ensure_paused:
        event = pause_physics(controller)
    frames = 0
    for _ in range(n_steps):
        event = advance_physics(controller, time_step)
        writer, n = capture_event_frame(writer, event, postprocess=postprocess)
        frames += n
    return event, writer, frames


def object_state_snapshot(event) -> dict[str, dict[str, Any]]:
    """Capture position and dynamic flags for all scene objects."""
    snap: dict[str, dict[str, Any]] = {}
    for obj in event.metadata.get("objects") or []:
        pos = obj.get("position") or {}
        snap[obj["objectId"]] = {
            "objectType": obj.get("objectType"),
            "name": obj.get("name"),
            "x": float(pos.get("x", 0.0)),
            "y": float(pos.get("y", 0.0)),
            "z": float(pos.get("z", 0.0)),
            "isMoving": bool(obj.get("isMoving", False)),
            "isBroken": bool(obj.get("isBroken", False)),
            "visible": bool(obj.get("visible", False)),
            "temperature": obj.get("temperature"),
            "isCooked": bool(obj.get("isCooked", False)),
        }
    return snap


def diff_states(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    *,
    min_position_delta_m: float = 0.05,
) -> list[dict[str, Any]]:
    """Return objects whose pose or dynamic flags changed."""
    changes: list[dict[str, Any]] = []
    for object_id, after_state in after.items():
        before_state = before.get(object_id)
        if before_state is None:
            continue
        dx = after_state["x"] - before_state["x"]
        dy = after_state["y"] - before_state["y"]
        dz = after_state["z"] - before_state["z"]
        pos_delta = math.sqrt(dx * dx + dy * dy + dz * dz)
        flag_changed = (
            after_state["isMoving"] != before_state["isMoving"]
            or after_state["isBroken"] != before_state["isBroken"]
            or after_state.get("temperature") != before_state.get("temperature")
            or after_state.get("isCooked") != before_state.get("isCooked")
        )
        if pos_delta >= min_position_delta_m or flag_changed:
            changes.append(
                {
                    "objectId": object_id,
                    "objectType": after_state.get("objectType"),
                    "position_delta_m": round(pos_delta, 4),
                    "before": before_state,
                    "after": after_state,
                }
            )
    changes.sort(key=lambda item: item["position_delta_m"], reverse=True)
    return changes


FALL_DELTA_M = 0.25


def objects_fallen(
    baseline: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    *,
    min_drop_m: float = FALL_DELTA_M,
) -> list[str]:
    """Return object ids displaced by at least min_drop_m vertically or in 3D."""
    fallen: list[str] = []
    for oid, base in baseline.items():
        end = after.get(oid)
        if end is None:
            continue
        drop = float(base["y"]) - float(end["y"])
        dx = float(end["x"]) - float(base["x"])
        dy = float(end["y"]) - float(base["y"])
        dz = float(end["z"]) - float(base["z"])
        if drop >= min_drop_m or math.dist((0.0, 0.0, 0.0), (dx, dy, dz)) >= min_drop_m:
            fallen.append(oid)
    return fallen


def apply_fog_overlay(frame, density: float):
    """Composite image-space fog; density in [0, 1]."""
    density = float(np.clip(density, 0.0, 1.0))
    if density <= 0.0:
        return frame
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC frame, got {arr.shape}")
    fog = np.full(arr.shape, FOG_COLOR_BGR[::-1], dtype=np.float32)
    blended = arr * (1.0 - density) + fog * density
    return np.clip(blended, 0, 255).astype(np.uint8)


def visibility_metric_from_density(density: float) -> float:
    """Simple visibility/confidence score used by the smoke demo."""
    return round(max(0.0, 1.0 - float(density)), 4)


HEAT_PANEL_VMIN = 22.0
HEAT_PANEL_VMAX = 600.0
HEAT_COLORBAR_WIDTH = 44
HEAT_OVERLAY_MAX_ALPHA = 0.8


def _world_to_grid(field, x: float, z: float) -> tuple[float, float] | None:
    norm_x = (x - (field.center_x - field.half_extent_x)) / (2.0 * field.half_extent_x)
    norm_z = (z - (field.center_z - field.half_extent_z)) / (2.0 * field.half_extent_z)
    if norm_x < 0.0 or norm_x > 1.0 or norm_z < 0.0 or norm_z > 1.0:
        return None
    return norm_x * (field.nx - 1), (1.0 - norm_z) * (field.nz - 1)


def render_heat_panel(
    field,
    overhead_bgr: np.ndarray,
    *,
    ignition_c: float = 180.0,
    agent_pos: dict[str, float] | None = None,
    stats: dict[str, Any] | None = None,
) -> np.ndarray:
    """Render greyscale top-down room view with heat dissipation overlaid."""
    height, map_w = overhead_bgr.shape[:2]
    gray = cv2.cvtColor(overhead_bgr, cv2.COLOR_BGR2GRAY)
    panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if field is not None and field.temperatures and field.nx > 0 and field.nz > 0:
        grid = np.array(field.temperatures, dtype=np.float32).reshape(field.nz, field.nx)
        grid = grid[::-1]
        norm = np.clip(
            (grid - HEAT_PANEL_VMIN) / max(1.0, HEAT_PANEL_VMAX - HEAT_PANEL_VMIN),
            0.0,
            1.0,
        )
        heat_u8 = (norm * 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)
        heat_bgr = cv2.resize(heat_bgr, (map_w, height), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(norm, (map_w, height), interpolation=cv2.INTER_LINEAR)
        alpha = (alpha * HEAT_OVERLAY_MAX_ALPHA)[:, :, np.newaxis]
        panel = (
            panel.astype(np.float32) * (1.0 - alpha) + heat_bgr.astype(np.float32) * alpha
        ).astype(np.uint8)

        if ignition_c > HEAT_PANEL_VMIN:
            iso_grid = (grid >= ignition_c).astype(np.uint8) * 255
            iso_resized = cv2.resize(iso_grid, (map_w, height), interpolation=cv2.INTER_LINEAR)
            _, iso_mask = cv2.threshold(iso_resized, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(iso_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)

        def _plot_point(x: float, z: float, color: tuple[int, int, int], radius: int = 4) -> None:
            coords = _world_to_grid(field, x, z)
            if coords is None:
                return
            px = int(coords[0] / max(1, field.nx - 1) * (map_w - 1))
            py = int(coords[1] / max(1, field.nz - 1) * (height - 1))
            cv2.circle(panel, (px, py), radius, color, -1, cv2.LINE_AA)

        for flame in field.flames:
            _plot_point(flame.x, flame.z, (0, 255, 255), radius=5)
        for obj in field.objects:
            if obj.thor_temperature == "Hot":
                _plot_point(obj.x, obj.z, (255, 255, 255), radius=3)
        if agent_pos is not None:
            _plot_point(float(agent_pos["x"]), float(agent_pos["z"]), (0, 255, 0), radius=6)

    colorbar = np.zeros((height, HEAT_COLORBAR_WIDTH, 3), dtype=np.uint8)
    for row in range(height):
        t = 1.0 - row / max(1, height - 1)
        val = int(np.clip(t * 255, 0, 255))
        colorbar[row, :] = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
    for temp, frac in ((600, 0.02), (400, 0.35), (200, 0.68), (22, 0.98)):
        y = int(frac * height)
        cv2.putText(
            colorbar, f"{temp}", (2, max(10, y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (240, 240, 240), 1, cv2.LINE_AA,
        )

    combined = np.hstack([panel, colorbar])
    cv2.putText(
        combined, "HEAT (C)", (6, 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA,
    )
    if stats:
        readout = (
            f"max={stats.get('max_temp_c', 0)}C  "
            f"agent={stats.get('agent_temp_c', 0)}C  "
            f"hot={stats.get('num_hot_objects', 0)}"
        )
        cv2.putText(
            combined, readout, (6, height - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA,
        )
    return combined


def find_objects(event, *, object_type: str | None = None, predicate=None) -> list[dict[str, Any]]:
    objs = list(event.metadata.get("objects") or [])
    if object_type is not None:
        objs = [o for o in objs if o.get("objectType") == object_type]
    if predicate is not None:
        objs = [o for o in objs if predicate(o)]
    return objs


def find_object(event, *, object_type: str, index: int = 0) -> dict[str, Any] | None:
    matches = find_objects(event, object_type=object_type)
    if not matches:
        return None
    return matches[min(index, len(matches) - 1)]


def move_object(
    controller,
    obj: dict[str, Any],
    position: dict[str, float],
    rotation: dict[str, float] | None = None,
):
    """Reposition an existing object via SetObjectPoses (PhysX-aware)."""
    rot = rotation or obj.get("rotation") or {"x": 0.0, "y": 0.0, "z": 0.0}
    return controller.step(
        action="SetObjectPoses",
        objectPoses=[
            {
                "objectName": obj["name"],
                "position": {
                    "x": float(position["x"]),
                    "y": float(position.get("y", obj["position"]["y"])),
                    "z": float(position["z"]),
                },
                "rotation": {
                    "x": float(rot.get("x", 0.0)),
                    "y": float(rot.get("y", 0.0)),
                    "z": float(rot.get("z", 0.0)),
                },
            }
        ],
    )


def settle_physics(controller, steps: int = 12):
    pause_physics(controller)
    event, _, _ = stepped_physics_capture(
        controller,
        steps,
        ensure_paused=False,
    )
    unpause_physics(controller)
    return event


def step_and_capture(
    controller,
    writer: cv2.VideoWriter | None,
    action: str,
    *,
    postprocess: Callable[[Any], Any] | None = None,
    **kwargs,
):
    event = controller.step(action=action, **kwargs) if kwargs else controller.step(action=action)
    writer, frames = capture_event_frame(writer, event, postprocess=postprocess)
    return event, writer, frames


def find_moveahead_success_pose(controller) -> tuple[dict[str, float], float] | None:
    """Search reachable poses for one where MoveAhead succeeds."""
    reachable = controller.step(action="GetReachablePositions").metadata.get("actionReturn") or []
    for point in reachable:
        for yaw in (0.0, 90.0, 180.0, 270.0):
            controller.step(
                action="Teleport",
                position=point,
                rotation={"x": 0.0, "y": float(yaw), "z": 0.0},
                horizon=0.0,
                standing=True,
            )
            ok = bool(controller.last_event.metadata.get("lastActionSuccess", False))
            if not ok:
                continue
            move = controller.step(action="MoveAhead")
            if move.metadata.get("lastActionSuccess", False):
                return dict(point), float(yaw)
    return None


def point_ahead(agent_pos: dict[str, float], yaw_deg: float, distance_m: float = 0.35) -> dict[str, float]:
    target_yaw = math.radians(yaw_deg)
    return {
        "x": float(agent_pos["x"]) + math.sin(target_yaw) * distance_m,
        "y": float(agent_pos.get("y", 0.9)),
        "z": float(agent_pos["z"]) + math.cos(target_yaw) * distance_m,
    }


def pickup_objects(controller, object_types: list[str]) -> list[str]:
    """Pick up one object per type using forceAction."""
    picked: list[str] = []
    for object_type in object_types:
        obj = find_object(controller.last_event, object_type=object_type)
        if obj is None:
            continue
        event = controller.step(
            action="PickupObject",
            objectId=obj["objectId"],
            forceAction=True,
        )
        if event.metadata.get("lastActionSuccess", False):
            picked.append(obj["objectId"])
    return picked


def extract_sample_frames(video_path: Path, out_dir: Path, indices: list[int]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    saved: list[Path] = []
    frame_i = 0
    want = set(indices)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i in want:
            path = out_dir / f"frame_{frame_i:04d}.png"
            cv2.imwrite(str(path), frame)
            saved.append(path)
        frame_i += 1
    cap.release()
    return saved
