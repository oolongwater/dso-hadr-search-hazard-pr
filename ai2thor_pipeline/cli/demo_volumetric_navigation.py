#!/usr/bin/env python3
"""Demo: agent navigates a ProcTHOR scene with live semantic volumetric map + earthquake."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.procthor_house import (
    default_local_executable,
    door_world_frame,
    house_schema,
    load_house_json,
    make_procedural_controller,
    reachable_on_floor,
    room_centroid,
)
from core.thor import (
    agent_pose,
    distance_xz,
    follow_path_discrete,
    get_reachable_positions,
    get_shortest_path,
    output_root,
    teleport,
)
from core.video import finalize_mp4, probe_video, rgb_to_bgr_uint8
from core.volumetric_map import (
    TRAV_NAMES,
    build_volume,
    clearance_at,
    color_map_from_event,
    column_class,
    overlay_instance_seg,
    render_nav_panel,
    restamp_objects,
    traversability_counts,
    visible_object_ids,
)
from hazard.functions import EarthquakeLatents, HazardConfig
from hazard.model import EarthquakeHazard
from hazard.utils import object_state_snapshot, unpause_physics

PANEL_HEIGHT = 720
NAV_STEP_M = 0.05
NAV_ROTATE_DEG = 5.0
NAV_CAPTURE_EVERY = 2
FPS = 15
QUAKE_ONSET_M = 1.0
QUAKE_WATCH_SECONDS = 8.0
HAZARD_TICK_EVERY = 8


def _default_house_path(label: str) -> Path:
    batch = Path(__file__).resolve().parents[1] / "assets" / "houses" / "batch"
    path = batch / f"{label}.json"
    if path.exists():
        return path
    return Path(__file__).resolve().parents[1] / "assets" / "houses" / f"{label}.json"


def _ground_floor_base_y(house: dict[str, Any]) -> float:
    floors = house.get("floors") or []
    if floors:
        return float(floors[0].get("baseY", 0.0))
    return 0.0


def _pick_destination(house: dict[str, Any], agent_x: float, agent_z: float) -> tuple[str, float, float]:
    rooms = {str(r.get("id")): r for r in (house.get("rooms") or []) if r.get("id")}
    if "room|7" in rooms:
        rc = room_centroid(rooms["room|7"])
        return "room|7", float(rc["x"]), float(rc["z"])
    best_id = ""
    best_d = -1.0
    gx, gz = agent_x, agent_z
    for rid, room in rooms.items():
        rc = room_centroid(room)
        d = distance_xz(agent_x, agent_z, float(rc["x"]), float(rc["z"]))
        if d > best_d:
            best_d = d
            best_id = rid
            gx, gz = float(rc["x"]), float(rc["z"])
    return best_id or "goal", gx, gz


def _doorway_snapshot(vol, house: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-interior-door column class and clearance at doorway center."""
    out: dict[str, dict[str, Any]] = {}
    for door in house.get("doors") or []:
        r0 = str(door.get("room0") or "")
        r1 = str(door.get("room1") or "")
        if not r0 or not r1 or r0 == r1:
            continue
        door_id = str(door.get("id") or "")
        try:
            frame = door_world_frame(door, house)
        except ValueError:
            continue
        cx = float(frame["center"]["x"])
        cz = float(frame["center"]["z"])
        cls = column_class(vol, cx, cz)
        out[door_id] = {
            "x": round(cx, 3),
            "z": round(cz, 3),
            "class": int(cls),
            "class_name": TRAV_NAMES.get(cls, str(cls)),
            "clearance_m": round(clearance_at(vol, cx, cz), 3),
        }
    return out


def _compose_frame(
    event,
    *,
    vol_panel: np.ndarray,
    lines: list[str],
    use_seg_overlay: bool = True,
) -> np.ndarray | None:
    if event.frame is None:
        return None
    fpv_bgr = rgb_to_bgr_uint8(event.frame)
    if use_seg_overlay:
        fpv_bgr = overlay_instance_seg(fpv_bgr, event)
    h = PANEL_HEIGHT
    scale = h / fpv_bgr.shape[0]
    fpv_panel = cv2.resize(fpv_bgr, (int(fpv_bgr.shape[1] * scale), h))
    right = vol_panel
    if right.shape[0] != h:
        right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    divider = np.full((h, 4, 3), (60, 60, 60), dtype=np.uint8)
    combined = np.hstack([fpv_panel, divider, right])
    y = 22
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(combined, (6, y - th - 4), (10 + tw, y + 4), (30, 30, 30), -1)
        cv2.putText(combined, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        y += th + 10
    return combined


def run_demo(
    *,
    label: str,
    house_path: Path,
    out_video: Path,
    headless: bool,
    local_executable: Path | None,
    earthquake: bool,
    severity: float,
    seed: int,
) -> dict[str, Any]:
    house = load_house_json(house_path)
    controller = make_procedural_controller(
        house,
        headless=headless,
        render_depth=False,
        render_instance_segmentation=True,
        local_executable_path=str(local_executable) if local_executable else None,
    )
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    trail: list[tuple[float, float]] = []
    discovered: set[str] = set()
    hazard: EarthquakeHazard | None = None
    hazard_step = 0
    quake_started = False
    quake_nav_frames = 0
    peak_moving = 0
    peak_rise_m = 0.0
    objects_displaced = 0
    walk_start_path = 0.0
    agent_path_m = 0.0
    prev_ax, prev_az = 0.0, 0.0
    doorway_at_start: dict[str, dict[str, Any]] = {}
    doorway_at_quake_onset: dict[str, dict[str, Any]] | None = None
    doorway_at_end: dict[str, dict[str, Any]] = {}
    trav_at_start: dict[str, int] = {}
    trav_at_quake_onset: dict[str, int] | None = None
    trav_at_end: dict[str, int] = {}

    try:
        event0 = controller.last_event
        objects = list(event0.metadata.get("objects") or [])
        seg_colors = color_map_from_event(event0)
        vol = build_volume(house, objects, label=label, color_map=seg_colors)
        doorway_at_start = _doorway_snapshot(vol, house)
        trav_at_start = traversability_counts(vol)

        base_y = _ground_floor_base_y(house)
        reachable = get_reachable_positions(controller)
        if house_schema(house) == "2.0.0":
            reachable = reachable_on_floor(reachable, base_y)

        ax, az, ayaw = agent_pose(event0)
        prev_ax, prev_az = ax, az
        agent = event0.metadata["agent"]
        agent_y = float(agent["position"]["y"])
        dest_label, gx, gz = _pick_destination(house, ax, az)
        path, path_dist = get_shortest_path(controller, gx, gz, y=agent_y)
        if not path or len(path) < 2:
            raise RuntimeError(f"No path to destination {dest_label} at ({gx:.2f}, {gz:.2f})")

        if earthquake:
            pre_pose = event0.metadata["agent"]
            pre_pos = pre_pose["position"]
            pre_rot = pre_pose["rotation"]
            pre_horizon = float(pre_pose.get("cameraHorizon", 0.0))
            config = HazardConfig(
                hazard_type="earthquake",
                scene=label,
                severity=severity,
                seed=seed,
                onset_step=0,
                total_ticks=200,
                native_effects=True,
                earthquake=EarthquakeLatents(
                    impulse_base_newtons=160.0,
                    shake_period_ticks=3,
                    impulse_scale=0.35,
                    integrity_threshold=3.5,
                ),
            )
            hazard = EarthquakeHazard(config)
            hazard.setup(controller, reachable=reachable, floor_base_y=base_y)
            teleport(
                controller,
                float(pre_pos["x"]),
                float(pre_pos["y"]),
                float(pre_pos["z"]),
                yaw=float(pre_rot["y"]),
                horizon=pre_horizon,
            )
            unpause_physics(controller)

        route_world = [(float(p["x"]), float(p["z"])) for p in path]
        lines = [
            f"Volumetric nav — {label}",
            f"goal={dest_label}  path={path_dist:.1f}m  corners={len(path)}",
        ]

        def _capture(event, *, phase: str, shaking_s: float = 0.0) -> None:
            nonlocal writer, frames_written
            ax_now, az_now, yaw_now = agent_pose(event)
            trail.append((ax_now, az_now))
            discovered.update(visible_object_ids(event))
            objects_now = list(event.metadata.get("objects") or [])
            if shaking_s > 0.0:
                restamp_objects(vol, objects_now, color_map=color_map_from_event(event))
            cls = column_class(vol, ax_now, az_now)
            clr = clearance_at(vol, ax_now, az_now)
            total_objs = len(vol.objects_table)
            disc_n = sum(1 for e in vol.objects_table if e["objectId"] in discovered)
            cov = (100.0 * disc_n / total_objs) if total_objs else 0.0
            title = [
                f"semantic map ({phase})  {disc_n}/{total_objs} ({cov:.0f}%)",
                f"clearance={clr:.2f}m  class={TRAV_NAMES.get(cls, cls)}",
            ]
            if shaking_s > 0.0:
                title.append(f"earthquake {shaking_s:.1f}s  moving={peak_moving}")
            panel = render_nav_panel(
                vol,
                agent_x=ax_now,
                agent_z=az_now,
                agent_yaw=yaw_now,
                route_world=route_world,
                trail_world=trail,
                discovered_object_ids=discovered,
                title_lines=title,
                height=PANEL_HEIGHT,
            )
            frame = _compose_frame(event, vol_panel=panel, lines=lines)
            if frame is None:
                return
            if writer is None:
                out_video.parent.mkdir(parents=True, exist_ok=True)
                fh, fw = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_video), fourcc, FPS, (fw, fh))
            writer.write(frame)
            frames_written += 1

        _capture(controller.last_event, phase="start")

        nav_i = {"n": 0}

        def _on_event(event) -> None:
            nonlocal hazard_step, quake_started, quake_nav_frames, peak_moving, peak_rise_m
            nonlocal agent_path_m, prev_ax, prev_az, walk_start_path
            nonlocal doorway_at_quake_onset, trav_at_quake_onset
            nav_i["n"] += 1
            ax_now, az_now, _ = agent_pose(event)
            agent_path_m += distance_xz(prev_ax, prev_az, ax_now, az_now)
            prev_ax, prev_az = ax_now, az_now

            shaking_s = 0.0
            if hazard is not None:
                if not quake_started and (agent_path_m - walk_start_path) >= QUAKE_ONSET_M:
                    quake_started = True
                    quake_nav_frames = 0
                    hazard.baseline = object_state_snapshot(event)
                    restamp_objects(vol, list(event.metadata.get("objects") or []), color_map=color_map_from_event(event))
                    doorway_at_quake_onset = _doorway_snapshot(vol, house)
                    trav_at_quake_onset = traversability_counts(vol)
                if quake_started:
                    quake_nav_frames += 1
                    shaking_s = quake_nav_frames / float(FPS)
                    if nav_i["n"] % HAZARD_TICK_EVERY == 0:
                        report = hazard.tick(controller, hazard_step)
                        hazard_step += 1
                        peak_moving = max(peak_moving, int(report.get("num_moving") or 0))
                        peak_rise_m = max(peak_rise_m, float(report.get("max_rise_m") or 0.0))

            if nav_i["n"] % NAV_CAPTURE_EVERY == 0:
                _capture(event, phase="navigating", shaking_s=shaking_s)

        follow_path_discrete(
            controller,
            path,
            step_m=NAV_STEP_M,
            rotate_deg=NAV_ROTATE_DEG,
            on_event=_on_event,
        )

        if hazard is not None:
            if not quake_started:
                quake_started = True
                quake_nav_frames = 0
                hazard.baseline = object_state_snapshot(controller.last_event)
                restamp_objects(
                    vol,
                    list(controller.last_event.metadata.get("objects") or []),
                    color_map=color_map_from_event(controller.last_event),
                )
                doorway_at_quake_onset = _doorway_snapshot(vol, house)
                trav_at_quake_onset = traversability_counts(vol)
            watch_frames = int(QUAKE_WATCH_SECONDS * FPS)
            for watch_i in range(watch_frames):
                event = controller.step(action="Pass")
                quake_nav_frames += 1
                shaking_s = quake_nav_frames / float(FPS)
                if watch_i % HAZARD_TICK_EVERY == 0:
                    report = hazard.tick(controller, hazard_step)
                    hazard_step += 1
                    peak_moving = max(peak_moving, int(report.get("num_moving") or 0))
                    peak_rise_m = max(peak_rise_m, float(report.get("max_rise_m") or 0.0))
                if watch_i % NAV_CAPTURE_EVERY == 0:
                    _capture(event, phase="shaking", shaking_s=shaking_s)

        hazard_finalize: dict[str, Any] = {}
        if hazard is not None:
            hazard_finalize = hazard.finalize(controller)
            objects_displaced = int(hazard_finalize.get("num_state_changes") or 0)
        restamp_objects(
            vol,
            list(controller.last_event.metadata.get("objects") or []),
            color_map=color_map_from_event(controller.last_event),
        )
        doorway_at_end = _doorway_snapshot(vol, house)
        trav_at_end = traversability_counts(vol)
        _capture(controller.last_event, phase="arrived")

        if writer is not None:
            writer.release()
            writer = None
        finalize_mp4(out_video)

        total_objs = len(vol.objects_table)
        disc_n = sum(1 for e in vol.objects_table if e["objectId"] in discovered)
        return {
            "label": label,
            "destination": dest_label,
            "path_distance_m": path_dist,
            "path_corners": len(path),
            "frames_written": frames_written,
            "earthquake": earthquake,
            "severity": severity if earthquake else None,
            "objects_discovered": disc_n,
            "objects_total": total_objs,
            "discovery_coverage_pct": round(100.0 * disc_n / total_objs, 1) if total_objs else 0.0,
            "peak_num_moving": peak_moving,
            "peak_rise_m": round(peak_rise_m, 4),
            "objects_displaced": objects_displaced,
            "doorway_at_start": doorway_at_start,
            "doorway_at_quake_onset": doorway_at_quake_onset,
            "doorway_at_end": doorway_at_end,
            "traversability_at_start": trav_at_start,
            "traversability_at_quake_onset": trav_at_quake_onset,
            "traversability_at_end": trav_at_end,
            "video": str(out_video.resolve()),
            "house_json": str(house_path.resolve()),
        }
    finally:
        if writer is not None:
            writer.release()
        controller.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic volumetric map navigation demo")
    parser.add_argument("--label", default="four_room_ring_1f")
    parser.add_argument("--house", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--local-executable", type=Path, default=None)
    parser.add_argument("--earthquake", action="store_true", help="Enable native Unity earthquake mid-walk")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    house_path = args.house or _default_house_path(args.label)
    if not house_path.exists():
        raise SystemExit(f"House JSON not found: {house_path}")

    suffix = "_nav_quake" if args.earthquake else "_nav"
    out_video = args.out or (output_root() / "volumetric" / f"{args.label}{suffix}.mp4")
    exe = args.local_executable or default_local_executable()

    summary = run_demo(
        label=args.label,
        house_path=house_path,
        out_video=out_video,
        headless=args.headless,
        local_executable=exe,
        earthquake=args.earthquake,
        severity=args.severity,
        seed=args.seed,
    )

    summary_path = out_video.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    probe = probe_video(out_video)
    print(f"Wrote {out_video} ({probe.get('duration_s', '?')}s, {summary['frames_written']} frames)")
    print(f"Wrote {summary_path}")
    if summary.get("earthquake"):
        print(
            f"Discovery: {summary['objects_discovered']}/{summary['objects_total']} "
            f"({summary['discovery_coverage_pct']}%), peak_moving={summary['peak_num_moving']}"
        )


if __name__ == "__main__":
    main()
