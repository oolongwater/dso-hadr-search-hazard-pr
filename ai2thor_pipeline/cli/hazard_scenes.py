#!/usr/bin/env python3
"""
Parameterized hazard scenes for AI2-THOR (proposal section 3.1.1).

Runs one of three hazard families (smoke / earthquake / obstruction) as a
causally evolving scene driven by latent variables, and renders an MP4 that
composites the agent first-person view with an overhead map view.

Each hazard exposes latent variables (severity, onset, spread/growth, seed)
and a transparent per-tick propagation rule (see hazard/model.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.thor import make_controller, output_root
from core.video import finalize_mp4, probe_video, rgb_to_bgr_uint8
from hazard.functions import HazardConfig
from hazard.model import build_hazard
from hazard.utils import (
    advance_physics,
    apply_fog_overlay,
    extract_sample_frames,
    hazard_artifact_paths,
    pause_physics,
    render_heat_panel,
    unpause_physics,
)
from scene_graph.export import (
    apply_hazard_obs,
    build_initial_scene_graph,
    diff_scene_graph,
    scene_graph_summary_block,
    write_scene_graph,
    _thor_id_to_node_id,
)
from scene_graph.panel import (
    build_graph,
    compute_layout,
    hazard_graph_edges,
    heat_state_ids,
    parent_snapshot,
    render_graph_panel,
    render_passage_panel,
    room_center_from_map_props,
    room_center_from_objects,
    select_tracked_nodes,
    _as_dict_list,
)

CAPTURE_SUBSTEPS = 3
LABEL_COLOR = (255, 255, 255)
LABEL_BG = (0, 0, 0)

# Per-scene capture profiles: (fps, substeps_per_tick, hold_paused, time_step).
# substeps * time_step is the simulated seconds per tick and must stay fixed when
# changing fps, otherwise hazard propagation (and thermal_delta_time) shifts too.
SCENE_CAPTURE_PROFILES = {
    "earthquake": {"fps": 30, "substeps": 8, "hold_paused": True, "time_step": 1.0 / 30.0},
    # smoke captures 12 frames per tick but writes at 30 fps, so the 8 s of simulated
    # time plays back over 16 s: twice the frame density at half speed.
    "smoke": {"fps": 30, "substeps": 12, "hold_paused": False, "time_step": 1.0 / 60.0},
    "obstruction": {"fps": 30, "substeps": CAPTURE_SUBSTEPS * 3, "hold_paused": False, "time_step": 0.05 / 3.0},
}
SCENE_TICK_DEFAULTS = {
    "smoke": 70,
}
SCENE_TAIL_FRAMES = {
    "smoke": 60,
}
DEFAULT_TICKS = 40
DEFAULT_TAIL_FRAMES = 16


def capture_profile(name: str, fps_override: int | None) -> dict[str, Any]:
    profile = dict(SCENE_CAPTURE_PROFILES.get(name, SCENE_CAPTURE_PROFILES["smoke"]))
    if fps_override is not None:
        profile["fps"] = fps_override
    return profile


def _shift_fpv(fpv_bgr: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """Image-space camera shake: translate the FPV half, edge-replicate borders."""
    dx, dy = int(shift[0]), int(shift[1])
    if dx == 0 and dy == 0:
        return fpv_bgr
    h, w = fpv_bgr.shape[:2]
    mat = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(fpv_bgr, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)


def add_overhead_camera(controller) -> bool:
    props_event = controller.step(action="GetMapViewCameraProperties")
    props = props_event.metadata.get("actionReturn")
    if not props or not props_event.metadata.get("lastActionSuccess", False):
        return False
    event = controller.step(
        action="AddThirdPartyCamera",
        position=props["position"],
        rotation=props["rotation"],
        orthographic=props.get("orthographic", True),
        orthographicSize=props.get("orthographicSize", 3.0),
    )
    return bool(event.metadata.get("lastActionSuccess", False))


def _label_lines(name: str, step: int, report: dict[str, Any]) -> list[str]:
    lines = [f"{name.upper()}  step {step}"]
    if name == "smoke":
        lines.append(f"visibility={report.get('visibility')}  density={report.get('agent_density')}")
        lines.append(
            f"max={report.get('max_temp_c')}C  agent={report.get('agent_temp_c')}C  "
            f"hot={report.get('num_hot_objects')}  broken={report.get('num_broken_total')}"
        )
    elif name == "earthquake":
        lines.append(f"pushed={report.get('num_pushed')}  moving={report.get('num_moving')}")
        lines.append(f"broken={report.get('num_broken_total')}  impulse={report.get('impulse_newtons')}N")
    elif name == "obstruction":
        lines.append(f"edge={report.get('edge_state')}  placed={report.get('placed_count')}")
        lines.append(f"path_cost={report.get('path_cost')}  move_ok={report.get('moveahead_ok')}")
    return lines


def _draw_labels(bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    y = 22
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(bgr, (6, y - th - 4), (10 + tw, y + 4), LABEL_BG, -1)
        cv2.putText(bgr, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR, 1, cv2.LINE_AA)
        y += th + 10
    return bgr


def _compose_frame(
    event,
    density: float,
    lines: list[str],
    fpv_shift: tuple[int, int] = (0, 0),
    heat_overlay: dict[str, Any] | None = None,
    graph_ctx: dict[str, Any] | None = None,
) -> np.ndarray | None:
    if event.frame is None:
        return None
    fpv_rgb = apply_fog_overlay(event.frame, density) if density > 0 else event.frame
    fpv_bgr = rgb_to_bgr_uint8(fpv_rgb)
    if fpv_shift != (0, 0):
        fpv_bgr = _shift_fpv(fpv_bgr, fpv_shift)
    h = fpv_bgr.shape[0]

    panels: list[np.ndarray] = [fpv_bgr]
    tpf = getattr(event, "third_party_camera_frames", None)
    over_bgr = None
    if tpf:
        over_bgr = rgb_to_bgr_uint8(tpf[0])
        scale = h / over_bgr.shape[0]
        over_bgr = cv2.resize(over_bgr, (int(over_bgr.shape[1] * scale), h))
        panels.append(over_bgr)
    if heat_overlay is not None and over_bgr is not None:
        heat_bgr = render_heat_panel(
            heat_overlay.get("field"),
            over_bgr,
            ignition_c=float(heat_overlay.get("ignition_c", 180.0)),
            agent_pos=heat_overlay.get("agent_pos"),
            stats=heat_overlay.get("stats"),
        )
        panels.append(heat_bgr)

    if graph_ctx is not None:
        if graph_ctx.get("hazard") == "obstruction":
            panels.append(
                render_passage_panel(
                    event,
                    graph_ctx.get("blocker_ids") or [],
                    graph_ctx.get("placed_ids") or [],
                    graph_ctx.get("viewpoint") or {},
                    float(graph_ctx.get("view_yaw") or 0.0),
                    str(graph_ctx.get("passage_state") or "open"),
                    graph_ctx.get("report") or {},
                    h,
                )
            )
        else:
            graph = build_graph(event, graph_ctx["tracked"], graph_ctx["room_center"])
            graph_bgr = render_graph_panel(
                graph,
                graph_ctx["layout"],
                h,
                hazard_edges=graph_ctx.get("hazard_edges"),
                changed_ids=graph_ctx.get("changed_ids"),
                hot_ids=graph_ctx.get("hot_ids"),
            )
            panels.append(graph_bgr)

    if len(panels) == 1:
        combined = panels[0]
    else:
        divider = np.full((h, 4, 3), (60, 60, 60), dtype=np.uint8)
        combined = panels[0]
        for panel in panels[1:]:
            combined = np.hstack([combined, divider, panel])
    return _draw_labels(combined, lines)


def run_scene(
    controller,
    name: str,
    config: HazardConfig,
    *,
    label: str | None = None,
    physics_steps_final: int = 0,
    fps_override: int | None = None,
) -> dict[str, Any]:
    demo_name = f"scene_{name}_{label}" if label else f"scene_{name}"
    paths = hazard_artifact_paths(config.scene, demo_name)
    video_path = paths["video"]
    video_path.parent.mkdir(parents=True, exist_ok=True)

    profile = capture_profile(name, fps_override)
    fps = int(profile["fps"])
    substeps = int(profile["substeps"])
    hold_paused = bool(profile["hold_paused"])
    time_step = float(profile["time_step"])

    controller.reset(config.scene)
    has_overhead = add_overhead_camera(controller)

    hazard = build_hazard(name, config)
    setup_info = hazard.setup(controller)
    if hasattr(hazard, "thermal_delta_time"):
        hazard.thermal_delta_time = time_step

    map_props_event = controller.step(action="GetMapViewCameraProperties")
    map_props = map_props_event.metadata.get("actionReturn")
    room_center = room_center_from_map_props(map_props if map_props else None)
    if room_center == (0.0, 0.0):
        room_center = room_center_from_objects(controller.last_event)
    tracked = select_tracked_nodes(controller.last_event, name)
    graph_layout = compute_layout(tracked, controller.last_event) if tracked else {}
    parent_baseline = parent_snapshot(controller.last_event, tracked) if tracked else {}
    graph_ctx: dict[str, Any] = {
        "hazard": name,
        "tracked": tracked,
        "layout": graph_layout,
        "room_center": room_center,
        "parent_baseline": parent_baseline,
        "hazard_edges": [],
        "changed_ids": set(),
        "hot_ids": set(),
        "passage_state": "open",
        "placed_ids": [],
        "report": {},
    }
    if name == "obstruction":
        graph_ctx["blocker_ids"] = list(setup_info.get("blocker_ids") or [])
        graph_ctx["viewpoint"] = dict(setup_info.get("viewpoint") or {})
        graph_ctx["view_yaw"] = float(setup_info.get("view_yaw") or 0.0)

    map_props_dict = map_props if isinstance(map_props, dict) else None
    initial_scene_graph = build_initial_scene_graph(
        config.scene,
        controller.last_event,
        hazard=name,
        setup_info=setup_info,
        map_props=map_props_dict,
    )
    thor_id_map = _thor_id_to_node_id(initial_scene_graph, list(controller.last_event.metadata.get("objects") or []))
    current_scene_graph = initial_scene_graph
    timeline: list[dict[str, Any]] = []

    writer: cv2.VideoWriter | None = None
    frames_written = 0
    trace: list[dict[str, Any]] = []
    heat_overlay: dict[str, Any] | None = None
    ignition_c = config.smoke.ignition_temperature_c if name == "smoke" else 180.0
    use_heat_panel = name == "smoke" and config.native_effects
    substep_fn = getattr(hazard, "substep", None)

    # For the earthquake regime we pause physics once and never unpause between
    # ticks, so no simulation time passes un-rendered (no teleporting objects).
    if hold_paused:
        pause_physics(controller)

    for step in range(hazard.total_steps()):
        report = hazard.tick(controller, step)
        density = float(getattr(hazard, "render_density", 0.0))
        shift = tuple(getattr(hazard, "render_shift", (0, 0)))

        parents_now = parent_snapshot(controller.last_event, tracked) if tracked else {}
        changed_ids: set[str] = set()
        for nid, before in parent_baseline.items():
            if before != parents_now.get(nid):
                changed_ids.add(nid)
        for item in _as_dict_list(report.get("newly_broken")):
            oid = item.get("objectId")
            if oid:
                changed_ids.add(oid)
        graph_ctx["changed_ids"] = changed_ids
        graph_ctx["report"] = report
        if name == "obstruction":
            graph_ctx["passage_state"] = str(report.get("edge_state") or "open")
            graph_ctx["placed_ids"] = list(report.get("placed_ids") or [])

        if not hold_paused:
            pause_physics(controller)
        for i in range(substeps):
            frac = (i + 1) / substeps
            if substep_fn is not None:
                substep_fn(controller, time_step, frac)
                density = float(getattr(hazard, "render_density", 0.0))
                live_stats = {**report, **getattr(hazard, "_substep_stats", {})}
                lines = _label_lines(name, step, live_stats)
                if use_heat_panel:
                    agent_pos = controller.last_event.metadata["agent"]["position"]
                    heat_field = getattr(hazard, "heat_field", None)
                    heat_overlay = {
                        "field": heat_field,
                        "ignition_c": ignition_c,
                        "agent_pos": agent_pos,
                        "stats": live_stats if heat_field is not None else None,
                    }
            else:
                lines = _label_lines(name, step, report)

            event = advance_physics(controller, time_step)
            frame_parents = parent_snapshot(event, tracked) if tracked else {}
            frame_changed: set[str] = set(changed_ids)
            for nid, before in parent_baseline.items():
                if before != frame_parents.get(nid):
                    frame_changed.add(nid)
            graph_ctx["changed_ids"] = frame_changed
            if tracked:
                graph_ctx["hazard_edges"] = hazard_graph_edges(
                    name,
                    event,
                    tracked,
                    report,
                    heat_field=getattr(hazard, "heat_field", None),
                    hot_threshold_c=config.thermal.hot_threshold_c,
                    parent_snapshot_now=frame_parents,
                    parent_baseline=parent_baseline,
                )
            if name == "smoke":
                graph_ctx["hot_ids"] = heat_state_ids(
                    getattr(hazard, "heat_field", None),
                    tracked,
                    threshold_c=config.thermal.hot_threshold_c,
                )
            frame = _compose_frame(
                event, density, lines, fpv_shift=shift,
                heat_overlay=heat_overlay, graph_ctx=graph_ctx,
            )
            if frame is None:
                continue
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            writer.write(frame)
            frames_written += 1
        if not hold_paused:
            unpause_physics(controller)

        trace.append({**report, **getattr(hazard, "_substep_stats", {})})

        prev_scene_graph = current_scene_graph
        current_scene_graph = apply_hazard_obs(
            initial_scene_graph,
            name,
            controller.last_event,
            report,
            step,
            heat_field=getattr(hazard, "heat_field", None),
            hot_threshold_c=config.thermal.hot_threshold_c,
            parent_baseline=parent_baseline,
            tracked_ids=tracked,
            setup_info=setup_info,
            thor_id_map=thor_id_map,
        )
        timeline.append(diff_scene_graph(prev_scene_graph, current_scene_graph, step))

    final_info = hazard.finalize(controller)

    # a short settling tail so the final state is visible
    density = float(getattr(hazard, "render_density", 0.0))
    tail_lines = _label_lines(name, hazard.total_steps() - 1, trace[-1] if trace else {})
    for _ in range(max(0, physics_steps_final)):
        event = controller.step(action="Pass")
        frame = _compose_frame(
            event, density, tail_lines,
            heat_overlay=heat_overlay if use_heat_panel else None,
            graph_ctx=graph_ctx,
        )
        if frame is not None and writer is not None:
            writer.write(frame)
            frames_written += 1

    if writer is not None:
        writer.release()

    if not video_path.is_file() or frames_written <= 0:
        raise RuntimeError(f"No frames written for {name} scene at {video_path}")

    finalize_mp4(video_path)
    probe = probe_video(video_path)
    sample_idx = [max(0, int(frames_written * frac) - 1) for frac in (0.25, 0.5, 0.75)]
    samples = extract_sample_frames(video_path, paths["frames_dir"], sample_idx)

    scenegraph_path, scenegraph_validation = write_scene_graph(
        paths, initial_scene_graph, current_scene_graph, timeline=timeline,
    )

    summary = {
        "hazard": name,
        "scene": config.scene,
        "config": config.to_dict(),
        "has_overhead_camera": has_overhead,
        "scene_graph_nodes": graph_ctx.get("blocker_ids") if name == "obstruction" else tracked,
        "scene_graph": scene_graph_summary_block(
            scenegraph_path, initial_scene_graph, current_scene_graph, scenegraph_validation,
        ),
        "setup": setup_info,
        "final": final_info,
        "capture_profile": {
            "fps": fps,
            "substeps_per_tick": substeps,
            "hold_paused": hold_paused,
            "time_step": time_step,
        },
        "num_ticks": len(trace),
        "trace_sample": trace[:: max(1, len(trace) // 10)],
        "trace_full": trace,
        "frames_written": frames_written,
        "video": probe,
        "sample_frames": [str(p.relative_to(output_root().parent)) for p in samples],
        "output_mp4": str(video_path.relative_to(output_root().parent)),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 64)
    print(f"HAZARD SCENE: {summary['hazard']} | scene: {summary['scene']}")
    print(f"  Video:  {summary['output_mp4']}")
    print(f"  Frames: {summary['frames_written']}  Probe: {summary['video']}")
    final = summary.get("final", {})
    for key in (
        "passage_blocked", "final_edge_state", "objects_placed",
        "broken_object_ids", "num_state_changes",
        "reachable_before", "reachable_after",
    ):
        if key in final:
            print(f"  {key}: {final[key]}")
    print("=" * 64)


def build_config(name: str, args) -> HazardConfig:
    native = bool(getattr(args, "local_executable", None))
    total_ticks = args.ticks if args.ticks is not None else SCENE_TICK_DEFAULTS.get(name, DEFAULT_TICKS)
    return HazardConfig(
        hazard_type=name,
        scene=args.scene,
        severity=args.severity,
        seed=args.seed,
        onset_step=args.onset,
        total_ticks=total_ticks,
        native_effects=native,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameterized hazard scenes (proposal 3.1.1) with MP4 per family."
    )
    parser.add_argument("hazard", choices=["smoke", "earthquake", "obstruction", "all"])
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--severity", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--onset", type=int, default=3)
    parser.add_argument("--ticks", type=int, default=None,
                        help="Simulation ticks (default: smoke=70, others=40).")
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Override writer FPS for all scenes (default: per-scene profile; "
             "earthquake=30, smoke/obstruction=10).",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--local-executable",
        default=None,
        help="Path to a custom AI2-THOR build (enables native Unity hazard effects).",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    names = ["smoke", "earthquake", "obstruction"] if args.hazard == "all" else [args.hazard]
    controller = None
    summaries: list[dict[str, Any]] = []
    try:
        controller = make_controller(
            args.scene,
            headless=args.headless,
            width=args.width,
            height=args.height,
            render_depth=False,
            local_executable_path=args.local_executable,
        )
        for name in names:
            print(f"\nRunning {name} hazard scene on {args.scene} "
                  f"(severity={args.severity}, seed={args.seed})...")
            config = build_config(name, args)
            tail_frames = SCENE_TAIL_FRAMES.get(name, DEFAULT_TAIL_FRAMES)
            summary = run_scene(
                controller, name, config,
                physics_steps_final=tail_frames, fps_override=args.fps,
            )
            summaries.append(summary)
            print_summary(summary)
    finally:
        if controller is not None:
            controller.stop()

    failed = [
        s for s in summaries
        if (s.get("video") or {}).get("frame_count", 0) <= 0
        or (s.get("video") or {}).get("duration", 0) <= 0
    ]
    if failed:
        print("FAIL: one or more hazard scenes produced empty videos.")
        sys.exit(1)


if __name__ == "__main__":
    main()
