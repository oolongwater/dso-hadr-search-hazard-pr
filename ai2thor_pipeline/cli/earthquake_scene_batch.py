#!/usr/bin/env python3
"""Generate earthquake videos for ProcTHOR houses (generated or corpus JSON)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.hazard_scenes import SCENE_CAPTURE_PROFILES, _compose_frame, _label_lines
from core.procthor_house import (
    default_local_executable,
    house_floors,
    house_schema,
    load_house_json,
    make_procedural_controller,
    rank_house_json_paths,
    reachable_on_floor,
)
from core.thor import get_reachable_positions
from core.video import finalize_mp4, probe_video
from hazard.functions import EarthquakeLatents, HazardConfig
from hazard.model import EarthquakeHazard
from hazard.utils import advance_physics, hazard_output_dir, pause_physics, unpause_physics

TAIL_FRAMES = 16
FALL_DELTA_M = 0.25
SETTLE_PHYSICS_STEPS = 24
SETTLE_PHYSICS_DT = 1.0 / 30.0
MAX_GEN_ATTEMPTS = 25
BATCH_HOUSE_DIR = Path(__file__).resolve().parents[1] / "assets" / "houses" / "batch"
CORPUS_SEVERITY = 0.85
EXPECTED_TOP10 = (
    "scene_0064",
    "scene_0079",
    "scene_0046",
    "scene_0085",
    "scene_0044",
    "scene_0025",
    "scene_0006",
    "scene_0053",
    "scene_0047",
    "scene_0062",
)


@dataclass(frozen=True)
class SceneSpec:
    label: str
    seed: int
    severity: float
    room_spec: str | None = None
    num_floors: int | None = None
    house_json: Path | None = None
    rank_metrics: dict[str, Any] | None = None


SCENE_MATRIX: tuple[SceneSpec, ...] = (
    SceneSpec("kitchen_living_1f", seed=11, severity=0.55, room_spec="kitchen-living-room"),
    SceneSpec("four_room_1f", seed=23, severity=0.70, room_spec="4-room"),
    SceneSpec("two_bed_bath_1f", seed=37, severity=0.85, room_spec="2-bed-1-bath"),
    SceneSpec("residential_2f", seed=53, severity=0.75, num_floors=2),
    SceneSpec("residential_3f", seed=71, severity=1.0, num_floors=3),
)


def add_overhead_camera_at_floor(controller, base_y: float) -> bool:
    props_event = controller.step(action="GetMapViewCameraProperties")
    props = props_event.metadata.get("actionReturn")
    if not props or not props_event.metadata.get("lastActionSuccess", False):
        return False
    position = dict(props["position"])
    position["y"] = float(base_y) + float(position.get("y", 8.0))
    event = controller.step(
        action="AddThirdPartyCamera",
        position=position,
        rotation=props["rotation"],
        orthographic=props.get("orthographic", True),
        orthographicSize=props.get("orthographicSize", 3.0),
    )
    return bool(event.metadata.get("lastActionSuccess", False))


def generate_or_load_house(
    spec: SceneSpec,
    *,
    local_executable: Path,
    force_regen: bool = False,
) -> dict[str, Any]:
    """Return house_json; always simulates on a fresh controller after caching."""
    if spec.house_json is not None:
        return load_house_json(spec.house_json)

    from procthor.generation import HouseGenerator
    from procthor.utils.types import InvalidFloorplan, InvalidMultiFloorPlan

    out_path = BATCH_HOUSE_DIR / f"{spec.label}.json"
    if out_path.is_file() and not force_regen:
        return load_house_json(out_path)

    controller = make_procedural_controller(
        None,
        headless=False,
        local_executable_path=str(local_executable),
    )
    house_obj = None
    try:
        for attempt in range(MAX_GEN_ATTEMPTS):
            gen_seed = spec.seed + attempt * 7919
            if spec.num_floors is not None:
                from procthor.generation.multifloor_generation import ensure_schema2_controller

                ensure_schema2_controller(controller)
                gen = HouseGenerator(
                    split="train",
                    seed=gen_seed,
                    num_floors=spec.num_floors,
                    controller=controller,
                )
            else:
                gen = HouseGenerator(
                    split="train",
                    seed=gen_seed,
                    room_spec=spec.room_spec,
                    controller=controller,
                )

            try:
                house_obj, _ = gen.sample()
            except (InvalidFloorplan, InvalidMultiFloorPlan):
                continue
            house_obj.validate(controller)
            warns = house_obj.data.get("metadata", {}).get("warnings") or {}
            if warns:
                hard_fail = any(
                    key in warns
                    for key in ("CreateHouse", "TeleportFull")
                ) or any(
                    "Failed to create house" in str(msg)
                    for msg in warns.values()
                )
                if hard_fail:
                    continue
                if spec.num_floors is None and warns:
                    continue
            break
        else:
            raise RuntimeError(
                f"failed to generate valid house for {spec.label} after {MAX_GEN_ATTEMPTS} attempts"
            )
    except Exception:
        controller.stop()
        raise

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(house_obj.data, indent=2), encoding="utf-8")
    controller.stop()
    return house_obj.data


def _ground_floor_base_y(house: dict[str, Any]) -> float:
    floors = house_floors(house)
    ground = next((f for f in floors if int(f.get("index", 0)) == 0), floors[0])
    return float(ground.get("baseY", 0.0))


def _objects_fallen(final_info: dict[str, Any], *, min_drop_m: float = FALL_DELTA_M) -> list[str]:
    baseline = final_info.get("baseline") or {}
    after = final_info.get("after") or {}
    fallen: list[str] = []
    for oid, base in baseline.items():
        end = after.get(oid)
        if end is None:
            continue
        drop = float(base.get("y", 0.0)) - float(end.get("y", 0.0))
        if drop >= min_drop_m:
            fallen.append(str(oid))
    return fallen


def _max_fall_delta(final_info: dict[str, Any]) -> float:
    baseline = final_info.get("baseline") or {}
    after = final_info.get("after") or {}
    max_drop = 0.0
    for oid, base in baseline.items():
        end = after.get(oid)
        if end is None:
            continue
        drop = float(base.get("y", 0.0)) - float(end.get("y", 0.0))
        max_drop = max(max_drop, drop)
    return max_drop


def _settle_scene_physics(controller) -> None:
    unpause_physics(controller)
    for _ in range(SETTLE_PHYSICS_STEPS):
        advance_physics(controller, SETTLE_PHYSICS_DT)
    pause_physics(controller)


def run_earthquake_scene(
    controller,
    config: HazardConfig,
    *,
    house: dict[str, Any],
    out_video: Path,
) -> dict[str, Any]:
    profile = dict(SCENE_CAPTURE_PROFILES["earthquake"])
    fps = int(profile["fps"])
    substeps = int(profile["substeps"])
    hold_paused = bool(profile["hold_paused"])
    time_step = float(profile["time_step"])

    base_y = _ground_floor_base_y(house)
    _settle_scene_physics(controller)
    add_overhead_camera_at_floor(controller, base_y)
    hazard = EarthquakeHazard(config)
    ground_y = _ground_floor_base_y(house)
    reachable = get_reachable_positions(controller)
    if house_schema(house) == "2.0.0":
        reachable = reachable_on_floor(reachable, ground_y)
    setup_info = hazard.setup(
        controller,
        reachable=reachable,
        floor_base_y=ground_y,
    )

    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    trace: list[dict[str, Any]] = []
    peak_moving = 0

    if hold_paused:
        pause_physics(controller)

    for step in range(hazard.total_steps()):
        report = hazard.tick(controller, step)
        shift = tuple(getattr(hazard, "render_shift", (0, 0)))
        peak_moving = max(peak_moving, int(report.get("num_moving") or 0))

        for _ in range(substeps):
            lines = _label_lines("earthquake", step, report)
            event = advance_physics(controller, time_step)
            frame = _compose_frame(event, 0.0, lines, fpv_shift=shift, graph_ctx=None)
            if frame is None:
                continue
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fh, fw = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_video), fourcc, fps, (fw, fh))
            writer.write(frame)
            frames_written += 1

        trace.append(report)

    final_info = hazard.finalize(controller)
    tail_report = trace[-1] if trace else {}
    tail_lines = _label_lines("earthquake", hazard.total_steps() - 1, tail_report)
    for _ in range(TAIL_FRAMES):
        event = controller.step(action="Pass")
        frame = _compose_frame(event, 0.0, tail_lines, graph_ctx=None)
        if frame is not None and writer is not None:
            writer.write(frame)
            frames_written += 1

    if writer is not None:
        writer.release()
    if not out_video.is_file() or frames_written <= 0:
        raise RuntimeError(f"no frames written to {out_video}")
    finalize_mp4(out_video)

    final_info["baseline"] = hazard.baseline
    from hazard.utils import object_state_snapshot

    final_info["after"] = object_state_snapshot(controller.last_event)

    return {
        "frames_written": frames_written,
        "trace": trace,
        "final_info": final_info,
        "setup_info": setup_info,
        "visible_fallable_at_viewpoint": int(setup_info.get("visible_fallable_at_viewpoint") or 0),
        "visible_fallable_ids": list(setup_info.get("visible_fallable_ids") or []),
        "peak_num_moving": peak_moving,
        "video_path": str(out_video),
        "house_schema": house_schema(house),
        "floor_count": len(house_floors(house)),
        "room_count": len(house.get("rooms") or []),
    }


def self_check(result: dict[str, Any]) -> None:
    video_path = Path(result["video_path"])
    probe = probe_video(video_path)
    assert result["frames_written"] > 0, "no frames written"
    assert probe.get("frame_count", 0) > 0, f"unreadable mp4: {video_path}"
    assert result["peak_num_moving"] > 0, "quake must move objects"
    final_info = result["final_info"]
    assert final_info.get("num_state_changes", 0) > 0, "need durable state changes"

    visible_fallable = int(result.get("visible_fallable_at_viewpoint") or 0)
    assert visible_fallable >= 3, f"need >=3 visible fallable objects (got {visible_fallable})"

    fallen_ids = set(_objects_fallen(final_info))
    assert len(fallen_ids) >= 5, f"need >=5 fallen objects (got {len(fallen_ids)})"

    visible_at_start = set(result.get("visible_fallable_ids") or [])
    fallen_and_visible = fallen_ids & visible_at_start
    assert len(fallen_and_visible) >= 2, (
        f"need >=2 fallen objects visible from viewpoint at t=0 (got {len(fallen_and_visible)})"
    )


def _scene_seed_from_label(label: str) -> int:
    if label.startswith("scene_") and label[6:].isdigit():
        return int(label[6:])
    return 0


def _corpus_json_paths(scenes_dir: Path) -> list[Path]:
    return sorted(scenes_dir.glob("scene_*.json"))


def build_corpus_specs(
    scenes_dir: Path,
    *,
    top: int,
    severity: float,
) -> tuple[SceneSpec, ...]:
    paths = _corpus_json_paths(scenes_dir)
    if not paths:
        raise RuntimeError(f"no scene_*.json files in {scenes_dir}")
    ranked = rank_house_json_paths(paths, top=top)
    specs: list[SceneSpec] = []
    for path, metrics in ranked:
        label = path.stem
        specs.append(
            SceneSpec(
                label=label,
                seed=_scene_seed_from_label(label),
                severity=severity,
                house_json=path,
                rank_metrics=metrics,
            )
        )
    return tuple(specs)


def rank_corpus(scenes_dir: Path, *, top: int) -> list[tuple[Path, dict[str, Any]]]:
    paths = _corpus_json_paths(scenes_dir)
    if len(paths) != 100:
        raise RuntimeError(f"expected 100 scene JSONs, found {len(paths)} in {scenes_dir}")
    ranked = rank_house_json_paths(paths, top=top)
    for _, metrics in ranked:
        assert metrics["cone"] >= 3, f"selected scene cone {metrics['cone']} < 3"
    multifloor = [
        m for _, m in rank_house_json_paths(paths, top=len(paths))
        if m["all_fallable"] > m["ground_fallable"]
    ]
    assert multifloor, "ground-floor filter never differs from whole-house count"
    labels = [path.stem for path, _ in ranked]
    if labels != list(EXPECTED_TOP10):
        raise RuntimeError(f"top-{top} mismatch: got {labels}, expected {list(EXPECTED_TOP10)}")
    return ranked


def print_rank_table(ranked: list[tuple[Path, dict[str, Any]]]) -> None:
    print(f"{'rank':>4}  {'scene':11s}  {'score':>6}  {'cone':>4}  {'g_fall':>6}")
    for i, (path, m) in enumerate(ranked, 1):
        print(
            f"{i:4d}  {path.stem:11s}  {m['score']:6.3f}  {m['cone']:4d}  "
            f"{m['ground_fallable']:6d}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--force-regen", action="store_true", help="Regenerate cached house JSON")
    parser.add_argument("--onset", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=55)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--local-executable",
        type=Path,
        default=default_local_executable(),
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Run subset of scene labels (default: all 5)",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        help="Directory of scene_*.json corpus files (enables corpus mode)",
    )
    parser.add_argument("--top", type=int, default=10, help="Top-N scenes from --scenes-dir")
    parser.add_argument(
        "--severity",
        type=float,
        default=CORPUS_SEVERITY,
        help=f"Earthquake severity for corpus mode (default: {CORPUS_SEVERITY})",
    )
    parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Score corpus and print top-N without launching Unity",
    )
    args = parser.parse_args()

    if args.scenes_dir is not None:
        if args.rank_only:
            ranked = rank_corpus(args.scenes_dir, top=args.top)
            print_rank_table(ranked)
            print(f"ranked {len(_corpus_json_paths(args.scenes_dir))} scenes, selected top {args.top}")
            return 0
        specs = build_corpus_specs(args.scenes_dir, top=args.top, severity=args.severity)
        if args.only:
            wanted = set(args.only)
            specs = tuple(s for s in specs if s.label in wanted)
            if not specs:
                raise RuntimeError(f"no matching scenes in --only {args.only}")
        out_dir = hazard_output_dir() / "earthquake" / "corpus"
    else:
        if args.rank_only:
            raise RuntimeError("--rank-only requires --scenes-dir")
        specs = SCENE_MATRIX
        if args.only:
            wanted = set(args.only)
            specs = tuple(s for s in SCENE_MATRIX if s.label in wanted)
            if not specs:
                raise RuntimeError(f"no matching scenes in --only {args.only}")
        out_dir = hazard_output_dir() / "earthquake" / "batch"

    if not args.local_executable.is_file():
        raise RuntimeError(
            f"custom executable not found: {args.local_executable}\n"
            "Run: ./ai2thor_custom/build_local.sh"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for spec in specs:
        print(f"=== {spec.label} (seed={spec.seed}, severity={spec.severity}) ===")
        house = generate_or_load_house(
            spec,
            local_executable=args.local_executable,
            force_regen=args.force_regen,
        )
        config = HazardConfig(
            hazard_type="earthquake",
            scene=spec.label,
            severity=spec.severity,
            seed=spec.seed,
            onset_step=args.onset,
            total_ticks=args.ticks,
            native_effects=True,
            earthquake=EarthquakeLatents(
                impulse_base_newtons=110.0,
                shake_period_ticks=3,
                impulse_scale=0.45,
                integrity_threshold=2.5,
            ),
        )
        out_video = out_dir / f"{spec.label}.mp4"
        controller = make_procedural_controller(
            house,
            headless=args.headless,
            width=args.width,
            height=args.height,
            render_depth=False,
            local_executable_path=str(args.local_executable),
        )
        try:
            result = run_earthquake_scene(
                controller,
                config,
                house=house,
                out_video=out_video,
            )
            self_check(result)
        finally:
            controller.stop()

        fallen = _objects_fallen(result["final_info"])
        house_json_path = (
            str(spec.house_json)
            if spec.house_json is not None
            else str(BATCH_HOUSE_DIR / f"{spec.label}.json")
        )
        summary = {
            "label": spec.label,
            "seed": spec.seed,
            "severity": spec.severity,
            "room_spec": spec.room_spec,
            "num_floors": spec.num_floors,
            "house_schema": result["house_schema"],
            "floor_count": result["floor_count"],
            "room_count": result["room_count"],
            "peak_num_moving": result["peak_num_moving"],
            "objects_fallen": len(fallen),
            "visible_fallable_at_viewpoint": result.get("visible_fallable_at_viewpoint"),
            "fallen_visible_from_viewpoint": len(set(fallen) & set(result.get("visible_fallable_ids") or [])),
            "max_fall_m": round(_max_fall_delta(result["final_info"]), 3),
            "num_state_changes": result["final_info"].get("num_state_changes"),
            "video": str(out_video),
            "house_json": house_json_path,
        }
        if spec.rank_metrics is not None:
            summary["rank_metrics"] = spec.rank_metrics
        summary_path = out_dir / f"{spec.label}.summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
        print(
            f"wrote {out_video} "
            f"(moving={result['peak_num_moving']} fallen={len(fallen)} "
            f"visible_fallable={result.get('visible_fallable_at_viewpoint')})"
        )

    index_path = out_dir / "batch_index.json"
    index_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {index_path} ({len(summaries)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
