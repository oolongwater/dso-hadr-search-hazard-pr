#!/usr/bin/env python3
"""Render standalone canonical scene-graph MP4s from .scenegraph.json artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.thor import output_root
from core.video import finalize_mp4, probe_video
from hazard.utils import hazard_output_dir
from scene_graph.export import replay_timeline
from scene_graph.schema import SceneGraph
from scene_graph.video import (
    build_tick_graphs,
    compute_graph_layout,
    load_scenegraph_payload,
    payload_to_scenegraph,
    render_graph_frame,
)

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


def _summary_path(scenegraph_path: Path) -> Path:
    stem = scenegraph_path.name.replace(".scenegraph.json", "")
    return scenegraph_path.with_name(f"{stem}.summary.json")


def _hazard_video_path(scenegraph_path: Path) -> Path:
    stem = scenegraph_path.name.replace(".scenegraph.json", "")
    return scenegraph_path.with_name(f"{stem}.mp4")


def _output_video_path(scenegraph_path: Path) -> Path:
    stem = scenegraph_path.name.replace(".scenegraph.json", "")
    return scenegraph_path.with_name(f"{stem}.scenegraph.mp4")


def _load_capture_profile(scenegraph_path: Path) -> dict[str, Any]:
    summary_path = _summary_path(scenegraph_path)
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    profile = dict(summary.get("capture_profile") or {})
    profile["frames_written"] = int(summary.get("frames_written") or 0)
    profile["hazard"] = summary.get("hazard")
    profile["scene"] = summary.get("scene")
    profile["num_ticks"] = int(summary.get("num_ticks") or 0)
    return profile


def _expand_tick_graphs(
    tick_graphs: list[SceneGraph],
    *,
    has_timeline: bool,
    num_ticks: int,
) -> list[SceneGraph]:
    if has_timeline:
        return tick_graphs
    if num_ticks <= 0 or len(tick_graphs) != 2:
        return tick_graphs
    initial, final = tick_graphs
    midpoint = max(1, num_ticks // 2)
    expanded = [initial] * midpoint + [final] * (num_ticks - midpoint)
    return expanded


def render_scenegraph_video(
    scenegraph_path: Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps_override: int | None = None,
) -> dict[str, Any]:
    payload = load_scenegraph_payload(scenegraph_path)
    profile = _load_capture_profile(scenegraph_path)
    fps = int(fps_override or profile.get("fps") or 30)
    substeps = int(profile.get("substeps_per_tick") or 1)
    frames_written = int(profile.get("frames_written") or 0)
    num_ticks = int(profile.get("num_ticks") or len(payload.get("timeline") or []))
    hazard = str(profile.get("hazard") or "hazard")
    scene_id = str(payload.get("scene_id") or profile.get("scene") or "scene")

    tick_graphs, has_timeline = build_tick_graphs(payload)
    if not has_timeline:
        print(
            f"  [WARN] {scenegraph_path.name} has no timeline; using initial/final fallback. "
            "Re-run hazard scenes for full evolution."
        )
    tick_graphs = _expand_tick_graphs(tick_graphs, has_timeline=has_timeline, num_ticks=num_ticks)
    if not tick_graphs:
        raise RuntimeError(f"No graph states to render for {scenegraph_path}")

    initial = payload_to_scenegraph(payload, "initial")
    layout = compute_graph_layout(initial)
    out_path = _output_video_path(scenegraph_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    written = 0
    last_frame = None
    total_ticks = len(tick_graphs)

    for tick_idx, sg in enumerate(tick_graphs):
        title = f"{scene_id} | {hazard} | tick {tick_idx + 1}/{total_ticks}"
        frame = render_graph_frame(sg, layout, title=title, width=width, height=height)
        last_frame = frame
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        for _ in range(substeps):
            writer.write(frame)
            written += 1

    if last_frame is not None and frames_written > written:
        assert writer is not None
        for _ in range(frames_written - written):
            writer.write(last_frame)
            written += 1

    if writer is not None:
        writer.release()

    if written <= 0:
        raise RuntimeError(f"No frames written for {out_path}")

    finalize_mp4(out_path)
    probe = probe_video(out_path)
    hazard_probe = probe_video(_hazard_video_path(scenegraph_path))

    if has_timeline and payload.get("final"):
        replayed = replay_timeline(initial, payload["timeline"])
        if replayed:
            final_dump = replayed[-1].model_dump(mode="json")
            expected = payload["final"]
            if final_dump != expected:
                print(f"  [WARN] timeline replay mismatch for {scenegraph_path.name}")

    parity_ok = True
    if frames_written > 0 and written != frames_written:
        parity_ok = False
        print(f"  [WARN] frame count {written} != hazard frames_written {frames_written}")
    hazard_frames = int(hazard_probe.get("frame_count") or 0)
    if hazard_frames > 0 and int(probe.get("frame_count") or written) != hazard_frames:
        parity_ok = False
        print(
            f"  [WARN] graph video frames {probe.get('frame_count')} "
            f"!= hazard video frames {hazard_frames}"
        )

    return {
        "input": str(scenegraph_path),
        "output_mp4": str(out_path.relative_to(output_root().parent)),
        "has_timeline": has_timeline,
        "ticks_rendered": total_ticks,
        "frames_written": written,
        "fps": fps,
        "substeps_per_tick": substeps,
        "parity_ok": parity_ok,
        "video": probe,
        "hazard_video": hazard_probe,
    }


def discover_scenegraphs(root: Path) -> list[Path]:
    return sorted(root.glob("**/*.scenegraph.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render canonical scene-graph MP4s.")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Paths or globs to .scenegraph.json files (omit with --all).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Render all graphs under {hazard_output_dir()}",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=None, help="Override output FPS.")
    args = parser.parse_args()

    paths: list[Path] = []
    if args.all:
        paths.extend(discover_scenegraphs(hazard_output_dir()))
    for item in args.inputs:
        candidate = Path(item)
        if any(ch in item for ch in "*?[]"):
            paths.extend(p.resolve() for p in Path(".").glob(item))
        elif candidate.is_file():
            paths.append(candidate.resolve())
    paths = sorted(set(paths))
    if not paths:
        parser.error("No .scenegraph.json files found.")

    results: list[dict[str, Any]] = []
    failed = False
    for path in paths:
        print(f"Rendering {path} ...")
        try:
            result = render_scenegraph_video(
                path,
                width=args.width,
                height=args.height,
                fps_override=args.fps,
            )
            results.append(result)
            print(
                f"  -> {result['output_mp4']}  frames={result['frames_written']}  "
                f"timeline={'yes' if result['has_timeline'] else 'fallback'}"
            )
            if not result["parity_ok"]:
                failed = True
        except Exception as exc:
            failed = True
            print(f"  FAIL: {exc}")

    summary_path = hazard_output_dir() / "scenegraph_videos.summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
