#!/usr/bin/env python3
"""Run contrasting hazard parameter variants and stitch comparison videos.

Each hazard family is simulated twice with different latent settings, then the
pair is stacked vertically with labeled banners to demonstrate customizability
of the high-level hazard_functions API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.thor import make_controller, output_root
from core.video import finalize_mp4, probe_video
from hazard.functions import (
    EarthquakeLatents,
    HazardConfig,
    ObstructionLatents,
    SmokeLatents,
)
from hazard.utils import hazard_artifact_paths, hazard_family_dir, hazard_output_dir

_SCENES_SPEC = importlib.util.spec_from_file_location(
    "hazard_scenes", Path(__file__).resolve().parent / "hazard_scenes.py"
)
_hazard_scenes = importlib.util.module_from_spec(_SCENES_SPEC)
assert _SCENES_SPEC.loader is not None
_SCENES_SPEC.loader.exec_module(_hazard_scenes)
run_scene = _hazard_scenes.run_scene
DEFAULT_TAIL_FRAMES = _hazard_scenes.DEFAULT_TAIL_FRAMES

BANNER_HEIGHT = 36
BANNER_BG = (20, 20, 20)
BANNER_FG = (240, 240, 240)

# SmokeFireHazard.finalize() clears native fog and stops flames, so post-finalize
# tail frames would show an identical cleaned-up room for every smoke variant.
# Earthquake and obstruction tails are informative (settled debris / blocker pile).
VARIANT_TAIL_FRAMES = {"smoke": 0}

# (variant_label, ticks, overrides). Overrides may name either a HazardConfig
# field (e.g. severity) or a field on the hazard's latents dataclass.
VARIANTS: dict[str, tuple[tuple[str, int, dict[str, Any]], ...]] = {
    # severity caps both room fill and agent density, so it has to differ between
    # variants for the final fog level to look different rather than just ramp speed.
    "smoke": (
        (
            "smolder",
            40,
            {
                "severity": 0.3,
                "spread_rate": 0.12,
                "room_fill_rate": 0.012,
            },
        ),
        (
            "inferno",
            40,
            {
                "severity": 1.0,
                "spread_rate": 0.35,
                "room_fill_rate": 0.06,
            },
        ),
    ),
    "earthquake": (
        (
            "tremor",
            30,
            {
                "impulse_base_newtons": 35.0,
                "shake_period_ticks": 10,
                "integrity_threshold": 8.0,
            },
        ),
        (
            "violent",
            30,
            {
                "impulse_base_newtons": 140.0,
                "shake_period_ticks": 4,
                "integrity_threshold": 2.0,
            },
        ),
    ),
    "obstruction": (
        ("trickle", 24, {"growth_rate": 0.1}),
        ("rapid", 24, {"growth_rate": 0.5}),
    ),
}


LATENT_CLASSES = {
    "smoke": SmokeLatents,
    "earthquake": EarthquakeLatents,
    "obstruction": ObstructionLatents,
}


def split_overrides(
    hazard: str, overrides: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition overrides into HazardConfig-level and family-latent-level."""
    if hazard not in LATENT_CLASSES:
        raise ValueError(f"Unknown hazard '{hazard}'")
    latent_names = {f.name for f in fields(LATENT_CLASSES[hazard])}
    config_names = {f.name for f in fields(HazardConfig)}
    latent_overrides = {k: v for k, v in overrides.items() if k in latent_names}
    config_overrides = {k: v for k, v in overrides.items() if k not in latent_names}
    unknown = set(config_overrides) - config_names
    if unknown:
        raise ValueError(f"Unknown override(s) for '{hazard}': {sorted(unknown)}")
    return config_overrides, latent_overrides


def build_variant_config(
    hazard: str,
    scene: str,
    ticks: int,
    overrides: dict[str, Any],
    *,
    native_effects: bool,
) -> HazardConfig:
    config_overrides, latent_overrides = split_overrides(hazard, overrides)
    latents = {
        "smoke": SmokeLatents(),
        "earthquake": EarthquakeLatents(),
        "obstruction": ObstructionLatents(),
    }
    latents[hazard] = LATENT_CLASSES[hazard](**latent_overrides)
    return HazardConfig(
        hazard_type=hazard,
        scene=scene,
        total_ticks=ticks,
        native_effects=native_effects,
        **config_overrides,
        **latents,
    )


def format_banner(variant: str, overrides: dict[str, Any]) -> str:
    parts = [variant.upper()] + [f"{k}={v}" for k, v in overrides.items()]
    return "  ".join(parts)


def _draw_banner(frame: np.ndarray, text: str) -> np.ndarray:
    banner = np.full((BANNER_HEIGHT, frame.shape[1], 3), BANNER_BG, dtype=np.uint8)
    cv2.putText(
        banner,
        text,
        (8, BANNER_HEIGHT - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        BANNER_FG,
        1,
        cv2.LINE_AA,
    )
    return np.vstack([banner, frame])


def _read_all_frames(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames in video: {path}")
    return frames, fps


def compose_comparison(
    top_path: Path,
    bottom_path: Path,
    top_banner: str,
    bottom_banner: str,
    output_path: Path,
) -> dict[str, Any]:
    top_frames, fps = _read_all_frames(top_path)
    bottom_frames, _ = _read_all_frames(bottom_path)
    n = max(len(top_frames), len(bottom_frames))
    last_top = top_frames[-1]
    last_bottom = bottom_frames[-1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    written = 0
    for i in range(n):
        top = top_frames[i] if i < len(top_frames) else last_top
        bottom = bottom_frames[i] if i < len(bottom_frames) else last_bottom
        panel_top = _draw_banner(top, top_banner)
        panel_bottom = _draw_banner(bottom, bottom_banner)
        combined = np.vstack([panel_top, panel_bottom])
        if writer is None:
            h, w = combined.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        writer.write(combined)
        written += 1
    if writer is not None:
        writer.release()
    if written <= 0:
        raise RuntimeError(f"No frames written for comparison at {output_path}")
    finalize_mp4(output_path)
    return probe_video(output_path)


def comparison_output_path(hazard: str, scene: str) -> Path:
    family = hazard_family_dir(f"compare_{hazard}")
    return hazard_output_dir() / family / f"compare_{hazard}_{scene}.mp4"


def run_variants_for_hazard(
    controller,
    hazard: str,
    scene: str,
    *,
    native_effects: bool,
) -> dict[str, Any]:
    specs = VARIANTS[hazard]
    summaries: list[dict[str, Any]] = []
    video_paths: list[Path] = []
    banners: list[str] = []

    for variant, ticks, overrides in specs:
        config = build_variant_config(
            hazard, scene, ticks, overrides, native_effects=native_effects
        )
        print(f"  variant {variant}: ticks={ticks} overrides={overrides}")
        summary = run_scene(
            controller,
            hazard,
            config,
            label=variant,
            physics_steps_final=VARIANT_TAIL_FRAMES.get(hazard, DEFAULT_TAIL_FRAMES),
        )
        summaries.append(summary)
        demo_name = f"scene_{hazard}_{variant}"
        video_paths.append(hazard_artifact_paths(scene, demo_name)["video"])
        banners.append(format_banner(variant, overrides))

    compare_path = comparison_output_path(hazard, scene)
    probe = compose_comparison(
        video_paths[0],
        video_paths[1],
        banners[0],
        banners[1],
        compare_path,
    )
    manifest = {
        "hazard": hazard,
        "scene": scene,
        "variants": [s["config"] for s in summaries],
        "variant_videos": [str(p.relative_to(output_root().parent)) for p in video_paths],
        "comparison_video": str(compare_path.relative_to(output_root().parent)),
        "comparison_probe": probe,
    }
    manifest_path = compare_path.with_suffix(".summary.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate side-by-side comparison videos for hazard parameter variants."
    )
    parser.add_argument("hazard", choices=["smoke", "earthquake", "obstruction", "all"])
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--local-executable",
        default=None,
        help="Custom AI2-THOR build (recommended; enables native hazard effects).",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    native = bool(args.local_executable)
    hazards = (
        ["smoke", "earthquake", "obstruction"]
        if args.hazard == "all"
        else [args.hazard]
    )

    controller = make_controller(
        args.scene,
        headless=args.headless,
        width=args.width,
        height=args.height,
        render_depth=False,
        local_executable_path=args.local_executable,
    )
    results: list[dict[str, Any]] = []
    try:
        for hazard in hazards:
            print(f"\n=== {hazard} variants on {args.scene} (native_effects={native}) ===")
            manifest = run_variants_for_hazard(
                controller,
                hazard,
                args.scene,
                native_effects=native,
            )
            results.append(manifest)
            print(f"  Comparison: {manifest['comparison_video']}")
            print(f"  Probe: {manifest['comparison_probe']}")
    finally:
        controller.stop()

    failed = [
        r for r in results
        if (r.get("comparison_probe") or {}).get("frame_count", 0) <= 0
    ]
    if failed:
        print("FAIL: one or more comparison videos are empty.")
        sys.exit(1)


if __name__ == "__main__":
    main()
