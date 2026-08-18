#!/usr/bin/env python3
"""Build analytic 3D voxel traversability map for a ProcTHOR house scene."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.procthor_house import (
    default_local_executable,
    load_house_json,
    make_procedural_controller,
)
from core.thor import get_reachable_positions, output_root
from core.volumetric_map import (
    build_volume,
    crosscheck_reachable,
    label_counts,
    render_diagnostic_panel,
    save_volume,
    traversability_counts,
)


def _default_house_path(label: str) -> Path:
    batch = Path(__file__).resolve().parents[1] / "assets" / "houses" / "batch"
    path = batch / f"{label}.json"
    if path.exists():
        return path
    return Path(__file__).resolve().parents[1] / "assets" / "houses" / f"{label}.json"


def _object_inventory(objects: list[dict[str, Any]], meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for obj in objects:
        bbox = obj.get("axisAlignedBoundingBox") or {}
        size = bbox.get("size") or {}
        rows.append(
            {
                "objectType": obj.get("objectType"),
                "objectId": obj.get("objectId"),
                "size": {
                    "x": float(size.get("x", 0.0)),
                    "y": float(size.get("y", 0.0)),
                    "z": float(size.get("z", 0.0)),
                },
                "structural_span": (
                    float(size.get("x", 0.0)) >= meta["width_m"] * 0.80
                    or float(size.get("z", 0.0)) >= meta["depth_m"] * 0.80
                ),
            }
        )
    return sorted(rows, key=lambda r: (str(r["objectType"]), -r["size"]["x"] * r["size"]["z"]))


def run_build(
    *,
    label: str,
    house_path: Path,
    out_dir: Path,
    headless: bool,
    resolution: float,
    local_executable: Path | None,
) -> dict[str, Any]:
    house = load_house_json(house_path)
    controller = make_procedural_controller(
        house,
        headless=headless,
        render_depth=False,
        local_executable_path=str(local_executable) if local_executable else None,
    )
    try:
        objects = list(controller.last_event.metadata.get("objects") or [])
        reachable = get_reachable_positions(controller)

        vol = build_volume(house, objects, resolution=resolution, label=label)
        crosscheck = crosscheck_reachable(vol, reachable)
        inventory = _object_inventory(objects, vol.meta)

        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / f"{label}.npz"
        json_path = out_dir / f"{label}.json"
        png_path = out_dir / f"{label}.png"

        save_volume(vol, npz_path)
        panel = render_diagnostic_panel(vol)
        cv2.imwrite(str(png_path), panel)

        summary = {
            "label": label,
            "house_json": str(house_path.resolve()),
            "meta": vol.meta,
            "label_counts": label_counts(vol),
            "traversability_counts": traversability_counts(vol),
            "reachable_crosscheck": crosscheck,
            "object_inventory_sample": inventory[:40],
            "artifacts": {
                "npz": str(npz_path.resolve()),
                "json": str(json_path.resolve()),
                "png": str(png_path.resolve()),
            },
        }
        json_path.write_text(json.dumps(summary, indent=2))

        print(f"Wrote {npz_path}")
        print(f"Wrote {png_path}")
        print(f"Wrote {json_path}")
        print(
            f"Reachable crosscheck: {crosscheck['walk_match']}/{crosscheck['total']} WALK "
            f"(non_walk_rate={crosscheck['non_walk_rate']})"
        )
        return summary
    finally:
        controller.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build analytic volumetric traversability map")
    parser.add_argument("--label", default="four_room_ring_1f", help="Scene label / batch house filename stem")
    parser.add_argument("--house", type=Path, default=None, help="Override house JSON path")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--resolution", type=float, default=0.10, help="Voxel size in metres")
    parser.add_argument("--headless", action="store_true", help="Run AI2-THOR headless")
    parser.add_argument(
        "--local-executable",
        type=Path,
        default=None,
        help="Custom AI2-THOR Unity build (default: ~/ai2thor-src-full/...)",
    )
    args = parser.parse_args()

    house_path = args.house or _default_house_path(args.label)
    if not house_path.exists():
        raise SystemExit(f"House JSON not found: {house_path}")

    out_dir = args.out_dir or (output_root() / "volumetric")
    exe = args.local_executable or default_local_executable()

    run_build(
        label=args.label,
        house_path=house_path,
        out_dir=out_dir,
        headless=args.headless,
        resolution=args.resolution,
        local_executable=exe,
    )


if __name__ == "__main__":
    main()
