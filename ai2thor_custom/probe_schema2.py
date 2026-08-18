#!/usr/bin/env python3
"""Verify schema-2 multi-floor CreateHouse on the custom hazard build."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai2thor_pipeline"))

from core.procthor_house import (  # noqa: E402
    default_house_path,
    default_local_executable,
    generate_multifloor_house,
    load_house_json,
    make_procedural_controller,
)
from hazard.functions import EarthquakeParams, start_earthquake, stop_earthquake  # noqa: E402


def _floor_y_clusters(reachable: list[dict[str, float]], *, min_gap: float = 2.0) -> list[float]:
    ys = sorted({round(float(pt.get("y", 0.0)), 2) for pt in reachable})
    clusters: list[float] = []
    for y in ys:
        if not clusters or abs(y - clusters[-1]) >= min_gap:
            clusters.append(y)
    return clusters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--house", type=Path, default=default_house_path())
    parser.add_argument(
        "--local-executable",
        type=Path,
        default=default_local_executable(),
        required=False,
    )
    parser.add_argument("--generate", action="store_true", help="Regenerate house JSON first")
    args = parser.parse_args()

    if not args.local_executable.is_file():
        raise RuntimeError(
            f"custom executable not found: {args.local_executable}\n"
            "Run: ./ai2thor_custom/build_local.sh"
        )

    house_path = args.house
    if args.generate or not house_path.is_file():
        house_path = generate_multifloor_house(
            house_path,
            local_executable_path=str(args.local_executable),
        )

    house = load_house_json(house_path)
    controller = make_procedural_controller(
        house,
        local_executable_path=str(args.local_executable),
    )
    checks: list[tuple[str, bool, str]] = []
    try:
        schema_event = controller.step(action="GetSupportedHouseSchemas")
        schemas = schema_event.metadata.get("actionReturn") or []
        ok = schema_event.metadata.get("lastActionSuccess", False) and set(schemas) >= {
            "1.0.0",
            "2.0.0",
        }
        checks.append(("GetSupportedHouseSchemas", ok, json.dumps(schemas)))

        create_ok = controller.last_event.metadata.get("lastActionSuccess", False)
        checks.append(("CreateHouse", create_ok, house_path.name))

        reach_event = controller.step(action="GetReachablePositions")
        reachable = reach_event.metadata.get("actionReturn") or []
        clusters = _floor_y_clusters(reachable)
        multi_y = len(clusters) >= 2
        checks.append(
            (
                "GetReachablePositions_multi_floor_y",
                multi_y,
                json.dumps(clusters),
            )
        )

        by_floor = defaultdict(int)
        for pt in reachable:
            y = round(float(pt.get("y", 0.0)), 1)
            by_floor[y] += 1
        checks.append(
            (
                "GetReachablePositions_counts",
                len(reachable) > 0,
                json.dumps(dict(sorted(by_floor.items()))),
            )
        )

        eq = start_earthquake(controller, EarthquakeParams(magnitude=2.5, frequency_hz=2.0))
        checks.append(("StartEarthquake", bool(eq.metadata.get("lastActionSuccess", False)), ""))
        stop_earthquake(controller)
    finally:
        controller.stop()

    print("Schema-2 probe:")
    failed = False
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        suffix = f" ({detail})" if detail else ""
        print(f"  {name}: {status}{suffix}")
        failed = failed or not ok
    if failed:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
