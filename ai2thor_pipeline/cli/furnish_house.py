#!/usr/bin/env python3
"""Append topple-prone furniture and falling clutter to a ProcTHOR house JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.furnish import furnish_house
from core.procthor_house import default_local_executable, load_house_json, make_procedural_controller


def _default_house_path(label: str) -> Path:
    batch = Path(__file__).resolve().parents[1] / "assets" / "houses" / "batch"
    path = batch / f"{label}.json"
    if path.exists():
        return path
    return Path(__file__).resolve().parents[1] / "assets" / "houses" / f"{label}.json"


def _validate_moveable(house: dict[str, Any], placed_ids: set[str], *, local_executable: Path) -> None:
    controller = make_procedural_controller(
        house,
        headless=False,
        render_depth=False,
        local_executable_path=str(local_executable),
    )
    try:
        by_id = {
            str(o.get("objectId") or o.get("id") or ""): o
            for o in controller.last_event.metadata.get("objects") or []
        }
        missing: list[str] = []
        not_moveable: list[str] = []
        for oid in sorted(placed_ids):
            obj = by_id.get(oid)
            if obj is None:
                missing.append(oid)
                continue
            if not (obj.get("moveable") or obj.get("pickupable")):
                not_moveable.append(oid)
        if missing:
            raise RuntimeError(f"placed objects missing from scene metadata: {missing}")
        if not_moveable:
            raise RuntimeError(f"placed objects not moveable/pickupable: {not_moveable}")
    finally:
        controller.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Add topple-prone furniture and clutter to a house JSON")
    parser.add_argument("--label", default="four_room_ring_1f", help="Input house label (batch stem)")
    parser.add_argument("--house", type=Path, default=None, help="Override input house JSON")
    parser.add_argument(
        "--out-label",
        default=None,
        help="Output label stem (default: {label}_quake)",
    )
    parser.add_argument("--per-door", type=int, default=2, help="Doorway blockers per side (max 2)")
    parser.add_argument("--interior-per-room", type=int, default=3, help="Interior wall topple items per room")
    parser.add_argument("--floor-clutter", type=int, default=6, help="Floor clutter items per room")
    parser.add_argument("--receptacle-clutter", type=int, default=2, help="Stacked clutter per receptacle host")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true", help="Load output in AI2-THOR and check moveable")
    parser.add_argument("--local-executable", type=Path, default=None)
    args = parser.parse_args()

    in_path = args.house or _default_house_path(args.label)
    if not in_path.exists():
        raise SystemExit(f"House JSON not found: {in_path}")

    out_label = args.out_label or f"{args.label}_quake"
    out_path = in_path.parent / f"{out_label}.json"

    house = load_house_json(in_path)
    furnished, report = furnish_house(
        house,
        per_door=args.per_door,
        interior_per_room=args.interior_per_room,
        floor_clutter_per_room=args.floor_clutter,
        receptacle_clutter=args.receptacle_clutter,
        seed=args.seed,
    )
    out_path.write_text(json.dumps(furnished, indent=1), encoding="utf-8")

    print(f"Wrote {out_path} (+{report['total_placed']} objects)")
    for stage in ("doorway", "interior", "floor_clutter", "receptacle_clutter"):
        sr = report[stage]
        print(f"  {stage}: placed={sr['placed']} skipped={sr['skipped']}")
        for row in sr["objects"][:5]:
            print(
                f"    {row['id']:40s} {row['assetId']:24s} "
                f"x={row['x']:.2f} z={row['z']:.2f}"
            )
        if sr["placed"] > 5:
            print(f"    ... and {sr['placed'] - 5} more")

    if args.validate:
        exe = args.local_executable or default_local_executable()
        if not exe.is_file():
            raise SystemExit(f"Custom executable not found: {exe}")
        placed_ids = {
            str(o["id"])
            for stage in ("doorway", "interior", "floor_clutter", "receptacle_clutter")
            for o in report[stage]["objects"]
        }
        _validate_moveable(furnished, placed_ids, local_executable=exe)
        print(f"Validated {len(placed_ids)} placed objects are moveable/pickupable in AI2-THOR")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
