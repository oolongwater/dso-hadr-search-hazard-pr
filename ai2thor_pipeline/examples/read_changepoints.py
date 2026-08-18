#!/usr/bin/env python3
"""Load a changepoints.json file and print per-node summaries."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import Changepoint, load_changepoints
from core.changepoint import SCHEMA_VERSION
from core.scene_action_map import CLUTTER_RADIUS_M, MIN_CLUSTER_OBJECTS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT = REPO_ROOT / "output_ai2thor/hazards/earthquake/sam/four_room_ring_1f.changepoints.json"


def _round_trip_ok(cp: Changepoint) -> bool:
    return Changepoint.from_dict(cp.to_dict()).to_dict() == cp.to_dict()


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    if not path.is_file():
        raise SystemExit(f"changepoints file not found: {path}")

    cps = load_changepoints(path)
    print(f"{path}: schema={SCHEMA_VERSION} count={len(cps)}")
    for cp in cps:
        print(
            f"  [{cp.visit_index}] {cp.summary()} "
            f"cluster={cp.cluster_size} types={cp.cluster_counts()} blocked={cp.is_blocked()}"
        )
        assert cp.id, "changepoint missing id"
        assert cp.cluster_size >= MIN_CLUSTER_OBJECTS, (
            f"{cp.id} cluster too small ({cp.cluster_size})"
        )
        assert _round_trip_ok(cp), f"round-trip failed for {cp.id}"
        if cp.payload_png:
            assert Path(cp.payload_png).is_file(), f"missing payload: {cp.payload_png}"
        if cp.clip:
            assert Path(cp.clip).is_file(), f"missing clip: {cp.clip}"

    for i, cp in enumerate(cps):
        assert cp.visit_index == i, f"{cp.id} visit_index {cp.visit_index} != {i}"
        if i > 0:
            assert cps[i - 1].visit_index < cp.visit_index

    print(f"ok: {len(cps)} changepoints validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
