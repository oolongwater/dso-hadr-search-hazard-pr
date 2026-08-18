#!/usr/bin/env python3
"""Validate a changepoints.json file (stdlib only)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from changepoint_kit.changepoint import Changepoint, ChangepointLog, SCHEMA_VERSION, load_changepoints
from changepoint_kit.thor import objects_near

DEFAULT = Path(__file__).with_name("example.changepoints.json")


class _StubEvent:
    def __init__(self, objects: list[dict]) -> None:
        self.metadata = {"objects": objects}


def _check_objects_near() -> None:
    event = _StubEvent(
        [
            {
                "objectId": "a|1",
                "objectType": "Chair",
                "pickupable": True,
                "position": {"x": 1.0, "z": 1.0},
            },
            {
                "objectId": "b|2",
                "objectType": "FloorLamp",
                "pickupable": True,
                "position": {"x": 5.0, "z": 5.0},
            },
            {
                "objectId": "c|3",
                "objectType": "BrokenThing",
                "pickupable": True,
                "isBroken": True,
                "position": {"x": 1.1, "z": 1.0},
            },
            {
                "objectId": "d|4",
                "objectType": "Wall",
                "pickupable": False,
                "moveable": False,
                "position": {"x": 1.0, "z": 1.0},
            },
        ]
    )
    ids, types = objects_near(event, 1.0, 1.0, 1.5)
    assert ids == ["a|1"], f"expected only nearby movable, got {ids}"
    assert types == ["Chair"], f"expected Chair type, got {types}"


def _check_log_open_append() -> None:
    import tempfile

    src = load_changepoints(DEFAULT)
    assert len(src) == 1
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.changepoints.json"
        log = ChangepointLog.open(DEFAULT)
        log.path = path
        log.flush()
        assert len(load_changepoints(path)) == 1
        extra = Changepoint(id="cp_manual_0", source="manual", decision="proceed")
        log.append(extra)
        loaded = load_changepoints(path)
        assert len(loaded) == 2, f"append truncated log, count={len(loaded)}"
        assert loaded[0].id == src[0].id
        assert loaded[1].id == "cp_manual_0"


def validate(path: Path) -> None:
    cps = load_changepoints(path)
    assert cps, f"no changepoints in {path}"
    for i, cp in enumerate(cps):
        assert cp.id, f"record {i} missing id"
        assert cp.visit_index == i, f"{cp.id} visit_index {cp.visit_index} != {i}"
        round_trip = Changepoint.from_dict(cp.to_dict())
        assert round_trip.to_dict() == cp.to_dict(), f"round-trip failed for {cp.id}"
        for e in cp.exits:
            assert e.src, f"{cp.id} exit missing src"
            assert e.dst, f"{cp.id} exit missing dst"
        if cp.blocked:
            assert not cp.traversable_exits(), (
                f"{cp.id} blocked=True but has traversable exits"
            )


def main() -> int:
    _check_objects_near()
    _check_log_open_append()
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
    if not path.is_file():
        raise SystemExit(f"changepoints file not found: {path}")
    validate(path)
    cps = load_changepoints(path)
    print(f"ok: schema={SCHEMA_VERSION} count={len(cps)} path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
