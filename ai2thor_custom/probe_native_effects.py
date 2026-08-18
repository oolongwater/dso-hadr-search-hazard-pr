#!/usr/bin/env python3
"""Smoke-test native hazard actions against a custom local AI2-THOR build."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ai2thor_pipeline"))

from hazard.utils import find_object, find_objects  # noqa: E402
from hazard.functions import (  # noqa: E402
    EarthquakeParams,
    FireParams,
    SmokeParams,
    ThermalParams,
    advance_heat_field,
    configure_thermal,
    set_smoke,
    start_earthquake,
    start_fire,
    stop_earthquake,
    stop_fire,
)
from core.thor import make_controller  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe custom hazard Unity actions.")
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--local-executable", required=True)
    args = parser.parse_args()

    controller = make_controller(
        args.scene,
        local_executable_path=args.local_executable,
        render_depth=False,
    )
    try:
        controller.reset(args.scene)
        checks: list[tuple[str, bool]] = []

        ev = set_smoke(controller, SmokeParams(density=0.45))
        checks.append(("SetSmokeDensity", bool(ev.metadata.get("lastActionSuccess"))))

        target = find_object(ev, object_type="Bottle")
        if target is None:
            broken = find_objects(ev, predicate=lambda o: o.get("breakable"))
            target = broken[0] if broken else None
        if target is not None:
            ev = start_fire(
                controller,
                FireParams(object_id=target["objectId"], severity=0.8),
            )
            checks.append(("StartHazardFire", bool(ev.metadata.get("lastActionSuccess"))))
        else:
            checks.append(("StartHazardFire", False))

        props_event = controller.step(action="GetMapViewCameraProperties")
        props = props_event.metadata.get("actionReturn")
        if props and props_event.metadata.get("lastActionSuccess", False):
            ev = configure_thermal(controller, ThermalParams(), props)
            checks.append(("SetThermalParams", bool(ev.metadata.get("lastActionSuccess"))))
            field = None
            advance_ok = False
            for _ in range(5):
                field = advance_heat_field(controller, 0.2)
                advance_ok = field is not None
            warmed = field is not None and field.max_c > ThermalParams().ambient_c + 5.0
            checks.append(("AdvanceHeatField", advance_ok))
            checks.append(("AdvanceHeatField_warming", warmed))
        else:
            checks.append(("SetThermalParams", False))
            checks.append(("AdvanceHeatField", False))
            checks.append(("AdvanceHeatField_warming", False))

        ev = start_earthquake(controller, EarthquakeParams(magnitude=2.5, frequency_hz=2.0))
        checks.append(("StartEarthquake", bool(ev.metadata.get("lastActionSuccess"))))

        controller.step(action="PausePhysicsAutoSim")
        ev = controller.step(action="AdvancePhysicsStep", timeStep=0.0333333)
        checks.append(("AdvancePhysicsStep", bool(ev.metadata.get("lastActionSuccess"))))
        controller.step(action="UnpausePhysicsAutoSim")

        frame_ok = ev.frame is not None and ev.frame.size > 0
        checks.append(("frame_capture", frame_ok))

        ev = stop_earthquake(controller)
        checks.append(("StopEarthquake", bool(ev.metadata.get("lastActionSuccess"))))

        ev = stop_fire(controller)
        checks.append(("StopHazardFire", bool(ev.metadata.get("lastActionSuccess"))))

        ev = set_smoke(controller, SmokeParams(density=0.0))
        checks.append(("SetSmokeDensity_clear", bool(ev.metadata.get("lastActionSuccess"))))

        print("Native hazard action probe:")
        failed = False
        for name, ok in checks:
            status = "OK" if ok else "FAIL"
            print(f"  {name}: {status}")
            failed = failed or not ok
        if failed:
            sys.exit(1)
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
