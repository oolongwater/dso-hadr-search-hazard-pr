"""Thermal field configuration and AI2-THOR action wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ThermalParams:
    """Unity heat-field solver parameters (deg C, metres, seconds)."""

    ambient_c: float = 22.0
    flame_core_c: float = 600.0
    diffusivity: float = 0.08
    convection_gain: float = 1.5
    cooling_rate: float = 0.12
    flame_radius_m: float = 1.0
    hot_threshold_c: float = 70.0
    resolution_x: int = 64
    resolution_z: int = 48


@dataclass
class HeatObjectSample:
    object_id: str
    x: float
    z: float
    temperature_c: float
    thor_temperature: str


@dataclass
class HeatFlameSample:
    x: float
    z: float
    severity: float


@dataclass
class HeatField:
    """Snapshot returned by AdvanceHeatField."""

    center_x: float
    center_z: float
    half_extent_x: float
    half_extent_z: float
    nx: int
    nz: int
    ambient_c: float
    max_c: float
    mean_c: float
    temperatures: list[float]
    flames: list[HeatFlameSample] = field(default_factory=list)
    objects: list[HeatObjectSample] = field(default_factory=list)
    delta_time: float = 0.0
    total_flame_severity: float = 0.0


def _parse_heat_field(raw: dict[str, Any]) -> HeatField:
    flames = [
        HeatFlameSample(
            x=float(f["x"]),
            z=float(f["z"]),
            severity=float(f.get("severity", 0.0)),
        )
        for f in (raw.get("flames") or [])
    ]
    objects = [
        HeatObjectSample(
            object_id=str(o["objectId"]),
            x=float(o["x"]),
            z=float(o["z"]),
            temperature_c=float(o.get("temperatureC", raw.get("ambientC", 22.0))),
            thor_temperature=str(o.get("thorTemperature", "RoomTemp")),
        )
        for o in (raw.get("objects") or [])
    ]
    temps = raw.get("temperatures") or []
    return HeatField(
        center_x=float(raw["centerX"]),
        center_z=float(raw["centerZ"]),
        half_extent_x=float(raw["halfExtentX"]),
        half_extent_z=float(raw["halfExtentZ"]),
        nx=int(raw["nx"]),
        nz=int(raw["nz"]),
        ambient_c=float(raw.get("ambientC", 22.0)),
        max_c=float(raw.get("maxC", raw.get("ambientC", 22.0))),
        mean_c=float(raw.get("meanC", raw.get("ambientC", 22.0))),
        temperatures=[float(t) for t in temps],
        flames=flames,
        objects=objects,
        delta_time=float(raw.get("deltaTime", 0.0)),
        total_flame_severity=float(raw.get("totalFlameSeverity", 0.0)),
    )


def map_view_bounds(camera_props: dict[str, Any], *, width: int, height: int) -> dict[str, float]:
    """Derive the thermal grid rectangle from GetMapViewCameraProperties."""
    pos = camera_props["position"]
    ortho = float(camera_props.get("orthographicSize", 3.0))
    aspect = float(width) / float(max(1, height))
    half_extent_z = ortho
    half_extent_x = ortho * aspect
    return {
        "center_x": float(pos["x"]),
        "center_z": float(pos["z"]),
        "half_extent_x": half_extent_x,
        "half_extent_z": half_extent_z,
    }


def configure_thermal(
    controller,
    params: ThermalParams,
    camera_props: dict[str, Any],
    *,
    width: int = 640,
    height: int = 480,
) -> Any:
    """Allocate the Unity heat grid aligned to the overhead map view."""
    bounds = map_view_bounds(camera_props, width=width, height=height)
    return controller.step(
        action="SetThermalParams",
        centerX=bounds["center_x"],
        centerZ=bounds["center_z"],
        halfExtentX=bounds["half_extent_x"],
        halfExtentZ=bounds["half_extent_z"],
        resolutionX=params.resolution_x,
        resolutionZ=params.resolution_z,
        ambientC=params.ambient_c,
        flameCoreC=params.flame_core_c,
        diffusivity=params.diffusivity,
        convectionGain=params.convection_gain,
        coolingRate=params.cooling_rate,
        flameRadiusM=params.flame_radius_m,
        hotThresholdC=params.hot_threshold_c,
    )


def advance_heat_field(controller, delta_time: float) -> HeatField | None:
    """Integrate the heat field for delta_time seconds and return the snapshot."""
    event = controller.step(action="AdvanceHeatField", deltaTime=float(delta_time))
    if not event.metadata.get("lastActionSuccess", False):
        return None
    raw = event.metadata.get("actionReturn")
    if not isinstance(raw, dict):
        return None
    return _parse_heat_field(raw)


def sample_agent_temperature_c(field: HeatField | None, agent_pos: dict[str, float]) -> float:
    """Bilinear sample agent temperature from a heat field snapshot."""
    if field is None or not field.temperatures:
        return 22.0
    x = float(agent_pos["x"])
    z = float(agent_pos["z"])
    norm_x = (x - (field.center_x - field.half_extent_x)) / (2.0 * field.half_extent_x)
    norm_z = (z - (field.center_z - field.half_extent_z)) / (2.0 * field.half_extent_z)
    if norm_x < 0.0 or norm_x > 1.0 or norm_z < 0.0 or norm_z > 1.0:
        return field.ambient_c
    gx = norm_x * (field.nx - 1)
    gz = norm_z * (field.nz - 1)
    ix0 = max(0, min(int(gx), field.nx - 2))
    iz0 = max(0, min(int(gz), field.nz - 2))
    tx = gx - ix0
    tz = gz - iz0
    t00 = field.temperatures[iz0 * field.nx + ix0]
    t10 = field.temperatures[iz0 * field.nx + ix0 + 1]
    t01 = field.temperatures[(iz0 + 1) * field.nx + ix0]
    t11 = field.temperatures[(iz0 + 1) * field.nx + ix0 + 1]
    t0 = t00 * (1.0 - tx) + t10 * tx
    t1 = t01 * (1.0 - tx) + t11 * tx
    return t0 * (1.0 - tz) + t1 * tz


def count_hot_objects(field: HeatField | None, hot_threshold_c: float) -> int:
    if field is None:
        return 0
    return sum(1 for o in field.objects if o.temperature_c >= hot_threshold_c)
