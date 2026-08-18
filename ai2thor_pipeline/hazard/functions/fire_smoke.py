"""Fire and smoke action params and high-level AI2-THOR wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FireParams:
    """Immediate Unity fire action parameters."""

    object_id: str
    severity: float = 0.7


@dataclass(frozen=True)
class SmokeParams:
    """Immediate Unity smoke/fog density (0–1)."""

    density: float = 0.0


@dataclass
class SmokeLatents:
    """Simulation latents for progressive smoke/fire hazard scenes."""

    emission_rate: float = 0.05
    spread_rate: float = 0.12
    fire_spread_rate: float = 0.08
    room_fill_rate: float = 0.025
    ignition_temperature_c: float = 220.0
    access_density_cutoff: float = 0.6
    burn_delay_ticks_min: int = 3
    burn_delay_ticks_max: int = 6
    source_position: dict[str, float] | None = None


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def start_fire(controller, params: FireParams) -> Any:
    """Start native Unity fire at an object (custom build required)."""
    return controller.step(
        action="StartHazardFire",
        objectId=params.object_id,
        moveMagnitude=_clamp01(params.severity),
        forceAction=True,
    )


def stop_fire(controller) -> Any:
    """Stop all native Unity fire effects."""
    return controller.step(action="StopHazardFire")


def set_smoke(controller, params: SmokeParams | float) -> Any:
    """Set scene smoke/fog density (custom build required)."""
    density = _clamp01(params.density if isinstance(params, SmokeParams) else params)
    return controller.step(action="SetSmokeDensity", density=density)
