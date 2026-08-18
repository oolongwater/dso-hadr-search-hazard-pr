"""Earthquake action params and high-level AI2-THOR wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EarthquakeParams:
    """Immediate Unity earthquake action parameters."""

    magnitude: float = 2.5
    frequency_hz: float = 2.0


@dataclass
class EarthquakeLatents:
    """Simulation latents for progressive earthquake hazard scenes."""

    impulse_base_newtons: float = 70.0
    integrity_threshold: float = 4.0
    shake_period_ticks: int = 6
    shake_axis_deg: float = 90.0
    impulse_scale: float = 0.16
    shake_pixels: int = 7


def start_earthquake(controller, params: EarthquakeParams | None = None) -> Any:
    """Start native Unity earthquake shake (custom build required)."""
    p = params or EarthquakeParams()
    return controller.step(
        action="StartEarthquake",
        magnitude=p.magnitude,
        frequencyHz=p.frequency_hz,
    )


def stop_earthquake(controller) -> Any:
    """Stop native Unity earthquake shake."""
    return controller.step(action="StopEarthquake")


PGA_BASE_G = 0.45
PGA_SEV_GAIN_G = 0.50
FREQ_BASE_HZ = 2.0
FREQ_SEV_GAIN_HZ = 1.0


def earthquake_params_from_latents(
    severity: float,
    latents: EarthquakeLatents,
) -> EarthquakeParams:
    """Map scene latents + severity to Unity StartEarthquake kwargs.

    AxisWave in Unity hits near-peak every cycle; real accelerograms peak briefly,
    so this nominal PGA is already more punishing than the same g on a record.
    """
    del latents  # severity drives PGA directly; latents kept for API compat
    sev = max(0.0, min(1.0, float(severity)))
    pga = 9.81 * (PGA_BASE_G + PGA_SEV_GAIN_G * sev)
    frequency_hz = FREQ_BASE_HZ + FREQ_SEV_GAIN_HZ * sev
    return EarthquakeParams(magnitude=pga, frequency_hz=frequency_hz)
