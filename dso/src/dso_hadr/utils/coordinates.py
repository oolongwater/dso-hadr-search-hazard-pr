"""Coordinate conversion at the AI2-THOR navigation boundary."""

from __future__ import annotations

import math


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def native_yaw_to_navigation(degrees: float) -> float:
    """Convert Unity yaw to yaw-zero-at-negative-Z with positive left turns."""

    return wrap_angle(math.radians(degrees - 180.0))


def navigation_yaw_to_native(yaw: float) -> float:
    """Convert the shared navigation yaw to Unity degrees."""

    return (math.degrees(yaw) + 180.0) % 360.0


__all__ = [
    "native_yaw_to_navigation",
    "navigation_yaw_to_native",
    "wrap_angle",
]
