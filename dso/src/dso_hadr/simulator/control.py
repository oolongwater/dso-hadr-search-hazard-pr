"""Primitive control actions accepted by the simulator interface."""

from __future__ import annotations


def move_ahead(magnitude: float, target_y: float | None = None) -> dict[str, object]:
    action: dict[str, object] = {"action": "MoveAhead", "moveMagnitude": magnitude}
    if target_y is not None:
        action["targetY"] = target_y
    return action


def rotate_left(degrees: float) -> dict[str, object]:
    return {"action": "RotateLeft", "degrees": degrees}


def rotate_right(degrees: float) -> dict[str, object]:
    return {"action": "RotateRight", "degrees": degrees}
