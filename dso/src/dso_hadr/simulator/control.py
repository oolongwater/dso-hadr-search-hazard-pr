"""Primitive control actions accepted by the simulator interface."""

from __future__ import annotations


def move_ahead(magnitude: float, target_y: float | None = None) -> dict[str, object]:
    action: dict[str, object] = {"action": "MoveAhead", "moveMagnitude": magnitude}
    if target_y is not None:
        action["targetY"] = target_y
    return action


def move_back(magnitude: float) -> dict[str, object]:
    return {"action": "MoveBack", "moveMagnitude": magnitude}


def move_left(magnitude: float) -> dict[str, object]:
    return {"action": "MoveLeft", "moveMagnitude": magnitude}


def move_right(magnitude: float) -> dict[str, object]:
    return {"action": "MoveRight", "moveMagnitude": magnitude}


def rotate_left(degrees: float) -> dict[str, object]:
    return {"action": "RotateLeft", "degrees": degrees}


def rotate_right(degrees: float) -> dict[str, object]:
    return {"action": "RotateRight", "degrees": degrees}


def look_up(degrees: float) -> dict[str, object]:
    return {"action": "LookUp", "degrees": degrees}


def look_down(degrees: float) -> dict[str, object]:
    return {"action": "LookDown", "degrees": degrees}
