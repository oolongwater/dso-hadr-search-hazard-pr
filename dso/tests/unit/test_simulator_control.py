import pytest

from dso_hadr.simulator.control import (
    look_down,
    look_up,
    move_ahead,
    move_back,
    move_left,
    move_right,
    rotate_left,
    rotate_right,
)


@pytest.mark.parametrize(
    ("factory", "value", "expected"),
    [
        (move_ahead, 0.25, {"action": "MoveAhead", "moveMagnitude": 0.25}),
        (move_back, 0.25, {"action": "MoveBack", "moveMagnitude": 0.25}),
        (move_left, 0.25, {"action": "MoveLeft", "moveMagnitude": 0.25}),
        (move_right, 0.25, {"action": "MoveRight", "moveMagnitude": 0.25}),
        (rotate_left, 90.0, {"action": "RotateLeft", "degrees": 90.0}),
        (rotate_right, 90.0, {"action": "RotateRight", "degrees": 90.0}),
        (look_up, 30.0, {"action": "LookUp", "degrees": 30.0}),
        (look_down, 30.0, {"action": "LookDown", "degrees": 30.0}),
    ],
)
def test_control_action(factory, value: float, expected: dict[str, object]) -> None:
    assert factory(value) == expected


def test_move_ahead_can_include_a_planned_target_height() -> None:
    assert move_ahead(0.5, target_y=3.9) == {
        "action": "MoveAhead",
        "moveMagnitude": 0.5,
        "targetY": 3.9,
    }
