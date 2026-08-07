import pytest

from dso_hadr.simulator.control import (
    move_ahead,
    rotate_left,
    rotate_right,
)


@pytest.mark.parametrize(
    ("factory", "value", "expected"),
    [
        (move_ahead, 0.25, {"action": "MoveAhead", "moveMagnitude": 0.25}),
        (rotate_left, 90.0, {"action": "RotateLeft", "degrees": 90.0}),
        (rotate_right, 90.0, {"action": "RotateRight", "degrees": 90.0}),
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
