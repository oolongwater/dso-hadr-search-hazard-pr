"""Obstruction action params and high-level AI2-THOR wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObstructionParams:
    """Stock AI2-THOR placement parameters for progressive obstruction."""

    object_id: str
    position: dict[str, float]


@dataclass
class ObstructionLatents:
    """Simulation latents for progressive obstruction hazard scenes."""

    growth_rate: float = 0.25
    constrained_cost_ratio: float = 1.1


def place_obstruction(controller, params: ObstructionParams) -> Any:
    """Place an object at a world position (stock AI2-THOR action)."""
    return controller.step(
        action="PlaceObjectAtPoint",
        objectId=params.object_id,
        position=params.position,
    )
