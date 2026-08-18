"""Shared hazard scene configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .earthquake import EarthquakeLatents
from .fire_smoke import SmokeLatents
from .obstruction import ObstructionLatents
from .thermal import ThermalParams


@dataclass
class HazardConfig:
    """Latent-variable configuration shared across hazard families."""

    hazard_type: str
    scene: str = "FloorPlan1"
    onset_step: int = 3
    severity: float = 0.7
    total_ticks: int = 40
    seed: int = 0
    native_effects: bool = False
    smoke: SmokeLatents = field(default_factory=SmokeLatents)
    earthquake: EarthquakeLatents = field(default_factory=EarthquakeLatents)
    obstruction: ObstructionLatents = field(default_factory=ObstructionLatents)
    thermal: ThermalParams = field(default_factory=ThermalParams)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
