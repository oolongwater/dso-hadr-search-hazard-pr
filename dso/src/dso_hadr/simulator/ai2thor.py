"""Minimal AI2-THOR lifecycle for configured ProcTHOR scenes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ai2thor.controller import Controller  # type: ignore[import-not-found]
from ai2thor.server import Event  # type: ignore[import-not-found]


@dataclass(frozen=True)
class SimulatorConfig:
    local_executable_path: Path
    width: int
    height: int
    quality: str
    render_depth: bool
    snap_to_grid: bool


def load_simulator_config(path: Path) -> SimulatorConfig:
    """Load AI2-THOR runtime parameters from JSON."""

    resolved_path = path.expanduser().resolve(strict=True)
    document = json.loads(resolved_path.read_text(encoding="utf-8"))
    return SimulatorConfig(
        local_executable_path=(resolved_path.parent / document["local_executable_path"]).resolve(
            strict=True
        ),
        width=document["width"],
        height=document["height"],
        quality=document["quality"],
        render_depth=document["render_depth"],
        snap_to_grid=document["snap_to_grid"],
    )


class ProcTHORSimulator:
    """Own one AI2-THOR controller and load ProcTHOR JSON scenes."""

    def __init__(self, config: SimulatorConfig) -> None:
        self._controller = Controller(
            scene="Procedural",
            local_executable_path=str(config.local_executable_path),
            width=config.width,
            height=config.height,
            quality=config.quality,
            renderDepthImage=config.render_depth,
            snapToGrid=config.snap_to_grid,
        )

    def load_scene(self, path: Path) -> Event:
        """Create a house and move the agent to the pose stored in its JSON."""

        house = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
        created: Event = self._controller.reset(scene=house)
        if not created:
            raise RuntimeError(created.metadata["errorMessage"])

        agent_pose = house["metadata"]["agent"]
        event: Event = self._controller.step(
            action="TeleportFull",
            raise_for_failure=True,
            **agent_pose,
        )
        return event

    def step(self, action: dict[str, object]) -> Event:
        """Execute one externally supplied AI2-THOR action."""

        event: Event = self._controller.step(action)
        return event

    def close(self) -> None:
        self._controller.stop()

    def __enter__(self) -> ProcTHORSimulator:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
