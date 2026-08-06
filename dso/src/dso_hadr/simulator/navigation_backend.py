"""Simulator-neutral navigation backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dso_hadr.types.navigation import NavigationAction, NavMesh, Observation, Pose


class NavigationBackend(ABC):
    """Operations required by symbolic-path execution."""

    @property
    @abstractmethod
    def navmesh(self) -> NavMesh:
        """Return the runtime navmesh exported for the active agent."""

    @abstractmethod
    def reset(
        self,
        scene_id: str,
        start_pose: Pose | None,
        seed: int | None = None,
    ) -> Observation:
        """Load a scene and place the agent."""

    @abstractmethod
    def step(self, action: NavigationAction) -> Observation:
        """Execute one discrete action."""

    @abstractmethod
    def move_forward(self, distance: float, target_y: float | None = None) -> Observation:
        """Move forward by one planner-selected distance toward an optional height."""

    @abstractmethod
    def rotate(self, angle_radians: float) -> Observation:
        """Rotate by a planner-selected signed navigation-frame angle."""

    @abstractmethod
    def get_observation(self) -> Observation:
        """Read the current observation without moving."""

    @abstractmethod
    def get_agent_pose(self) -> Pose:
        """Return the current agent pose."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""

    def get_scene_metadata(self) -> dict[str, object]:
        return {}

    def __enter__(self) -> NavigationBackend:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["NavigationBackend"]
