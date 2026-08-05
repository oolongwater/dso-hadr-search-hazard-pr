"""Simulator-neutral navigation backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dso_hadr.types.navigation import NavigationAction, Observation, Point3, Pose, ShortestPath


class NavigationBackend(ABC):
    """Operations required by symbolic-path execution."""

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
    def sample_navigable_point(self) -> Point3:
        """Sample a navigable point using the configured seed."""

    @abstractmethod
    def get_navmesh_path(
        self,
        start: Point3,
        goal: Point3,
        max_path_length: float,
    ) -> ShortestPath | None:
        """Query a bounded ground-truth navmesh path between two points."""

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
