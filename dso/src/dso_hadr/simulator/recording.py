"""Record AI2-THOR action results without choosing the actions."""

from __future__ import annotations

import json
from pathlib import Path

from ai2thor.server import Event  # type: ignore[import-not-found]
from PIL import Image


class StepRecorder:
    """Write one RGB image and one JSON line for each simulator step."""

    def __init__(self, output_dir: Path, scene_id: str) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.scene_id = scene_id
        self.rgb_dir = self.output_dir / "rgb"
        self.trajectory_path = self.output_dir / "trajectory.jsonl"
        self.output_dir.mkdir(parents=True)
        self.rgb_dir.mkdir()
        self.trajectory_path.touch()
        self._step_id = 0

    def record(self, action: dict[str, object], event: Event) -> dict[str, object]:
        """Persist the result of one externally supplied action."""

        image_path = self.rgb_dir / f"{self._step_id:06d}.png"
        Image.fromarray(event.frame).save(image_path)

        agent = event.metadata["agent"]
        record: dict[str, object] = {
            "scene_id": self.scene_id,
            "step_id": self._step_id,
            "action": dict(action),
            "success": event.metadata["lastActionSuccess"],
            "pose": {
                "position": dict(agent["position"]),
                "rotation": dict(agent["rotation"]),
                "camera_horizon": agent["cameraHorizon"],
            },
            "rgb": image_path.relative_to(self.output_dir).as_posix(),
        }
        with self.trajectory_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._step_id += 1
        return record
