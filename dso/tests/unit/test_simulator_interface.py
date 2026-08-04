from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

import dso_hadr.simulator.ai2thor as simulator_module
from dso_hadr.simulator.ai2thor import (
    ProcTHORSimulator,
    SimulatorConfig,
    load_simulator_config,
)
from dso_hadr.simulator.recording import StepRecorder


class FakeEvent:
    def __init__(self, *, success: bool = True) -> None:
        self.frame = np.zeros((4, 6, 3), dtype=np.uint8)
        self.metadata = {
            "lastActionSuccess": success,
            "errorMessage": "",
            "agent": {
                "position": {"x": 1.0, "y": 0.9, "z": 2.0},
                "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                "cameraHorizon": 30.0,
            },
        }

    def __bool__(self) -> bool:
        return self.metadata["lastActionSuccess"]


class FakeController:
    latest = None

    def __init__(self, **parameters: object) -> None:
        self.parameters = parameters
        self.reset_scene = None
        self.actions = []
        self.stopped = False
        FakeController.latest = self

    def reset(self, *, scene: dict[str, object]) -> FakeEvent:
        self.reset_scene = scene
        return FakeEvent()

    def step(self, action: object = None, **parameters: object) -> FakeEvent:
        self.actions.append((action, parameters))
        return FakeEvent()

    def stop(self) -> None:
        self.stopped = True


def _config() -> SimulatorConfig:
    return SimulatorConfig(
        local_executable_path=Path("/test/thor-Linux64"),
        width=300,
        height=300,
        quality="Low",
        snap_to_grid=False,
    )


def test_load_simulator_config(tmp_path: Path) -> None:
    path = tmp_path / "ai2thor.json"
    executable = tmp_path / "thor-Linux64"
    executable.touch()
    path.write_text(
        json.dumps(
            {
                "local_executable_path": "thor-Linux64",
                "width": 640,
                "height": 480,
                "quality": "Low",
                "snap_to_grid": False,
            }
        ),
        encoding="utf-8",
    )

    assert load_simulator_config(path) == SimulatorConfig(
        local_executable_path=executable,
        width=640,
        height=480,
        quality="Low",
        snap_to_grid=False,
    )


def test_load_scene_step_and_close(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(simulator_module, "Controller", FakeController)
    scene_path = tmp_path / "scene_0001.json"
    scene = {
        "metadata": {
            "agent": {
                "position": {"x": 1.0, "y": 0.9, "z": 2.0},
                "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
                "horizon": 30.0,
                "standing": True,
            }
        }
    }
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    simulator = ProcTHORSimulator(_config())
    loaded = simulator.load_scene(scene_path)
    moved = simulator.step({"action": "MoveAhead", "moveMagnitude": 0.25})
    simulator.close()

    controller = FakeController.latest
    assert controller.parameters == {
        "scene": "Procedural",
        "local_executable_path": "/test/thor-Linux64",
        "width": 300,
        "height": 300,
        "quality": "Low",
        "snapToGrid": False,
    }
    assert controller.reset_scene == scene
    assert controller.actions == [
        (
            "TeleportFull",
            {
                "raise_for_failure": True,
                **scene["metadata"]["agent"],
            },
        ),
        ({"action": "MoveAhead", "moveMagnitude": 0.25}, {}),
    ]
    assert loaded
    assert moved
    assert controller.stopped


def test_step_recorder_writes_rgb_and_trajectory(tmp_path: Path) -> None:
    output_dir = tmp_path / "episode"
    recorder = StepRecorder(output_dir, "scene_0001")
    event = FakeEvent()

    record = recorder.record(
        {"action": "MoveAhead", "moveMagnitude": 0.25},
        event,
    )

    assert record == {
        "scene_id": "scene_0001",
        "step_id": 0,
        "action": {"action": "MoveAhead", "moveMagnitude": 0.25},
        "success": True,
        "pose": {
            "position": {"x": 1.0, "y": 0.9, "z": 2.0},
            "rotation": {"x": 0.0, "y": 90.0, "z": 0.0},
            "camera_horizon": 30.0,
        },
        "rgb": "rgb/000000.png",
    }
    saved = json.loads((output_dir / "trajectory.jsonl").read_text(encoding="utf-8"))
    assert saved == record
    with Image.open(output_dir / "rgb/000000.png") as image:
        assert image.size == (6, 4)
