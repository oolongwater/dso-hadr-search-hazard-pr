import json
from pathlib import Path

import numpy as np

from dso_hadr.types.navigation import NavigationAction, Observation, Pose
from dso_hadr.utils.record_utils.navigation_episode import (
    prepare_navigation_episode_output,
    record_observation,
    write_json,
)


def test_navigation_episode_recording_functions(tmp_path: Path) -> None:
    output_directory, trajectory_path = prepare_navigation_episode_output(tmp_path / "episode")
    observation = Observation(
        rgb=np.zeros((4, 6, 3), dtype=np.uint8),
        depth=np.ones((4, 6), dtype=np.float32),
        pose=Pose(1.0, 3.0, 2.0, 0.5),
        collision=False,
    )

    record_observation(
        output_directory=output_directory,
        trajectory_path=trajectory_path,
        scene_id="scene_0019",
        step_id=0,
        action=NavigationAction.MOVE_FORWARD,
        observation=observation,
    )
    write_json(output_directory / "summary.json", {"success": True})

    record = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert record["action"] == "move_forward"
    assert record["pose"] == [1.0, 3.0, 2.0, 0.5]
    assert (output_directory / record["rgb"]).is_file()
    np.testing.assert_array_equal(
        np.load(output_directory / record["depth"]),
        observation.depth,
    )
    assert json.loads((output_directory / "summary.json").read_text(encoding="utf-8")) == {
        "success": True
    }
