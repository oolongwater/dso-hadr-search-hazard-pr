"""Run every configured navigation episode once, without fallback routes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _resolved_paths(template_path: Path, template: dict[str, Any]) -> dict[str, str]:
    paths = template["paths"]
    return {
        key: str((template_path.parent / value).resolve(strict=key != "output_directory"))
        for key, value in paths.items()
        if key != "output_directory"
    }


def _run_episode(
    runner: Path,
    config_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(runner), str(config_path)],
        cwd=runner.parents[2],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = output_directory / "logs" / f"{config_path.stem}.log"
    log_path.write_text(completed.stdout, encoding="utf-8")
    summary_path = output_directory / config_path.stem / "summary.json"
    if completed.returncode != 0 or not summary_path.exists():
        return {
            "scene_id": config_path.stem,
            "status": "FAILED",
            "return_code": completed.returncode,
            "log": str(log_path),
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    execution = summary["execution"]
    success = (
        bool(execution["success"])
        and bool(execution["trajectory_on_navmesh"])
        and int(execution["collisions"]) == 0
    )
    return {
        "scene_id": config_path.stem,
        "status": "PASS" if success else "FAILED",
        "return_code": completed.returncode,
        "steps": int(execution["steps"]),
        "traveled_distance": float(execution["traveled_distance"]),
        "final_distance": float(execution["final_distance"]),
        "collisions": int(execution["collisions"]),
        "log": str(log_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    episodes_path = args.episodes.expanduser().resolve(strict=True)
    batch = json.loads(episodes_path.read_text(encoding="utf-8"))
    template_path = episodes_path.parent / batch["episode_config"]
    template = json.loads(template_path.read_text(encoding="utf-8"))
    output_directory = args.output_directory.expanduser().resolve(strict=False)
    output_directory.mkdir(parents=True, exist_ok=False)
    configs_directory = output_directory / "configs"
    logs_directory = output_directory / "logs"
    configs_directory.mkdir()
    logs_directory.mkdir()
    paths = _resolved_paths(template_path, template)

    config_paths = []
    for episode in batch["episodes"]:
        scene_id = str(episode["scene_id"])
        config = {
            "controller": template["controller"],
            "episode": {
                "scene_id": scene_id,
                "start_room_id": episode["start_room_id"],
                "goal_room_id": episode["goal_room_id"],
                "start_position": episode["start_position"],
                "goal_position": episode["goal_position"],
                "geodesic_distance": episode["geodesic_distance"],
            },
            "paths": {
                **paths,
                "output_directory": str(output_directory / scene_id),
            },
            "scene_graph": template["scene_graph"],
        }
        config_path = configs_directory / f"{scene_id}.json"
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_paths.append(config_path)

    runner = Path(__file__).resolve().with_name("run_navigation_episode.py")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(_run_episode, runner, config_path, output_directory)
            for config_path in config_paths
        ]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    records.sort(key=lambda record: str(record["scene_id"]))
    passed = [record for record in records if record["status"] == "PASS"]
    distances = [float(record["traveled_distance"]) for record in passed]
    report: dict[str, object] = {
        "scene_count": len(records),
        "passed": len(passed),
        "failed": len(records) - len(passed),
        "records": records,
    }
    if distances:
        report["traveled_distance"] = {
            "minimum": min(distances),
            "maximum": max(distances),
            "mean": statistics.fmean(distances),
        }
    (output_directory / "batch-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if len(passed) != len(records):
        raise RuntimeError(f"{len(records) - len(passed)} navigation demos failed")


if __name__ == "__main__":
    main()
