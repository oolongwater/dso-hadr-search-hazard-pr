#!/usr/bin/env python3
"""Verify the downloadable scene corpus and patched AI2-THOR Linux runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_LAUNCHER_SHA256 = "95aa60b3bead39f7f6e86d1189664d420d8323243c63a45766601cd8fadeba88"
_BUILD_NAME = "thor-schema2-direct-navmesh-Linux64"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_scenes(project_root: Path, scenes_directory: Path) -> None:
    manifest_path = project_root / "dso/configs/scenes/procthor-scenes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest["scenes"]
    if len(records) != manifest["scene_count"]:
        raise ValueError("scene manifest count does not match its records")
    expected_names = {str(record["filename"]) for record in records}
    actual_names = {path.name for path in scenes_directory.glob("scene_*.json")}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(f"scene file set mismatch: missing={missing}, unexpected={unexpected}")
    mismatches = [
        str(record["filename"])
        for record in records
        if _sha256(scenes_directory / str(record["filename"])) != record["sha256"]
    ]
    if mismatches:
        raise ValueError(f"scene content hash mismatch: {mismatches}")
    print(f"Verified {len(records)} ProcTHOR scenes in {scenes_directory}")


def _verify_build(build_directory: Path) -> None:
    launcher = build_directory / _BUILD_NAME
    data_directory = build_directory / f"{_BUILD_NAME}_Data"
    unity_player = build_directory / "UnityPlayer.so"
    required = (
        launcher,
        data_directory / "data.unity3d",
        data_directory / "boot.config",
        data_directory / "Plugins/libjpeg.so",
        unity_player,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"AI2-THOR runtime is incomplete: {missing}")
    if not launcher.stat().st_mode & 0o111:
        raise ValueError(f"AI2-THOR launcher is not executable: {launcher}")
    actual_sha256 = _sha256(launcher)
    if actual_sha256 != _LAUNCHER_SHA256:
        raise ValueError(
            f"AI2-THOR launcher hash mismatch: expected {_LAUNCHER_SHA256}, got {actual_sha256}"
        )
    print(f"Verified patched AI2-THOR runtime in {build_directory}")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes-directory",
        type=Path,
        default=project_root / "data/procthor/dso-procthor-levels-1-3-100-v1/scenes",
    )
    parser.add_argument(
        "--build-directory",
        type=Path,
        default=project_root.parent / "procthor/build/ai2thor/builds/schema2-procedural",
    )
    parser.add_argument("--skip-scenes", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.skip_scenes and args.skip_build:
        raise ValueError("cannot skip both scene and build verification")
    project_root = Path(__file__).resolve().parents[2]
    if not args.skip_scenes:
        _verify_scenes(project_root, args.scenes_directory.expanduser().resolve(strict=True))
    if not args.skip_build:
        _verify_build(args.build_directory.expanduser().resolve(strict=True))


if __name__ == "__main__":
    main()
