"""Encode validated navigation RGB trajectories as H.264 demo videos."""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Any


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def _validated_frames(scene_directory: Path) -> tuple[Path, ...]:
    summary = json.loads((scene_directory / "summary.json").read_text(encoding="utf-8"))
    execution = summary["execution"]
    if not execution["success"]:
        raise ValueError(f"{scene_directory.name} navigation did not succeed")
    if not execution["trajectory_on_navmesh"]:
        raise ValueError(f"{scene_directory.name} trajectory left the navmesh")
    if int(execution["collisions"]) != 0:
        raise ValueError(f"{scene_directory.name} trajectory contains collisions")

    frames = tuple(sorted((scene_directory / "rgb").glob("*.png")))
    expected_names = tuple(f"{index:06d}.png" for index in range(len(frames)))
    if tuple(frame.name for frame in frames) != expected_names:
        raise ValueError(f"{scene_directory.name} RGB frame sequence is not contiguous")
    trajectory_records = sum(
        1
        for line in (scene_directory / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    )
    if not frames or len(frames) != trajectory_records:
        raise ValueError(
            f"{scene_directory.name} has {len(frames)} frames and "
            f"{trajectory_records} trajectory records"
        )
    dimensions = {_png_dimensions(frame) for frame in frames}
    if len(dimensions) != 1:
        raise ValueError(f"{scene_directory.name} RGB frame dimensions are inconsistent")
    return frames


def _probe_video(path: Path, *, frame_count: int, fps: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    probe = json.loads(completed.stdout)
    streams = probe.get("streams", ())
    if len(streams) != 1:
        raise ValueError(f"{path} does not contain exactly one video stream")
    stream = streams[0]
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise ValueError(f"{path} is not H.264 yuv420p")
    if int(stream["nb_frames"]) != frame_count:
        raise ValueError(f"{path} frame count does not match its RGB trajectory")
    if Fraction(stream["r_frame_rate"]) != fps:
        raise ValueError(f"{path} frame rate is not {fps} fps")
    return probe


def _encode_scene(scene_directory: Path, *, fps: int, overwrite: bool) -> dict[str, Any]:
    frames = _validated_frames(scene_directory)
    width, height = _png_dimensions(frames[0])
    output_path = scene_directory / "video.mp4"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"video already exists: {output_path}")
    temporary_path = scene_directory / f"video.{os.getpid()}.tmp.mp4"
    temporary_path.unlink(missing_ok=True)
    try:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-start_number",
                "0",
                "-i",
                str(scene_directory / "rgb" / "%06d.png"),
                "-frames:v",
                str(len(frames)),
                "-c:v",
                "libx264",
                "-profile:v",
                "main",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(temporary_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {scene_directory.name}: {completed.stderr.strip()}"
            )
        probe = _probe_video(temporary_path, frame_count=len(frames), fps=fps)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "scene_id": scene_directory.name,
        "status": "PASS",
        "video": str(output_path),
        "frames": len(frames),
        "fps": fps,
        "width": width,
        "height": height,
        "duration": float(probe["format"]["duration"]),
        "size_bytes": output_path.stat().st_size,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_directory", type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.fps <= 0 or args.workers <= 0 or args.expected_scene_count <= 0:
        raise ValueError("fps, workers, and expected scene count must be positive")
    root = args.episodes_directory.expanduser().resolve(strict=True)
    scene_directories = tuple(
        path for path in sorted(root.glob("scene_[0-9][0-9][0-9][0-9]")) if path.is_dir()
    )
    if len(scene_directories) != args.expected_scene_count:
        raise ValueError(
            f"expected {args.expected_scene_count} scene directories, "
            f"found {len(scene_directories)}"
        )

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _encode_scene,
                scene_directory,
                fps=args.fps,
                overwrite=args.overwrite,
            )
            for scene_directory in scene_directories
        ]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    records.sort(key=lambda record: str(record["scene_id"]))
    report = {
        "scene_count": len(records),
        "passed": len(records),
        "failed": 0,
        "fps": args.fps,
        "total_frames": sum(int(record["frames"]) for record in records),
        "total_duration": sum(float(record["duration"]) for record in records),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "records": records,
    }
    (root / "video-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
