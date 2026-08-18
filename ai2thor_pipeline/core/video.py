"""Video helpers for AI2-THOR FPV recording (ported from autonomous_nav_benevolence2.py)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def rgb_to_bgr_uint8(rgb) -> np.ndarray:
    """Convert HxWx3 RGB (uint8 or float) to BGR uint8 for OpenCV VideoWriter."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected HxWx>=3 RGB, got {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _resolve_ffmpeg_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for d in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        p = Path(d) / name
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def finalize_mp4(path: Path) -> None:
    """Re-encode mp4v to H.264 Baseline for QuickTime / browser / IDE preview."""
    if path.suffix.lower() != ".mp4" or not path.is_file():
        return
    ffmpeg_bin = _resolve_ffmpeg_tool("ffmpeg")
    if ffmpeg_bin is None:
        print(f"  [WARN] ffmpeg not found; {path.name} stays mp4v (may not play in QuickTime).")
        return
    tmp = path.with_suffix(".h264.tmp.mp4")
    vf: list[str] = []
    ffprobe_bin = _resolve_ffmpeg_tool("ffprobe")
    if ffprobe_bin:
        try:
            pr = subprocess.run(
                [
                    ffprobe_bin,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0:s=x",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if pr.returncode == 0 and pr.stdout.strip():
                parts = pr.stdout.strip().split("x")
                if len(parts) == 2:
                    w0, h0 = int(parts[0]), int(parts[1])
                    if min(w0, h0) < 480:
                        if w0 >= h0:
                            vf = ["-vf", "scale=480:-2:flags=neighbor"]
                        else:
                            vf = ["-vf", "scale=-2:480:flags=neighbor"]
        except (ValueError, OSError):
            pass
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=mono",
        "-i",
        str(path),
        "-shortest",
        "-map",
        "1:v:0",
        "-map",
        "0:a:0",
        *vf,
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.0",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "48k",
        "-ac",
        "1",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
        tmp.replace(path)
        if sys.platform == "darwin":
            subprocess.run(["xattr", "-c", str(path)], check=False, capture_output=True)
        print(f"  Finalized {path.name} -> H.264")
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"  [WARN] ffmpeg finalize failed for {path}: {e}")
        if tmp.is_file():
            tmp.unlink(missing_ok=True)


def probe_video(path: Path) -> dict:
    """Return width, height, frame_count, duration via ffprobe (empty dict if unavailable)."""
    ffprobe_bin = _resolve_ffmpeg_tool("ffprobe")
    if ffprobe_bin is None or not path.is_file():
        return {}
    try:
        pr = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,nb_frames,duration",
                "-of",
                "json",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if pr.returncode != 0:
            return {}
        import json

        blob = json.loads(pr.stdout)
        streams = blob.get("streams") or []
        if not streams:
            return {}
        s = streams[0]
        return {
            "width": int(s.get("width", 0)),
            "height": int(s.get("height", 0)),
            "frame_count": int(s.get("nb_frames", 0) or 0),
            "duration": float(s.get("duration", 0) or 0),
        }
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
