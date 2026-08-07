"""Download DSO data and demos from Dropbox with rclone."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

_REMOTE_ROOT = "dropbox:Projects/HADR Navigation"


def _copy(remote: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["rclone", "copy", remote, str(destination), "--checksum"],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"rclone failed while downloading {remote}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "selection",
        choices=("data", "demo", "all"),
        default="data",
        nargs="?",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if shutil.which("rclone") is None:
        raise RuntimeError("rclone is required")

    project_root = Path(__file__).resolve().parents[2]
    selections = ("data", "demo") if args.selection == "all" else (args.selection,)
    for selection in selections:
        destination = project_root / "data"
        if selection == "demo":
            destination /= "demo"
        _copy(f"{_REMOTE_ROOT}/{selection}", destination)


if __name__ == "__main__":
    main()
