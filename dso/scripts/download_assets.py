"""Download and install versioned DSO assets over HTTPS without rclone."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported asset manifest schema")
    assets = document.get("assets")
    profiles = document.get("profiles")
    if not isinstance(assets, dict) or not isinstance(profiles, dict):
        raise TypeError("asset manifest must contain assets and profiles")
    return document


def _select_assets(document: dict[str, Any], selections: tuple[str, ...]) -> tuple[str, ...]:
    assets = document["assets"]
    profiles = document["profiles"]
    requested = selections or tuple(document["default_profiles"])
    selected: list[str] = []
    for selection in requested:
        if selection in profiles:
            candidates = profiles[selection]
        elif selection in assets:
            candidates = (selection,)
        else:
            choices = sorted(set(assets) | set(profiles))
            raise ValueError(f"unknown asset/profile {selection!r}; choose from {choices}")
        for asset_id in candidates:
            if asset_id not in assets:
                raise ValueError(f"profile {selection!r} references unknown asset {asset_id!r}")
            if asset_id not in selected:
                selected.append(asset_id)
    return tuple(selected)


def _download(url: str, partial_path: Path) -> None:
    completed = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
            str(partial_path),
            url,
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"curl failed while downloading {url}")


def _archive(
    asset_id: str,
    record: dict[str, Any],
    cache_directory: Path,
) -> Path:
    archive = cache_directory / str(record["filename"])
    expected_hash = str(record["sha256"])
    expected_bytes = int(record["size_bytes"])
    if archive.exists():
        if archive.stat().st_size == expected_bytes and _sha256(archive) == expected_hash:
            print(f"Verified cached {asset_id}: {archive}")
            return archive
        raise ValueError(f"cached archive does not match the manifest: {archive}")

    partial = archive.with_suffix(archive.suffix + ".part")
    _download(str(record["url"]), partial)
    actual_bytes = partial.stat().st_size
    if actual_bytes != expected_bytes:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"downloaded byte count mismatch for {asset_id}: {actual_bytes} != {expected_bytes}"
        )
    actual_hash = _sha256(partial)
    if actual_hash != expected_hash:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"downloaded SHA-256 mismatch for {asset_id}: {actual_hash} != {expected_hash}"
        )
    os.replace(partial, archive)
    print(f"Downloaded and verified {asset_id}: {archive}")
    return archive


def _install(
    project_root: Path,
    asset_id: str,
    record: dict[str, Any],
    archive: Path,
) -> None:
    installed_paths = tuple(project_root / value for value in record["installed_paths"])
    extraction_directory = project_root / str(record["extract_directory"])
    extraction_directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "tar",
            "--zstd",
            "-xf",
            str(archive),
            "-C",
            str(extraction_directory),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"failed to extract {asset_id} into {extraction_directory}")
    missing = tuple(path for path in installed_paths if not path.exists())
    if missing:
        raise ValueError(f"{asset_id} extraction did not create required paths: {missing}")
    print(f"Installed {asset_id} into {extraction_directory}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("selections", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    manifest_path = project_root / "dso/configs/assets/assets.json"
    document = _load_manifest(manifest_path)
    if args.list:
        print(json.dumps(document, indent=2, sort_keys=True))
        return
    if shutil.which("curl") is None or shutil.which("tar") is None:
        raise RuntimeError("curl and tar with zstd support are required")

    asset_ids = _select_assets(document, tuple(args.selections))
    cache_directory = project_root / "data/downloads"
    cache_directory.mkdir(parents=True, exist_ok=True)
    for asset_id in asset_ids:
        record = document["assets"][asset_id]
        if not args.force:
            existing = tuple(
                project_root / value
                for value in record["installed_paths"]
                if (project_root / value).exists()
            )
            if existing:
                raise FileExistsError(
                    f"{asset_id} is already present at {existing}; use --force to "
                    "overwrite matching archive paths"
                )
        archive = _archive(asset_id, record, cache_directory)
        _install(project_root, asset_id, record, archive)
        archive.unlink()


if __name__ == "__main__":
    main()
