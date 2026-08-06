#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
release_repository="hadr-nav/dso-hadr-search"
release_tag="dso-runtime-assets-v1"
asset_name="thor-schema2-direct-navmesh-Linux64.tar.gz"
asset_sha256="ca67b005f968fa59d90debd9edb62b76ecbcc9c418fdd8852b38fcf9964c4a5f"
destination=${1:-"$project_root/../procthor/build/ai2thor/builds/schema2-procedural"}
verifier="$project_root/dso/scripts/verify_runtime_assets.py"

if ! command -v gh >/dev/null; then
    echo "GitHub CLI is required; install it and run: gh auth login" >&2
    exit 1
fi

launcher="$destination/thor-schema2-direct-navmesh-Linux64"
if [[ -e "$launcher" ]]; then
    if python3 "$verifier" --build-directory "$destination" --skip-scenes; then
        echo "Patched AI2-THOR runtime is already complete."
        exit 0
    fi
    echo "Existing AI2-THOR runtime does not match the checked-in version: $destination" >&2
    exit 1
fi

temporary_directory=$(mktemp -d)
trap 'rm -rf "$temporary_directory"' EXIT

gh release download "$release_tag" \
    --repo "$release_repository" \
    --pattern "$asset_name" \
    --dir "$temporary_directory"
printf '%s  %s\n' "$asset_sha256" "$temporary_directory/$asset_name" | sha256sum -c -
mkdir -p "$destination"
tar -xzf "$temporary_directory/$asset_name" -C "$destination"
python3 "$verifier" --build-directory "$destination" --skip-scenes
