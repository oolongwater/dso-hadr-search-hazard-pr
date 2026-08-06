#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
download_url="https://www.dropbox.com/scl/fo/w06xunp6artrgj3rmx8wr/AIWUdXs2eXMqklO5NbPqQtY?rlkey=ata7d2hi970pjb0wyfhedm4d5&dl=1"
destination=${1:-"$project_root/data/procthor/dso-procthor-levels-1-3-100-v1/scenes"}
verifier="$project_root/dso/scripts/verify_runtime_assets.py"

for command in curl unzip; do
    if ! command -v "$command" >/dev/null; then
        echo "$command is required to download the ProcTHOR scene corpus." >&2
        exit 1
    fi
done

if [[ -d "$destination" ]] && [[ -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    if python3 "$verifier" --scenes-directory "$destination" --skip-build; then
        echo "ProcTHOR scene corpus is already complete."
        exit 0
    fi
    echo "Existing scene directory does not match the checked-in manifest: $destination" >&2
    exit 1
fi

temporary_directory=$(mktemp -d)
trap 'rm -rf "$temporary_directory"' EXIT
archive="$temporary_directory/scenes.zip"

curl --fail --location --retry 3 --retry-all-errors "$download_url" --output "$archive"
mkdir -p "$destination"
unzip -qjo "$archive" '*.json' -d "$destination"
python3 "$verifier" --scenes-directory "$destination" --skip-build
