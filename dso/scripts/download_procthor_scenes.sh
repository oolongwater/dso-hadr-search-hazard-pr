#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "$0")/../.." && pwd)"
download_url="https://www.dropbox.com/scl/fo/w06xunp6artrgj3rmx8wr/AIWUdXs2eXMqklO5NbPqQtY?rlkey=ata7d2hi970pjb0wyfhedm4d5&dl=1"
local_dir="$project_root/data/procthor/dso-procthor-levels-1-3-100-v1/scenes"
archive="$local_dir.zip"

mkdir -p "$local_dir"
curl -fsSL "$download_url" -o "$archive"
unzip -qjo "$archive" '*.json' -d "$local_dir"
rm "$archive"
