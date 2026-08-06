#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$script_directory/download_procthor_scenes.sh"
"$script_directory/download_ai2thor_build.sh"
