#!/usr/bin/env bash
# Regenerate smoke + earthquake hazard MP4s with the custom Unity build.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_EXE="${1:-${AI2THOR_LOCAL_EXE:-}}"

if [[ -z "$BUILD_EXE" ]]; then
  BUILD_EXE="$HOME/ai2thor-src-full/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR"
fi

if [[ ! -x "$BUILD_EXE" ]]; then
  echo "ERROR: custom AI2-THOR executable not found or not executable:" >&2
  echo "  $BUILD_EXE" >&2
  echo "Build first: ./ai2thor_custom/build_local.sh" >&2
  exit 1
fi

cd "$REPO_ROOT"
source .venv-ai2thor/bin/activate

echo "Probing native hazard actions..."
python3 ai2thor_custom/probe_native_effects.py --local-executable "$BUILD_EXE"

echo "Regenerating smoke scene..."
python3 ai2thor_pipeline/cli/hazard_scenes.py smoke --scene FloorPlan1 --local-executable "$BUILD_EXE"

echo "Regenerating earthquake scene..."
python3 ai2thor_pipeline/cli/hazard_scenes.py earthquake --scene FloorPlan1 --local-executable "$BUILD_EXE"

echo "Regenerating variant comparison videos (smoke, earthquake, obstruction)..."
python3 ai2thor_pipeline/cli/hazard_variants.py all --scene FloorPlan1 --headless --local-executable "$BUILD_EXE"

echo "Done. Outputs under output_ai2thor/hazards/{fire_smoke,earthquake,obstruction}/"
