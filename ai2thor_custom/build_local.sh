#!/usr/bin/env bash
# Build a FloorPlan1-only custom AI2-THOR player with native hazard effects.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

AI2THOR_SRC="${AI2THOR_SRC:-$HOME/ai2thor-src-full}"
PINNED_COMMIT="24f79883b4889e3f0e6f4ae301808b9025872dfc"
UNITY_VERSION="2020.3.25f1"

if [[ ! -d "$AI2THOR_SRC/.git" ]]; then
  echo "Cloning ai2thor into $AI2THOR_SRC ..."
  git clone --filter=blob:none --sparse https://github.com/allenai/ai2thor.git "$AI2THOR_SRC"
  cd "$AI2THOR_SRC"
  git checkout "$PINNED_COMMIT"
  git sparse-checkout set unity
  git checkout HEAD -- tasks.py setup.py ai2thor
else
  cd "$AI2THOR_SRC"
  git fetch --depth 1 origin "$PINNED_COMMIT" 2>/dev/null || true
  git checkout "$PINNED_COMMIT"
fi

"$SCRIPT_DIR/apply_custom_unity.sh" "$AI2THOR_SRC"

echo "Pulling Unity LFS assets (may take a while) ..."
git lfs pull --include="unity/**"

if [[ ! -d "/Applications/Unity/Hub/Editor/${UNITY_VERSION}" ]]; then
  echo "Unity ${UNITY_VERSION} not installed. Attempting headless install via Unity Hub ..."
  if [[ -x "/Applications/Unity Hub.app/Contents/MacOS/Unity Hub" ]]; then
    "/Applications/Unity Hub.app/Contents/MacOS/Unity Hub" -- --headless install \
      --version "$UNITY_VERSION" || true
  fi
fi

if [[ ! -x "/Applications/Unity/2020.3.25f1/Unity.app/Contents/MacOS/Unity" ]]; then
  echo "ERROR: Unity 2020.3.25f1 is required. Install/activate it in Unity Hub, then rerun." >&2
  exit 1
fi

if [[ -f "${VIRTUAL_ENV:-$REPO_ROOT/.venv-ai2thor}/bin/activate" ]]; then
  # ponytail: venv optional; Unity build does not require Python deps
  source "${VIRTUAL_ENV:-$REPO_ROOT/.venv-ai2thor}/bin/activate"
fi

echo "Building FloorPlan1_physics + Procedural/Procedural (GUI-mode; works with Unity Personal after Hub sign-in) ..."
export HAZARD_UNITY_BUILD_DIR="$AI2THOR_SRC/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local"
mkdir -p "$HAZARD_UNITY_BUILD_DIR"
"/Applications/Unity/2020.3.25f1/Unity.app/Contents/MacOS/Unity" \
  -quit \
  -projectpath "$AI2THOR_SRC/unity" \
  -executeMethod HazardLocalBuildRunner.Run \
  -logFile "$AI2THOR_SRC/thor-OSXIntel64-local-gui.log"

BUILD_EXE="$AI2THOR_SRC/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR"
if [[ ! -x "$BUILD_EXE" ]]; then
  BUILD_EXE="$(find "$AI2THOR_SRC/unity/builds" -name 'AI2-THOR' -type f | head -1)"
fi

if [[ -z "$BUILD_EXE" || ! -x "$BUILD_EXE" ]]; then
  echo "ERROR: build finished but AI2-THOR executable not found under unity/builds" >&2
  exit 1
fi

echo "Build OK: $BUILD_EXE"
