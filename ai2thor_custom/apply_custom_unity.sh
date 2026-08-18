#!/usr/bin/env bash
# Copy custom hazard Unity scripts into a checked-out ai2thor source tree.
set -euo pipefail

AI2THOR_SRC="${1:-$HOME/ai2thor-src-full}"
PROCTHOR_HADR="${PROCTHOR_HADR:-$HOME/procthor-hadr}"
PINNED_COMMIT="24f79883b4889e3f0e6f4ae301808b9025872dfc"
SCHEMA2_PATCH="$PROCTHOR_HADR/integrations/ai2thor/ai2thor-schema2-multifloor.patch"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$AI2THOR_SRC/unity/Assets/Scripts"
EDITOR_DEST="$AI2THOR_SRC/unity/Assets/Editor"
BUILD_CS="$EDITOR_DEST/Build.cs"

if [[ ! -d "$AI2THOR_SRC/.git" ]]; then
  echo "ERROR: ai2thor source not found at $AI2THOR_SRC" >&2
  exit 1
fi

HEAD="$(git -C "$AI2THOR_SRC" rev-parse HEAD)"
if [[ "$HEAD" != "$PINNED_COMMIT" ]]; then
  echo "ERROR: ai2thor HEAD=$HEAD, expected $PINNED_COMMIT (run build_local.sh)" >&2
  exit 1
fi

if [[ ! -f "$AI2THOR_SRC/unity/Assets/Scripts/VerticalConnectorAsset.cs" ]]; then
  if [[ ! -f "$SCHEMA2_PATCH" ]]; then
    echo "ERROR: schema-2 patch not found at $SCHEMA2_PATCH" >&2
    exit 1
  fi
  git -C "$AI2THOR_SRC" apply --check "$SCHEMA2_PATCH"
  git -C "$AI2THOR_SRC" apply "$SCHEMA2_PATCH"
  echo "Applied schema-2 multifloor patch"
fi

mkdir -p "$DEST" "$EDITOR_DEST"
cp "$SCRIPT_DIR/unity/Assets/Scripts/HazardEffects.cs" "$DEST/"
cp "$SCRIPT_DIR/unity/Assets/Scripts/HazardEffectsActions.Core.cs" "$DEST/"
for family in fire_smoke earthquake obstruction; do
  mkdir -p "$DEST/$family"
  if compgen -G "$SCRIPT_DIR/unity/Assets/Scripts/$family/*" > /dev/null; then
    cp -R "$SCRIPT_DIR/unity/Assets/Scripts/$family/." "$DEST/$family/"
  fi
done
rm -f "$DEST/HazardEffectsActions.cs"
cp "$SCRIPT_DIR/unity/Assets/Editor/HazardLocalBuildRunner.cs" "$EDITOR_DEST/"
cp "$SCRIPT_DIR/unity/Assets/Editor/StairPrefabGenerator.cs" "$EDITOR_DEST/"

RESOURCES_DEST="$AI2THOR_SRC/unity/Assets/Resources"
FLAME_SRC="$AI2THOR_SRC/unity/Assets/Physics/SceneSetupPrefabs/CandleFlameLight.prefab"
mkdir -p "$RESOURCES_DEST"
if [[ ! -f "$FLAME_SRC" ]]; then
  echo "ERROR: CandleFlameLight prefab not found at $FLAME_SRC" >&2
  exit 1
fi
cp "$FLAME_SRC" "$RESOURCES_DEST/HazardFlame.prefab"
if [[ -f "$FLAME_SRC.meta" ]]; then
  python3 - <<PY
from pathlib import Path
import uuid
src_meta = Path("$FLAME_SRC.meta").read_text()
new_guid = uuid.uuid4().hex
lines = []
for line in src_meta.splitlines():
    if line.startswith("guid: "):
        lines.append(f"guid: {new_guid}")
    else:
        lines.append(line)
Path("$RESOURCES_DEST/HazardFlame.prefab.meta").write_text("\\n".join(lines) + "\\n")
PY
else
  echo "WARNING: missing CandleFlameLight.meta; Unity may regenerate import metadata" >&2
fi

if ! rg -q 'HazardLocalBuild' "$BUILD_CS"; then
  python3 - <<PY
from pathlib import Path
import re
path = Path("$BUILD_CS")
text = path.read_text()
if "HazardLocalBuild" in text:
    raise SystemExit(0)
pattern = r"(    static void OSXIntel64\(\)\s*\{\s*\n\s*build\(GetBuildName\(\),\s*BuildTargetGroup\.Standalone,\s*BuildTarget\.StandaloneOSX\);\s*\n\s*\})"
match = re.search(pattern, text)
if not match:
    raise SystemExit("Could not patch Build.cs: OSXIntel64 block not found")
insert = match.group(1) + "\n\n    public static void HazardLocalBuild() {\n        OSXIntel64();\n    }"
path.write_text(text[: match.start()] + insert + text[match.end() :])
PY
  echo "Patched $BUILD_CS with HazardLocalBuild()"
fi

echo "Applied hazard Unity scripts to $DEST and $EDITOR_DEST"
