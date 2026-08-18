# Custom AI2-THOR hazard build

Native Unity effects for SIM-05 smoke/fire and earthquake hazards.

## Layout

| Path | Purpose |
| --- | --- |
| `unity/Assets/Scripts/HazardEffects.cs` | Coordinator MonoBehaviour (partial class entry point) |
| `unity/Assets/Scripts/HazardEffectsActions.Core.cs` | Shared `EnsureHazardEffects()` wiring |
| `unity/Assets/Scripts/fire_smoke/` | Fire/smoke + thermal field: `StartHazardFire`, `StopHazardFire`, `SetSmokeDensity`, `SetThermalParams`, `AdvanceHeatField` |
| `unity/Assets/Scripts/earthquake/` | Earthquake physics/camera shake + `StartEarthquake`, `StopEarthquake` actions |
| `unity/Assets/Scripts/obstruction/` | No native C# (Python-only hazard); placeholder README |
| `apply_custom_unity.sh` | Copy scripts into a local ai2thor checkout |
| `build_local.sh` | Clone/checkout, LFS pull, FloorPlan1 + Procedural build |
| `probe_native_effects.py` | Verify custom actions against a local executable |
| `regenerate_native_hazards.sh` | Probe + regenerate smoke/earthquake MP4s |

## One-time setup

1. **Unity Hub** (installed via Homebrew cask `unity-hub`).
2. **Unity Editor 2020.3.25f1** — installed here to user Applications:
   `/Users/a65945/Applications/Unity/Unity.app`
   Symlinks exist at `/Applications/Unity/Hub/Editor/2020.3.25f1/Unity.app`.
3. **Activate Personal license once** (required before batchmode builds):
   - Open Unity Hub, sign in with a Unity ID, and install/activate **2020.3.25f1**.
   - Or open the editor once: `open ~/Applications/Unity/Unity.app`
   - Batchmode builds fail with `License is not active (com.unity.editor.headless)` until this step completes.
4. **ai2thor source** at `~/ai2thor-src-full`, commit `f0825767cd50d69f666c7f282e54abfe58f1e917`.

## Build

```bash
./ai2thor_custom/build_local.sh
```

Requires `invoke`, `boto3`, and `pip install -e ~/ai2thor-src-full` in `.venv-ai2thor`.

## Regenerate videos

```bash
./ai2thor_custom/regenerate_native_hazards.sh
```

Or pass an explicit executable path.

## Python integration

High-level hazard parameters and action wrappers: [`ai2thor_pipeline/hazard/functions/README.md`](../ai2thor_pipeline/hazard/functions/README.md).

Custom build executable (after `./ai2thor_custom/build_local.sh`):

```
~/ai2thor-src-full/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR
```

Pass `--local-executable` to `cli/hazard_scenes.py` or `cli/hazard_variants.py` to enable `HazardConfig.native_effects=True` automatically.

Generate parameter comparison videos:

```bash
python3 ai2thor_pipeline/cli/hazard_variants.py all --scene FloorPlan1 --headless \
  --local-executable ~/ai2thor-src-full/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR
```
