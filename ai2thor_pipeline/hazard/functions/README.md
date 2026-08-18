# Hazard parameters and high-level AI2-THOR actions

Typed parameters and thin wrappers over `controller.step` for earthquake, fire/smoke, and obstruction hazards. Scene runners in [`hazard/model.py`](../model.py) and [`cli/hazard_scenes.py`](../../cli/hazard_scenes.py) build on these; this folder is the canonical place to discover and call the API.

## Quick start

Scripts add `ai2thor_pipeline` to `sys.path` (see `cli/hazard_scenes.py`), then import from `hazard.functions`:

```python
from core.thor import make_controller
from hazard.functions import (
    EarthquakeParams,
    FireParams,
    SmokeParams,
    ObstructionParams,
    start_earthquake,
    stop_earthquake,
    start_fire,
    stop_fire,
    set_smoke,
    place_obstruction,
)

controller = make_controller("FloorPlan1", local_executable_path="/path/to/custom/AI2-THOR")
controller.reset("FloorPlan1")

# Native effects (custom Unity build required)
start_earthquake(controller, EarthquakeParams(magnitude=2.5, frequency_hz=2.0))
start_fire(controller, FireParams(object_id="Bottle|...", severity=0.8))
set_smoke(controller, SmokeParams(density=0.55))
set_smoke(controller, 0.55)  # float shorthand also works

stop_earthquake(controller)
stop_fire(controller)
set_smoke(controller, 0.0)

# Obstruction uses stock AI2-THOR (no custom build)
place_obstruction(
    controller,
    ObstructionParams(
        object_id="Stool|...",
        position={"x": 0.0, "y": 0.0, "z": 0.0},
    ),
)
```

Probe all native actions after building a custom player:

```bash
python3 ai2thor_custom/probe_native_effects.py \
  --local-executable /path/to/AI2-THOR
```

Run full parameterized scenes:

```bash
python3 ai2thor_pipeline/cli/hazard_scenes.py smoke --scene FloorPlan1 \
  --local-executable /path/to/AI2-THOR
```

Passing `--local-executable` sets `HazardConfig.native_effects=True` automatically.

## Scene generation (step by step)

Two ways to generate a hazard scene MP4: the **CLI** (fastest) or a **Python script** (full control over `HazardConfig`).

### Prerequisites

1. From the repo root, activate the AI2-THOR venv:

```bash
source .venv-ai2thor/bin/activate
```

2. Choose a player:
   - **Stock player** (no extra build): smoke uses image-space fog; earthquake uses `DirectionalPush` + camera shake. Obstruction always works on stock.
   - **Custom Unity build** (native fire/smoke/earthquake): build once per [`ai2thor_custom/README.md`](../../ai2thor_custom/README.md), then pass `--local-executable /path/to/AI2-THOR`.

---

### Option A — CLI (recommended)

**Step 1.** Pick a hazard family: `smoke`, `earthquake`, or `obstruction`.

**Step 2.** Run the scene runner from the repo root:

```bash
python3 ai2thor_pipeline/cli/hazard_scenes.py smoke --scene FloorPlan1 \
  --severity 0.7 --seed 0 --onset 3 --ticks 70
```

For native Unity effects, add the custom executable:

```bash
python3 ai2thor_pipeline/cli/hazard_scenes.py earthquake --scene FloorPlan1 \
  --local-executable ~/ai2thor-src-full/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR
```

**Step 3.** Wait for the run to finish. The script will print the output path and a short summary (broken objects, passage blocked, etc.).

**Step 4.** Find artifacts under `output_ai2thor/hazards/<family>/`:

| Artifact | Example path |
| --- | --- |
| Video | `output_ai2thor/hazards/fire_smoke/scene_smoke_FloorPlan1.mp4` |
| Summary JSON | `output_ai2thor/hazards/fire_smoke/scene_smoke_FloorPlan1.summary.json` |
| Sample frames | `output_ai2thor/hazards/fire_smoke/scene_smoke_FloorPlan1_frames/` |

**Step 5.** (Optional) Run all three families in one invocation:

```bash
python3 ai2thor_pipeline/cli/hazard_scenes.py all --scene FloorPlan1 --headless
```

Useful CLI flags: `--severity`, `--seed`, `--onset`, `--ticks`, `--fps`, `--headless`, `--width`, `--height`, `--local-executable`.

---

### Option B — Python script (programmatic)

Use this when you need custom latents (`config.smoke.*`, `config.earthquake.*`, `config.obstruction.*`) or to drive scenes from your own code.

**Step 1.** Add `ai2thor_pipeline` to `sys.path` and import the scene runner:

```python
import sys
import importlib.util
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent / "ai2thor_pipeline"  # adjust if needed
sys.path.insert(0, str(PIPELINE))

from hazard.functions import HazardConfig, SmokeLatents
from hazard.model import build_hazard
from core.thor import make_controller

_spec = importlib.util.spec_from_file_location(
    "hazard_scenes", PIPELINE / "cli" / "hazard_scenes.py"
)
_hazard_scenes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hazard_scenes)
run_scene = _hazard_scenes.run_scene
```

If your script already lives in `ai2thor_pipeline/`, set `PIPELINE = Path(__file__).resolve().parent` instead.

**Step 2.** Build a `HazardConfig` for the hazard you want:

```python
config = HazardConfig(
    hazard_type="smoke",       # "smoke" | "earthquake" | "obstruction"
    scene="FloorPlan1",
    severity=0.7,
    seed=0,
    onset_step=3,              # tick when hazard starts
    total_ticks=70,              # smoke default in CLI is 70; others 40
    native_effects=False,      # True when using --local-executable / custom build
    smoke=SmokeLatents(
        spread_rate=0.15,
        room_fill_rate=0.03,
        source_position=None,  # auto-pick a breakable object if unset
    ),
)
```

Set `native_effects=True` (or pass `local_executable_path` to `make_controller`) when you want `start_fire`, `set_smoke`, and `start_earthquake` instead of overlay mode.

**Step 3.** Create the AI2-THOR controller and run the scene:

```python
controller = make_controller(
    config.scene,
    headless=True,
    width=640,
    height=480,
    local_executable_path=None,  # or path to custom AI2-THOR binary
)

try:
    summary = run_scene(
        controller,
        "smoke",           # must match config.hazard_type
        config,
        physics_steps_final=60,  # tail frames after finalize (smoke CLI uses 60)
        fps_override=None,       # or 30 to override capture profile
    )
finally:
    controller.stop()
```

**Step 4.** Read results from the returned `summary` dict and from disk:

```python
print(summary["output_mp4"])      # relative path to MP4
print(summary["final"])           # feasibility + final state (broken ids, blocked passage, …)
print(summary["config"])          # full HazardConfig as dict
```

**Step 5.** What happens inside `run_scene` (you normally do not call these yourself):

1. `controller.reset(scene)` and attach an overhead map camera.
2. `build_hazard(name, config)` → `SmokeFireHazard`, `EarthquakeHazard`, or `ProgressiveObstructionHazard`.
3. `hazard.setup(controller)` — site viewpoint, baseline state.
4. For each tick: `hazard.tick(controller, step)` — discrete events (ignition, breaking). Smoke also calls `hazard.substep(controller, dt, frac)` on every physics substep to interpolate fog density and advance the thermal field.
5. Physics substeps are advanced and frames are written to the MP4.
6. `hazard.finalize(controller)` — cleanup (`stop_fire`, `stop_earthquake`, settle physics).
7. Summary JSON is written next to the video.

**Step 6.** (Optional) Call low-level functions directly without the full scene loop — useful for probes or custom scripts:

```python
from hazard.functions import EarthquakeParams, start_earthquake, stop_earthquake

controller.reset("FloorPlan1")
start_earthquake(controller, EarthquakeParams(magnitude=2.5, frequency_hz=2.0))
# … your own stepping / capture …
stop_earthquake(controller)
```

See [`probe_native_effects.py`](../../ai2thor_custom/probe_native_effects.py) for a minimal native-action smoke test.

---

| Layer | Purpose | Examples |
| --- | --- | --- |
| **Action params** | One-shot Unity / stock AI2-THOR calls | `EarthquakeParams`, `FireParams`, `SmokeParams`, `ObstructionParams` |
| **Simulation latents** | Per-tick scene evolution in hazard classes | `EarthquakeLatents`, `SmokeLatents`, `ObstructionLatents` nested in `HazardConfig` |

Action params hide AI2-THOR naming quirks (`severity` maps to `moveMagnitude`, `frequency_hz` to `frequencyHz`).

## Action parameters

### Earthquake (`earthquake.py`)

| Field | Default | Unity action |
| --- | --- | --- |
| `magnitude` | `2.5` | `StartEarthquake(magnitude=...)` |
| `frequency_hz` | `2.0` | `StartEarthquake(frequencyHz=...)` |

Functions: `start_earthquake`, `stop_earthquake`, `earthquake_params_from_latents(severity, latents)`.

Requires a **custom Unity build** with native earthquake actions.

### Fire / smoke (`fire_smoke.py`)

| Class | Field | Default | Unity action |
| --- | --- | --- | --- |
| `FireParams` | `object_id` | (required) | `StartHazardFire(objectId=...)` |
| `FireParams` | `severity` | `0.7` | `StartHazardFire(moveMagnitude=...)` |
| `SmokeParams` | `density` | `0.0` | `SetSmokeDensity(density=...)` clamped 0–1 |

Functions: `start_fire`, `stop_fire`, `set_smoke`.

Requires a **custom Unity build** for native particles and fog. Without it, scene runners use image-space fog overlays instead.

### Obstruction (`obstruction.py`)

| Field | Default | AI2-THOR action |
| --- | --- | --- |
| `object_id` | (required) | `PlaceObjectAtPoint(objectId=...)` |
| `position` | (required) | `PlaceObjectAtPoint(position=...)` |

Function: `place_obstruction`.

Works with the **stock prebuilt player** — no custom Unity build.

## Simulation latents (`HazardConfig`)

Shared timeline fields on `HazardConfig`:

| Field | Default | Used by |
| --- | --- | --- |
| `hazard_type` | (required) | all |
| `scene` | `"FloorPlan1"` | runner |
| `onset_step` | `3` | all |
| `severity` | `0.7` | smoke, earthquake |
| `total_ticks` | `40` | all |
| `seed` | `0` | all |
| `native_effects` | `False` | smoke, earthquake |

Nested family latents:

**`config.smoke`** (`SmokeLatents`): `emission_rate`, `spread_rate`, `fire_spread_rate` (overlay mode only; native spread uses the thermal field), `room_fill_rate`, `ignition_temperature_c`, `access_density_cutoff`, `burn_delay_ticks_min/max`, `source_position`.

**`config.thermal`** (`ThermalParams`): `ambient_c`, `flame_core_c`, `diffusivity`, `convection_gain`, `cooling_rate`, `flame_radius_m`, `hot_threshold_c`, `resolution_x`, `resolution_z`. Used when `native_effects=True`; the grid is aligned to the overhead map-view camera rectangle.

**`config.earthquake`** (`EarthquakeLatents`): `impulse_base_newtons`, `integrity_threshold`, `shake_period_ticks`, `shake_axis_deg`, `impulse_scale`, `shake_pixels`.

**`config.obstruction`** (`ObstructionLatents`): `growth_rate`, `constrained_cost_ratio`.

Example:

```python
from hazard.functions import HazardConfig, SmokeLatents, EarthquakeLatents

config = HazardConfig(
    hazard_type="smoke",
    severity=0.8,
    native_effects=True,
    smoke=SmokeLatents(spread_rate=0.15, room_fill_rate=0.03),
    earthquake=EarthquakeLatents(shake_period_ticks=8),
)
```

## Native vs overlay modes

| Hazard | `native_effects=False` (stock player) | `native_effects=True` (custom build) |
| --- | --- | --- |
| Smoke / fire | Python fog overlay; distance-based ignition fallback | `start_fire`, `set_smoke`, `stop_fire`, thermal field (`configure_thermal`, `advance_heat_field`) |
| Earthquake | `DirectionalPush` loop + image-space camera shake | `start_earthquake`, `stop_earthquake` |
| Obstruction | `place_obstruction` (stock) | same |

## Variant comparison demos

Demonstrate parameter customizability with vertically stacked comparison videos (two contrasting latent settings per hazard):

```bash
python3 ai2thor_pipeline/cli/hazard_variants.py all --scene FloorPlan1 --headless \
  --local-executable ~/ai2thor-src-full/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR
```

| Hazard | Variants | Latents varied |
| --- | --- | --- |
| smoke | `smolder` vs `inferno` | `severity`, `spread_rate`, `room_fill_rate` |
| earthquake | `tremor` vs `violent` | `impulse_base_newtons`, `shake_period_ticks`, `integrity_threshold` |
| obstruction | `trickle` vs `rapid` | `growth_rate` |

Outputs:

- `output_ai2thor/hazards/fire_smoke/compare_smoke_FloorPlan1.mp4`
- `output_ai2thor/hazards/earthquake/compare_earthquake_FloorPlan1.mp4`
- `output_ai2thor/hazards/obstruction/compare_obstruction_FloorPlan1.mp4`

Per-variant MP4s and summaries are also written (e.g. `scene_smoke_smolder_FloorPlan1.mp4`).

Native smoke scene MP4s are **four panels** side-by-side: agent FPV | overhead map | greyscale top-down room with heat dissipation overlaid (deg C inferno colormap at 80% max alpha, 22–600 C scale, ignition isotherm, flame/agent markers) | **ground-truth scene graph**. Earthquake comparison videos use three panels (FPV | overhead | scene graph). Obstruction uses a **passage-clearance panel** instead of the generic scene graph.

The scene graph panel (`../../scene_graph/panel.py`) is built from live AI2-THOR metadata each frame. Up to 10 nodes are chosen as `(child, parent)` pairs (top-scored breakable/pickupable/moveable objects plus their resolved parent from `parentReceptacles`, picking the smallest parent AABB). Layout tiers by structural supporter role, not the AI2-THOR `receptacle` flag. Spatial predicates (ThinkGraphs Sec. 3.2) are computed but not drawn (`GRAPH_PANEL_WIDTH = 560`).

**Node colors:** grey = normal, green = moving, orange = hot, cyan = cooked, brown = open, red = broken. Yellow ring = support relationship changed this frame.

**Drawn edges:** arced grey = on/in support; solid orange = hazard; dashed red = severed support (earthquake). Smoke heat is orange node fill via `heat_state_ids`.

The passage panel (`render_passage_panel`) shows HERE → clearance bar → AHEAD, a fixed six-blocker roster (pending hollow / in-walkway orange / above-walkway dashed), and readout `placed=N/6 gap=X.XXm MoveAhead=… state=…`. Only floor-level blockers shrink the clearance bar.

## Canonical scene graph (JSON artifact)

Each hazard run also writes a **canonical scene graph** JSON artifact beside the MP4:
`scene_<family>_<Scene>.scenegraph.json` (or variant names like `scene_smoke_smolder_FloorPlan1.scenegraph.json`).

This is separate from the rendered panel in `../../scene_graph/panel.py` — the panel is unchanged; the JSON is the LLM-parseable SayPlan-style schema for oracle and online pipelines.

| Module | Role |
| --- | --- |
| `../../scene_graph/schema.py` | Pydantic v2 models: four node levels (FLOOR, REGION, CONNECTOR, OBJECT), dual `gt`/`obs` state layers, typed edges, `to_networkx()` / `from_networkx()`, `from_ai2thor(house, objects)` |
| `../../scene_graph/validators.py` | Structured `ValidationReport`: referential integrity, `connected_by` via checks, region connectivity, multi-floor, ID convention |
| `../../scene_graph/export.py` | iTHOR adapter (`house_from_ithor`), hazard obs updates (`apply_hazard_obs`), JSON writer |

The artifact contains `{schema_version, scene_id, initial, final, validation}` so ground truth at setup is distinguishable from the final hazard state. Each `obs` object carries `last_updated` (episode timestep).

For iTHOR single-room scenes (`FloorPlan1`), one region is synthesized from map-view bounds; obstruction runs add `room_here_0` / `room_ahead_0` plus a `conn_passage_0` connector. ProcTHOR multi-room mapping is fully implemented in `from_ai2thor()` for future layouts.

Summary JSON includes a `scene_graph` block with node counts and `validation_ok`.

## Thermal coupling (native smoke)

When `native_effects=True`, Unity maintains a 2D floor-plane temperature field (deg C). Flames inject heat; the field diffuses and cools toward ambient; object positions are sampled each substep to set AI2-THOR `Temperature.Hot` and cook cookables. Python reads the field via `advance_heat_field()` once per rendered frame and renders the third video panel.

```python
from hazard.functions import ThermalParams, configure_thermal, advance_heat_field

props = controller.step(action="GetMapViewCameraProperties").metadata["actionReturn"]
configure_thermal(controller, ThermalParams(), props)
start_fire(controller, FireParams(object_id="Bottle|...", severity=0.8))
field = advance_heat_field(controller, delta_time=0.2)
print(field.max_c, field.mean_c, len(field.objects))
```

| Action | Purpose |
| --- | --- |
| `SetThermalParams` | Allocate grid over map-view bounds (via `configure_thermal`) |
| `AdvanceHeatField` | Integrate field for `deltaTime` seconds; returns grid + per-object temps |

Fire spread under native effects ignites the hottest breakable object above `config.smoke.ignition_temperature_c`, capped at `1 + round(severity * 10)` total flames (smolder at severity 0.3 allows ~4, inferno at 1.0 allows ~11). `fire_spread_rate` only applies in overlay mode (distance-based fallback); native mode ignores it. The legacy `heat_threshold` latent was removed; use `ignition_temperature_c` instead.

Variant overrides may name either a `HazardConfig` field or a field on the hazard's
latents dataclass; `cli/hazard_variants.py` partitions them automatically. Varying
`severity` matters for smoke because it caps both room fill and agent density, so
two variants that differ only in spread rates converge on the same final fog level.

## Module map

| File | Contents |
| --- | --- |
| `config.py` | `HazardConfig` |
| `earthquake.py` | `EarthquakeParams`, `EarthquakeLatents`, earthquake wrappers |
| `fire_smoke.py` | `FireParams`, `SmokeParams`, `SmokeLatents`, fire/smoke wrappers |
| `thermal.py` | `ThermalParams`, `HeatField`, `configure_thermal`, `advance_heat_field` |
| `../../scene_graph/panel.py` | Rendered scene graph + passage panel (visual only): `select_tracked_nodes`, `build_graph`, `heat_state_ids`, `render_graph_panel`, `render_passage_panel`, `compute_passage_clearance`, `hazard_graph_edges` |
| `../../scene_graph/schema.py` | Canonical Pydantic schema: `SceneGraph`, enums, `from_ai2thor`, NetworkX interop |
| `../../scene_graph/validators.py` | `validate_scene_graph` → structured `ValidationReport` |
| `../../scene_graph/export.py` | iTHOR house adapter + hazard obs export to `.scenegraph.json` |
| `obstruction.py` | `ObstructionParams`, `ObstructionLatents`, `place_obstruction` |
| `__init__.py` | Public re-exports |

## Related docs

- Custom Unity build: [`ai2thor_custom/README.md`](../../ai2thor_custom/README.md)
- Physics capability matrix: [`../../docs/UNITY_PHYSICS.md`](../../docs/UNITY_PHYSICS.md)
- Scene runner CLI: [`../../cli/hazard_scenes.py`](../../cli/hazard_scenes.py)
- Variant comparison demos: [`../../cli/hazard_variants.py`](../../cli/hazard_variants.py)
