# Unity Physics in AI2-THOR (SIM-03 / SIM-04)

This document explains how to manipulate the Unity PhysX engine that AI2-THOR uses, using **Python action-dispatch only** (no Unity Editor install).

## Architecture

```
Python client (ai2thor.controller.Controller.step)
        │  action string + kwargs
        ▼
AI2-THOR Unity build (PhysX rigidbody simulation)
        │  event.frame (RGB), event.metadata (objects, agent, success)
        ▼
Demo scripts capture video + state deltas
```

AI2-THOR scenes run inside a Unity player. 

Object motion, collisions, and support relations are handled by **Unity PhysX**. 

The Python client does not implement physics; it sends actions such as `AdvancePhysicsStep`, `DropHandObject`, and `SetObjectPoses` into the Unity build.

## SIM-03 Feature-Compatibility Matrix


| Capability                               | Python action-dispatch (prebuilt AI2-THOR)                                   | Unity C# custom build required                     |
| ---------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------- |
| Manual deterministic physics stepping    | `PausePhysicsAutoSim`, `AdvancePhysicsStep(timeStep=...)`                    | Optional custom timestep policies                  |
| Gravity / rigidbody drop                 | `DropHandObject`, `ThrowObject`                                              | Custom mass/drag curves                            |
| Applied force / impulse                  | `TouchThenApplyForce`, `PushObject`, `ApplyForceObject`, `DirectionalPush` | Continuous force fields; **`StartEarthquake`** (custom build)      |
| Break / debris                           | `BreakObject`, `SliceObject`                                                 | New fracture assets, particle debris                               |
| Reposition / spawn obstruction           | `SetObjectPoses`, `PlaceObjectAtPoint`, `TeleportObject` (collision-checked) | New prefabs, procedural debris meshes                              |
| Mass / stability tuning                  | `SetMassProperties`                                                          | Per-object stability components                                    |
| Object state toggles                     | `OpenObject`, `CloseObject`, `ToggleObjectOn/Off`                            | Custom hazard state machines                                       |
| Reachability / path re-query             | `GetReachablePositions`, `GetShortestPathToPoint`                            | Runtime NavMesh rebake                                             |
| Navigation collision response            | `MoveAhead` success/failure from PhysX collisions                            | Custom agent collider shapes                                       |
| RGB observation stream                   | `event.frame` each step                                                      | Custom camera shaders                                              |
| Depth observation stream                 | `renderDepthImage=True`                                                      | Custom depth/noise models                                          |
| Smoke / fire volumetrics                 | Image-space fog overlay (`apply_fog_overlay`) on prebuilt player             | **`SetSmokeDensity`**, **`StartHazardFire`** (custom build)        |
| Environment-wide earthquake shake        | Image-space FPV shift + per-object `DirectionalPush` loop                      | **`StartEarthquake`** / **`StopEarthquake`** (custom build)        |
| Progressive connector blocking over time | Partial — instant placement via `SetObjectPoses`; growth needs C#            | Animated obstacle growth, sliding debris                           |
| Deterministic replay logging             | Metadata snapshots + action trace                                            | Custom Unity-side state serializers                                |




### API boundary rule of thumb

If the effect can be expressed as **(a)** applying an existing AI2-THOR action to an existing scene object, or **(b)** post-processing `event.frame`, it works with the prebuilt engine. If it requires **new geometry, shaders, particle systems, or simulation rules**, it needs cloning [allenai/ai2thor](https://github.com/allenai/ai2thor), editing Unity C#, and rebuilding the player.

## Runnable hazard scenes (SIM-04 / SIM-05)

Run from repository root:

```bash
source .venv-ai2thor/bin/activate
python3 ai2thor_pipeline/cli/hazard_scenes.py all --scene FloorPlan1 --severity 0.7 --seed 0
```

Individual families:

```bash
python3 ai2thor_pipeline/cli/hazard_scenes.py smoke --scene FloorPlan1
python3 ai2thor_pipeline/cli/hazard_scenes.py earthquake --scene FloorPlan1
python3 ai2thor_pipeline/cli/hazard_scenes.py obstruction --scene FloorPlan1
```

Outputs land in family subfolders under `output_ai2thor/hazards/`:

| Family             | Folder        | Scene video                         | Summary                                 |
| ------------------ | ------------- | ----------------------------------- | --------------------------------------- |
| Earthquake         | `earthquake/` | `scene_earthquake_FloorPlan1.mp4`   | `scene_earthquake_FloorPlan1.summary.json` |
| Obstruction        | `obstruction/`| `scene_obstruction_FloorPlan1.mp4`  | `scene_obstruction_FloorPlan1.summary.json` |
| Smoke / visibility | `fire_smoke/` | `scene_smoke_FloorPlan1.mp4`        | `scene_smoke_FloorPlan1.summary.json`   |

Each run also writes a canonical `.scenegraph.json` beside the MP4. Use
`cli/hazard_variants.py` for side-by-side compare videos and
`cli/scene_graph_videos.py` for `.scenegraph.mp4` replays.

## Demo 1 — Falling / impact-like hazard

**Unity mechanism:** Pick up objects (`PickupObject` + `forceAction=True`), release them (`DropHandObject`), optionally break an object (`BreakObject`), apply impulses (`TouchThenApplyForce`), then integrate PhysX with `PausePhysicsAutoSim` + repeated `AdvancePhysicsStep`.

**Latent properties perturbed:** object support (held → unsupported), stability (impulses), structural integrity (`isBroken`).

**Observable propagation:** object positions change; items fall to floor; broken object state changes.

### Four feasibility questions


| Question                                | Answer                                                                 |
| --------------------------------------- | ---------------------------------------------------------------------- |
| Can it be rendered?                     | Yes — RGB frames capture falling/debris motion.                        |
| Can it alter simulator state?           | Yes — `event.metadata["objects"]` positions and `isBroken` change.     |
| Can the robot sense the consequence?    | Yes — RGB/depth show moved/fallen objects; metadata exposes new poses. |
| Can it change planning or task success? | Yes — fallen objects can obstruct paths or occlude targets.            |


**Unity-side code required:** No (prebuilt AI2-THOR actions only).

## Demo 2 — Blocked-passage / progressive obstruction

**Unity mechanism:** Find a pose where `MoveAhead` succeeds, move a `Stool` into the choke point with `SetObjectPoses`, settle with manual PhysX steps, then re-attempt `MoveAhead`.

**Latent properties perturbed:** connector clearance / traversability.

**Observable propagation:** `MoveAhead` transitions from success to failure; collision error in metadata.

### Four feasibility questions


| Question                                | Answer                                                                  |
| --------------------------------------- | ----------------------------------------------------------------------- |
| Can it be rendered?                     | Yes — agent view shows obstacle blocking forward path.                  |
| Can it alter simulator state?           | Yes — object pose changes; navigation action outcome changes.           |
| Can the robot sense the consequence?    | Yes — collision failure + visual occlusion of free space.               |
| Can it change planning or task success? | Yes — direct traversability change; graph edge should become `blocked`. |


**Unity-side code required:** No for instant blockage. Yes for *progressive* growth (e.g. debris pile expanding over time).

## Demo 3 — Smoke / visibility degradation

**Unity mechanism (API-only approximation):** Agent rotates/holds while Python composites a growing fog overlay onto each `event.frame` (`apply_fog_overlay`). Visibility metric = `1 - density`.

**Latent properties perturbed (conceptual):** local visibility / observation confidence.

**Observable propagation:** RGB contrast drops; visibility score decreases in summary trace.

### Four feasibility questions


| Question                                | Answer                                                                        |
| --------------------------------------- | ----------------------------------------------------------------------------- |
| Can it be rendered?                     | Yes — degraded RGB is visible in FPV video.                                   |
| Can it alter simulator state?           | Not in Unity physics state (overlay is post-processed).                       |
| Can the robot sense the consequence?    | Yes — degraded RGB/depth-like sensing from obscured frames.                   |
| Can it change planning or task success? | Yes — visibility/confidence penalties can trigger replanning or skip regions. |


**Unity-side code required:** Yes for physically grounded smoke (particle emission, light scattering, depth noise tied to density field). The current demo is an honest API-only stand-in noted in summaries.

## Key actions reference (AI2-THOR 5.0.0, verified locally)

```python
controller.step(action="PausePhysicsAutoSim")
controller.step(action="AdvancePhysicsStep", timeStep=0.05)
controller.step(action="UnpausePhysicsAutoSim")

controller.step(action="PickupObject", objectId=obj_id, forceAction=True)
controller.step(action="DropHandObject")

controller.step(action="TouchThenApplyForce", objectId=obj_id,
                direction={"x": 0, "y": 0, "z": 1}, forceMagnitude=180.0, forceAction=True)

controller.step(action="BreakObject", objectId=obj_id, forceAction=True)

controller.step(action="SetObjectPoses", objectPoses=[{
    "objectName": obj["name"],
    "position": {"x": 0.0, "y": 0.0, "z": 0.5},
    "rotation": obj["rotation"],
}])
```

**Notes from local probing:**

- `PickupObject` / force actions need `forceAction=True` when the object is not in the visibility cone.
- `AdvancePhysicsStep` only works while auto-simulation is paused.
- `TeleportObject` can fail on collision (`"... is colliding with ..."`); `SetObjectPoses` is more reliable for a single deliberate placement.
- `SetObjectPoses` **removes every moveable/pickupable object not in the list.** It is a whole-scene pose reset, not an incremental move. To move one object while keeping the rest, use `PlaceObjectAtPoint` (used by the progressive-obstruction scene to accumulate a pile without deleting other objects).
- `PlaceObjectAtPoint` fails with "Spawn area not clear" if the target overlaps the agent, a wall, or another object; the obstruction scene tries several candidate slots per placement.
- Always read `event.metadata["lastActionSuccess"]` and `errorMessage` when validating an action on a new scene.



## Mapping to proposal hazard families


| Proposal hazard family   | Demo      | Latent variables (conceptual) | Propagation                              |
| ------------------------ | --------- | ----------------------------- | ---------------------------------------- |
| Earthquake / impact-like | `falling` | support, stability, integrity | objects fall/break; debris alters layout |
| Progressive obstruction  | `blocked` | clearance, connector state    | passage open → blocked                   |
| Smoke / fire-like        | `smoke`   | density, visibility           | observation confidence decreases         |




## Parameterized hazard scenes (proposal section 3.1.1)

The three one-shot demos above (SIM-04) are extended into **parameterized, causally
evolving scenes** (the SIM-05 direction) in [hazard/model.py](hazard/model.py) and the
runner [cli/hazard_scenes.py](cli/hazard_scenes.py). Each family exposes latent variables
via `HazardConfig` and a transparent per-tick propagation rule, and renders an MP4 that
composites the agent first-person view with an overhead map view.

Run:

```bash
python3 ai2thor_pipeline/cli/hazard_scenes.py all --scene FloorPlan1 --severity 0.7 --seed 0
# or one family:
python3 ai2thor_pipeline/cli/hazard_scenes.py smoke --scene FloorPlan1
python3 ai2thor_pipeline/cli/hazard_scenes.py earthquake --scene FloorPlan1
python3 ai2thor_pipeline/cli/hazard_scenes.py obstruction --scene FloorPlan1
```

Outputs under `output_ai2thor/hazards/<family>/`: `scene_<family>_<Scene>.mp4`, matching
`*.summary.json` (full `HazardConfig` + per-tick trace + feasibility), and sample stills.

**Per-scene capture profiles** (writer FPS, substeps/tick, paused regime, PhysX time step) live
in `SCENE_CAPTURE_PROFILES` in `cli/hazard_scenes.py`:


| Scene       | FPS | Substeps/tick | Regime                            | `time_step` |
| ----------- | --- | ------------- | --------------------------------- | ----------- |
| earthquake  | 60  | 16            | held paused (every step rendered) | 1/60        |
| smoke       | 30  | 12            | pause/advance/unpause per tick    | 1/60        |
| obstruction | 30  | 9             | pause/advance/unpause per tick    | 0.0167      |

`substeps * time_step` is the simulated seconds per tick and is held constant across FPS
changes, so raising FPS adds rendered frames without altering hazard propagation.

Smoke deliberately writes fewer FPS than it captures substeps: 12 frames per tick played at
30 fps stretches 8 s of simulated time over a 16 s video, so the thermal field and smoke
front are legible at half speed. Each rendered frame calls `AdvanceHeatField` with
`time_step` (1/60 s), so smoke density and the heat panel update every frame rather than
once per tick. Earthquake matches FPS to substeps to stay at real time.

**Scene graph panel (smoke and earthquake comparison videos):** a fourth fixed-width column
(`GRAPH_PANEL_WIDTH = 560`) renders a ground-truth node-link graph from live
`event.metadata["objects"]`, following ThinkGraphs representation conventions:
node label + hazard-relevant `state` attributes (`broken`, `hot`, `cooked`, `moving`,
`open`), authoritative `on`/`in` support edges from `parentReceptacles` (resolved to
the smallest parent AABB via `_specific_parent`), and deterministic directional
predicates (`left`, `right`, `above`, `under`, `near`) from axis-aligned bounding
boxes with a room-center anchor (top-5 neighbours within 2.0 m). Spatial predicates
are computed but **not drawn**. Up to 10 tracked nodes are chosen as `(child, parent)`
pairs so every child has a drawable support edge; layout tiers by structural supporter
role (not the AI2-THOR `receptacle` flag). Hazard overlays: smoke — hot nodes from
thermal field (`heat_state_ids`); earthquake — severed support edges (dashed red).

**Passage panel (obstruction comparison videos):** replaces the generic scene graph with
a dedicated clearance view. A fixed roster of six blockers (from `setup.blocker_ids`)
lights up as each is placed; a lateral clearance bar at the chokepoint shows the
remaining walkway gap (computed geometrically from placed AABBs). Blockers on the
countertop are tagged `above walkway` and do not shrink the bar. Readout:
`placed=N/6  gap=X.XXm  MoveAhead=…  state=open|constrained|blocked`.

**Canonical scene graph JSON:** each hazard MP4 is accompanied by a
`*.scenegraph.json` artifact built from [`scene_graph/schema.py`](scene_graph/schema.py) and
[`scene_graph/export.py`](scene_graph/export.py). The file stores `initial` (oracle
ground truth at setup) and `final` (obs layer after the last tick, with hazard-specific
updates and `last_updated` timestamps). Validated via [`validators.py`](validators.py).
The rendered panels above are unchanged; the JSON is the LLM-consumable schema for
search/connectivity planning. On iTHOR single-room scenes one region is synthesized;
obstruction adds here/ahead regions plus a passage connector.


Override the writer FPS for any run with `--fps N`. The chosen profile is recorded under
`capture_profile` in each `*.summary.json`.

### Shared latent variables (`HazardConfig`)

Full parameter reference and high-level action wrappers: [`hazard/functions/README.md`](hazard/functions/README.md).

Nested structure: shared timeline fields on `HazardConfig`, family-specific latents under `config.smoke`, `config.earthquake`, and `config.obstruction`.


| Field             | Meaning                                        |
| ----------------- | ---------------------------------------------- |
| `hazard_type`     | `smoke` / `earthquake` / `obstruction`         |
| `onset_step`      | tick at which the hazard begins                |
| `severity`        | 0-1 master intensity (scales density, impulse) |
| `total_ticks`     | number of propagation ticks                    |
| `seed`            | RNG seed for reproducibility                   |
| `smoke.source_position` | hazard origin (auto-selected if unset)   |




### Smoke / fire-like (`SmokeFireHazard`)

- Latents: `config.smoke.emission_rate`, `config.smoke.spread_rate`, `config.smoke.ignition_temperature_c`, `config.smoke.access_density_cutoff`, source position. Thermal solver params on `config.thermal`.
- Rule: `source_density` rises over time (capped by severity); a smoke `radius` grows from the
source; agent-local density drives fog (overlay or native). Per-tick trace logs `visibility`,
`max_temp_c`, `agent_temp_c`, `num_hot_objects`, and ignition counts.
- **Native thermal coupling** (`native_effects=True`): Unity integrates a 2D floor temperature
field on every capture substep (`AdvanceHeatField(time_step)`). Flames inject heat; the field diffuses and
cools; objects above `config.thermal.hot_threshold_c` become `Temperature.Hot` and cookables
are cooked. Fire spread ignites the hottest breakable above `ignition_temperature_c`, capped at
`1 + round(severity * 10)` flames. `fire_spread_rate` applies only in overlay mode. Python renders a third video panel: the overhead camera
frame converted to greyscale with the temperature field alpha-blended on top (inferno colormap,
fixed 22–600 C scale).
- Limitations: 2D floor-plane field (no wall occlusion), effective diffusivity (not a full CFD
model), explicit integration with CFL substeps and temperature clamping.
- Boundary: with the **prebuilt player**, visibility degradation is an image-space overlay and
  ignition uses a distance-based fallback. With a **custom Unity build**, `set_smoke()` /
  `start_fire()` drive scene fog + particles, and the thermal field drives object temperature
  and spread.



### Earthquake / impact-like (`EarthquakeHazard`)

- Latents: `config.earthquake.impulse_base_newtons` x severity, per-object stability (mass-scaled),
`config.earthquake.integrity_threshold`, `config.earthquake.shake_period_ticks`, `config.earthquake.shake_axis_deg`, `config.earthquake.impulse_scale`,
`config.earthquake.shake_pixels`, seed.
- Rule (prebuilt / overlay mode): a shaking floor is, in the room frame, one
*synchronized pseudo-force* on every object. Each tick computes
`phase = sin(2*pi*step/shake_period_ticks)`; **all** un-broken movable objects receive a
`DirectionalPush` along one shared shake axis (`shake_axis_deg` when `phase >= 0`, flipped
180 deg otherwise) with magnitude `impulse_base_newtons * severity * abs(phase) * impulse_scale`.
PhysX integrates each object's fall from its own position — no per-object trajectory is
computed, so nothing teleports. Cumulative impulse past `integrity_threshold` triggers
`BreakObject` (debris). Finalize settles physics and diffs object states; reachable-position
count is compared before/after (traversability change).
- Rule (custom build / native mode): at hazard onset call `start_earthquake()` once; Unity-side `FixedUpdate` applies rigidbody impulses/torque
and shakes the agent camera. Per-tick `BreakObject` still runs from the same integrity threshold.
- Smoothness / no teleporting: the runner pauses PhysX **once** at scene start and never
unpauses between ticks (`hold_paused`), so zero simulation time passes un-rendered; every
`AdvancePhysicsStep` is captured. In overlay mode, camera shake is an **image-space** FPV shift
(`render_shift`, +-`shake_pixels` scaled by `phase`); in native mode the shake is Unity-side.
- Viewpoint: `setup()` scores subsampled reachable poses x 8 yaws by how many movable
(preferably elevated) objects fall inside a ~80 deg FOV wedge within 4 m, then teleports to the
best pose and `LookDown` ~18 deg so falling objects stay in frame.
- Boundary: whole-environment oscillation requires the custom build; the synchronized per-object
impulse loop is the prebuilt-player equivalent.



### Progressive obstruction (`ProgressiveObstructionHazard`)

- Latents: spawn chokepoint (from `find_moveahead_success_pose`), `config.obstruction.growth_rate`
(placements per tick), `config.obstruction.constrained_cost_ratio`.
- Rule: every `round(1/growth_rate)` ticks a blocker is placed into the passage with
`place_obstruction()` (`PlaceObjectAtPoint` under the hood), largest object dead-centre first (to block `MoveAhead`), then smaller
objects into widely-spaced side/back slots to grow a visible pile. Each tick re-tests
`MoveAhead` and path cost (`GetShortestPathToPoint`), driving the connector state machine
`open -> constrained -> blocked`.
- FPV aim: `aim_at_pile()` (called in `setup()` and after every tick's tests) yaws the agent toward
the placed-object centroid (or the slot centre ahead before anything is placed) and `LookDown`
~30 deg, so every captured FPV frame is framed on the accumulation spot rather than the door.
- Boundary: instant per-object placement is API-side; smoothly *animated* growth would need Unity C#.



## Custom Unity build for native smoke/fire and earthquake (SIM-05+)

The prebuilt player cannot render volumetric smoke, procedural flames, scene fog, or gravity
oscillation. This repo ships custom Unity C# under [../ai2thor_custom/](../ai2thor_custom/) and
Python integration that switches to native effects when `--local-executable` is passed.

### One-time prerequisites

1. Install **Unity Hub** and **Unity Editor 2020.3.25f1** (Intel editor; runs under Rosetta on Apple Silicon).
2. Clone ai2thor at the pinned commit (with git-lfs) into `~/ai2thor-src-full`:

```bash
git clone https://github.com/allenai/ai2thor.git ~/ai2thor-src-full
cd ~/ai2thor-src-full
git checkout f0825767cd50d69f666c7f282e54abfe58f1e917
git lfs pull
pip install invoke
pip install -e .
```

3. Apply hazard scripts and build FloorPlan1 only (~10-15 min):

```bash
./ai2thor_custom/apply_custom_unity.sh ~/ai2thor-src-full
cd ~/ai2thor-src-full
invoke local-build --scenes FloorPlan1
```

The player lands at `~/ai2thor-src-full/unity/builds/thor-OSXIntel64-local/.../AI2-THOR`.

Or run the helper: `./ai2thor_custom/build_local.sh` (checks Unity + LFS + invoke).

### Native hazard actions (custom build only)

High-level Python API (preferred): see [`hazard/functions/README.md`](hazard/functions/README.md).

```python
from hazard.functions import (
    EarthquakeParams, FireParams, SmokeParams,
    start_earthquake, stop_earthquake,
    start_fire, stop_fire, set_smoke,
)

start_fire(controller, FireParams(object_id=obj_id, severity=0.8))
stop_fire(controller)
set_smoke(controller, SmokeParams(density=0.55))
start_earthquake(controller, EarthquakeParams(magnitude=2.5, frequency_hz=2.0))
stop_earthquake(controller)
```

Probe after building:

```bash
python3 ai2thor_custom/probe_native_effects.py \
  --local-executable ~/ai2thor-src-full/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR
```

### Regenerate hazard videos with native effects

```bash
python3 ai2thor_pipeline/cli/hazard_scenes.py smoke --scene FloorPlan1 \
  --local-executable ~/ai2thor-src-full/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR

python3 ai2thor_pipeline/cli/hazard_scenes.py earthquake --scene FloorPlan1 \
  --local-executable ~/ai2thor-src-full/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR
```

Passing `--local-executable` sets `HazardConfig.native_effects=True` automatically. The stock
prebuilt player path (no flag) keeps image-space fog and `DirectionalPush` earthquake behaviour.

Unity sources added by this repo:

- `unity/Assets/Scripts/HazardEffects.cs` — coordinator partial class
- `unity/Assets/Scripts/fire_smoke/` — fire/smoke particles, scene fog, smoke actions
- `unity/Assets/Scripts/earthquake/` — earthquake ground acceleration + camera shake actions
- `unity/Assets/Scripts/obstruction/` — no native C# (Python-only)

## Next steps toward SIM-05 / SIM-06

1. Persist the `HazardConfig` schema and per-tick trace as the hazard-config artifact (done in summaries).
2. Feed the timestamped graph attribute changes (connector `blocked`, region `visibility`) into the scene graph.
3. ~~For volumetric smoke or smoothly animated debris growth, plan a custom Unity build path and extend the matrix.~~ Native smoke/fire/earthquake effects are implemented in `ai2thor_custom/`; obstruction growth animation remains API-side only.

