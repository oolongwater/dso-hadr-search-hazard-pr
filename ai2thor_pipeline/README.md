# AI2-THOR Hazard Scenes

Unity PhysX hazard demonstrations through AI2-THOR Python actions — no Unity Editor required.

Run all commands from the **repository root**.

## Layout

```
ai2thor_pipeline/
  cli/                 Entry-point scripts
  hazard/              Hazard simulation (model, utils, custom parameters)
  scene_graph/         OpenCV panel + canonical schema/export/video
  core/                Shared AI2-THOR controller and video helpers
  docs/                Extended physics / API reference
  requirements*.txt
```

| Folder | Contents |
| --- | --- |
| `cli/` | `hazard_scenes.py`, `hazard_variants.py`, `scene_graph_videos.py` |
| `hazard/` | `model.py`, `utils.py`, `functions/` (latent params + action wrappers) |
| `scene_graph/` | `panel.py`, `schema.py`, `export.py`, `video.py`, `validators.py` |
| `core/` | `thor.py`, `video.py`, `changepoint.py` (re-export shim → [`changepoint_kit/`](changepoint_kit/)) |
| `docs/` | `UNITY_PHYSICS.md` |

## Pinned environment

| Component | Locked version |
| --- | --- |
| Python | 3.12.x |
| ai2thor | 5.0.0 |
| Unity build COMMIT_ID | `f0825767cd50d69f666c7f282e54abfe58f1e917` |

Lockfiles: [requirements.txt](requirements.txt), [requirements.lock.txt](requirements.lock.txt)

## Install

```bash
python3.12 -m venv .venv-ai2thor
source .venv-ai2thor/bin/activate
pip install -r ai2thor_pipeline/requirements.lock.txt
```

**ffmpeg** is required for H.264 video finalization: `brew install ffmpeg`

## Parameterized hazard scenes

```bash
python3 ai2thor_pipeline/cli/hazard_scenes.py all --scene FloorPlan1 --severity 0.7 --seed 0
```

Individual families: `smoke`, `earthquake`, `obstruction`.

Options: `--severity`, `--seed`, `--onset`, `--ticks`, `--fps`, `--local-executable PATH`.

Regenerate with native Unity effects: `./ai2thor_custom/regenerate_native_hazards.sh`

## Variant comparison videos

```bash
python3 ai2thor_pipeline/cli/hazard_variants.py all --scene FloorPlan1 --headless \
  --local-executable ~/ai2thor-src-full/unity/builds/thor-OSXIntel64-local/thor-OSXIntel64-local.app/Contents/MacOS/AI2-THOR
```

## Canonical scene-graph videos

```bash
python3 ai2thor_pipeline/cli/scene_graph_videos.py --all
```

## Custom parameters

Latent variables per hazard family: [hazard/functions/README.md](hazard/functions/README.md)

Variant overrides: [cli/hazard_variants.py](cli/hazard_variants.py)

## Outputs

All artifacts under `output_ai2thor/hazards/<family>/` — compare MP4s, scene MP4s, `.scenegraph.json`, `.scenegraph.mp4`, and `.summary.json` files.

## Changepoints (SAM earthquake demo)

The scene-action-map demo writes a streaming **`{label}.changepoints.json`** beside the video under `output_ai2thor/hazards/earthquake/sam/`. Each record is one visited decision node: world pose, door/room context, clutter cluster, traversable exits, controller decision, and artifact paths (`clip`, `payload_png`, `views`).

Schema owner: [`changepoint_kit/changepoint.py`](changepoint_kit/changepoint.py) (`SCHEMA_VERSION`, `Changepoint`, `ChangepointLog`, `load_changepoints`). Import without AI2-THOR or OpenCV:

```python
from changepoint_kit import load_changepoints
# or: from core import load_changepoints  # re-export shim

for cp in load_changepoints("output_ai2thor/hazards/earthquake/sam/four_room_ring_1f.changepoints.json"):
    print(cp.visit_index, cp.id, cp.summary(), cp.cluster_counts())
```

Validate a file:

```bash
python3 ai2thor_pipeline/changepoint_kit/validate.py
python3 ai2thor_pipeline/examples/read_changepoints.py
```

Run the demo:

```bash
python3 ai2thor_pipeline/cli/demo_scene_action_map.py --headless
```

See [docs/UNITY_PHYSICS.md](docs/UNITY_PHYSICS.md) for the API-vs-Unity-C# compatibility matrix and propagation rules.
