# DSO HADR Search Project

This repository is the shared development workspace for the DSO HADR Search project. It is based on an AI2-THOR source snapshot and adds DSO-specific scaffolding for hazard simulation, symbolic scene graphs, and hazard-aware navigation experiments.

The motivating problem is search and navigation in household disaster-response settings where fire, smoke, and earthquake effects may change what an embodied agent can see, reach, traverse, or safely use. The project aims to build a controlled simulator and benchmark workflow for studying these changes.

## Development setup

Pixi is the environment and task runner for DSO-specific Python work. From the
repository root:

~~~bash
python3 dso/scripts/download_assets.py core
python3 dso/scripts/verify_runtime_assets.py
pixi run test
pixi run format
pixi run typecheck
~~~

The connected v3 ProcTHOR corpus and matching Linux runtime are available as
versioned, checksum-pinned Dropbox downloads over ordinary HTTPS. The installer
uses `curl`, never rclone. See dso/README.md for core, demo-video, full RGB-D,
local regeneration, and Unity build commands. Generated builds, scenes,
trajectories, videos, and other experiment results remain outside Git.

Do not create a separate pip, Conda, or uv environment for this workspace.

See [the ProcTHOR scene corpus contract](docs/dso/procthor-scene-corpus.md) for
the configuration, content lock, and current limitations.
See [the navigation stack](docs/dso/navigation-stack.md) for the implemented
scene extraction, planning, execution, and recording pipeline.

## Workstreams

The simulation and benchmark workstream extends AI2-THOR with parameterized hazard events, visible effects, physical or perceptual scene changes, and metadata exported from Unity to Python.

The scene-graph navigation workstream uses simulator ground truth to construct symbolic scene graphs, update graph state under hazards, sample valid start-goal pairs, search over the graph, follow waypoints, validate routes, and replan when graph state changes.

## Current Milestone

The current milestone covers:

1. Parameterized fire, smoke, and earthquake events in Unity.
2. Visible hazard effects controlled by hazard parameters.
3. Hazard-related scene state exposed from Unity to Python.
4. Symbolic scene graph extraction from simulator ground truth.
5. Hazard-conditioned graph updates for visibility, traversability, accessibility, and connectivity.
6. Valid random start-goal sampling.
7. Graph search, waypoint following, route validation, and replanning.

This repository now contains the ProcTHOR corpus configuration, content
manifest, validation tooling, and a non-hazard navigation baseline. The
baseline extracts a symbolic scene graph and ground-truth traversability map,
runs symbolic Dijkstra and metric A* planning, follows the route in AI2-THOR,
and records aligned RGB-D observations and poses. Hazard behavior,
hazard-conditioned graph updates, route validation, replanning, and
place-recognition evaluation are not implemented yet.

## Relationship To AI2-THOR

The original upstream AI2-THOR README is preserved at [docs/dso/ai2thor-original-readme.md](docs/dso/ai2thor-original-readme.md). Upstream AI2-THOR attribution and citation information remain in that preserved README, and the upstream license remains available in [LICENSE](LICENSE).

DSO-specific documentation lives under `docs/dso/`. DSO-specific Python code belongs under `dso/`. DSO-specific Unity code and assets belong under `unity/Assets/DSOHADR/`.

Changes to upstream AI2-THOR files should be minimized, reviewed carefully, and documented when unavoidable.

## Getting Started

1. Read [PROJECT.md](PROJECT.md), [docs/dso/milestone-1.md](docs/dso/milestone-1.md), and [docs/dso/team-workflow.md](docs/dso/team-workflow.md).
2. Pick or create a GitHub issue for the task.
3. Create one branch for that issue.
4. Work only in the appropriate ownership area unless the issue calls for shared interface work.
5. Run the lightweight checks for `dso/` when editing Python scaffolding or tooling.
