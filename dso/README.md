# DSO Python Workspace

This workspace contains DSO-specific Python package boundaries, tests,
configuration, scripts, and examples.

The current stage provides a configurable, content-locked 100-scene ProcTHOR
corpus and a non-hazard navigation baseline. The pipeline extracts a symbolic
scene graph and ground-truth traversability map, plans with symbolic Dijkstra
and metric A*, follows the route through the AI2-THOR interface, and records
aligned RGB-D observations and poses. It does not implement localization,
hazards, route validation, or replanning.

See [the navigation stack](../docs/dso/navigation-stack.md) for the data flow,
component boundaries, configuration, and recorded artifacts.

## Runtime Assets

Download and verify the exact scene corpus from the existing public Dropbox
folder:

~~~bash
./dso/scripts/download_procthor_scenes.sh
python3 dso/scripts/verify_runtime_assets.py --skip-build
~~~

The patched Linux AI2-THOR runtime is compiled from the checked-in Unity project.
Install and activate Unity 2020.3.25f1, then run from the repository root:

~~~bash
unity_editor="$HOME/Unity/Hub/Editor/2020.3.25f1/Editor/Unity"
build_directory="$PWD/../procthor/build/ai2thor/builds/schema2-procedural"
mkdir -p "$build_directory"
UNITY_BUILD_NAME="$build_directory/thor-schema2-direct-navmesh-Linux64" \
  "$unity_editor" -quit -batchmode \
  -logFile "$build_directory/build.log" \
  -projectpath "$PWD/unity" \
  -buildTarget Linux64 \
  -executeMethod Build.Linux64
python3 dso/scripts/verify_runtime_assets.py
~~~

The build directory matches the checked-in simulator configuration. Generated
Unity builds, trajectories, and videos remain outside Git.

## Lightweight Checks

From the repository root, use only Pixi:

~~~bash
pixi run test
pixi run format
pixi run typecheck
~~~

The locally generated JSON houses stay under the ignored data/ directory. The
checked-in configs/scenes/procthor-corpus.json file owns generation parameters
and the explicit 80/20 split. The separate configs/scenes/procthor-scenes.json
manifest locks the files by content hash. configs/simulator/ai2thor.json points
to the local schema-2-capable Unity executable and owns rendering parameters.
