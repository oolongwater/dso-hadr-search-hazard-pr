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

Download and verify the exact connected v3 scene corpus and Linux runtime over
HTTPS from the versioned Dropbox release:

~~~bash
python3 dso/scripts/download_assets.py core
python3 dso/scripts/verify_runtime_assets.py
~~~

The downloader uses `curl` and `tar`, not rclone. It validates the exact byte
count and SHA-256 checksum before extracting an archive. Additional profiles
are available for the 100 videos and lightweight records (`demos`) and the
complete RGB-D trajectories (`full-rgbd`):

~~~bash
python3 dso/scripts/download_assets.py --list
python3 dso/scripts/download_assets.py demos
python3 dso/scripts/download_assets.py full-rgbd
~~~

Corpus regeneration remains available through
`dso/scripts/generate_procthor_corpus.py` when changing the dataset itself.

The patched Linux AI2-THOR runtime is compiled from the integrated AI2-THOR
source under `../procthor/build/ai2thor/unity`.
Install and activate Unity 2020.3.25f1, then run from the repository root:

~~~bash
unity_editor="$HOME/Unity/Hub/Editor/2020.3.25f1/Editor/Unity"
ai2thor_project="$PWD/../procthor/build/ai2thor/unity"
build_directory="$PWD/../procthor/build/ai2thor/builds/schema2-procedural"
mkdir -p "$build_directory"
DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1 \
LD_LIBRARY_PATH="$PWD/../procthor/build/unity-compat/.pixi/envs/default/lib" \
UNITY_BUILD_NAME="$build_directory/thor-schema2-direct-navmesh-Linux64" \
  "$unity_editor" -quit -batchmode \
  -logFile "$build_directory/build.log" \
  -projectPath "$ai2thor_project" \
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

The locally generated JSON houses and demos stay under the ignored data/
directory. The checked-in configs/scenes/procthor-corpus.json file owns generation parameters
and the explicit 80/20 split. The separate configs/scenes/procthor-scenes.json
manifest locks the files by content hash. configs/simulator/ai2thor.json points
to the local schema-2-capable Unity executable and owns rendering parameters.
