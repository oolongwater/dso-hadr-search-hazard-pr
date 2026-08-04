# DSO Python Workspace

This workspace contains DSO-specific Python package boundaries, tests,
configuration, scripts, and examples.

The current stage adds a configurable, content-locked 100-scene ProcTHOR
corpus and a minimal AI2-THOR interface. The interface loads a scene, applies
its saved agent pose, executes externally supplied actions, and records RGB,
pose, action, success, scene ID, and step ID. It does not choose actions or
implement planning, waypoint following, localization, or hazards.

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
