# Navigation Stack

This document describes the currently implemented, non-hazard navigation
baseline. It loads a configured ProcTHOR scene, builds a local scene
representation, plans a room route and a metric route, executes that route in
AI2-THOR, and records aligned RGB-D observations and poses.

Hazard simulation, hazard-conditioned graph updates, route validation,
replanning, and place-recognition evaluation are outside this baseline.

## Entry Point

Run one configured episode from the repository root with Pixi:

~~~bash
pixi run python dso/scripts/run_navigation_episode.py \
  dso/configs/navigation/trajectory-following.json
~~~

The episode configuration contains every path, scene selection, graph,
traversability, and controller parameter used by the run. The loader does not
supply hidden defaults. `dso/configs/navigation/all-scenes.json` contains the
configured start and goal rooms for the 100-scene corpus.

## Execution Flow

```text
episode and simulator configs
            |
            v
load scene and obtain reachable navmesh samples
            |
            v
extract floors, regions, connectivity, and a traversability map
            |
            v
Dijkstra search over symbolic region connectivity
            |
            v
ground symbolic subgoals to traversability-map points
            |
            v
A* search over the stored traversability map
            |
            v
waypoint follower -> navigation backend -> AI2-THOR
            |
            v
RGB-D observations, poses, actions, plans, and summary
```

The top-level sequence is intentionally visible in
[`run_navigation_episode.py`](../../dso/scripts/run_navigation_episode.py).
Reusable behavior remains in the package rather than in the entry point.

## Component Boundaries

| Component | Responsibility |
| --- | --- |
| [`types/navigation.py`](../../dso/src/dso_hadr/types/navigation.py) | Simulator-neutral poses, points, actions, observations, paths, and follower results. |
| [`graph/model.py`](../../dso/src/dso_hadr/graph/model.py) | Scene-graph and traversability-map data structures only. |
| [`scenes/scene_graph.py`](../../dso/src/dso_hadr/scenes/scene_graph.py) | Extract floors, semantic regions, containment, doors, open boundaries, and cross-floor connectors from a scene JSON. |
| [`scenes/traversability.py`](../../dso/src/dso_hadr/scenes/traversability.py) | Build the current ground-truth traversability map from AI2-THOR reachable points and verified navmesh paths. |
| [`planner/symbolic/dijkstra.py`](../../dso/src/dso_hadr/planner/symbolic/dijkstra.py) | Find a minimum-cost route through symbolic regions. |
| [`planner/grounding.py`](../../dso/src/dso_hadr/planner/grounding.py) | Map symbolic transitions and the goal to points in the traversability map. |
| [`planner/motion/astar.py`](../../dso/src/dso_hadr/planner/motion/astar.py) | Find a metric route using only a `TraversabilityMap`, start point, and goal point. |
| [`controller/waypoint_follower.py`](../../dso/src/dso_hadr/controller/waypoint_follower.py) | Convert the metric route into configured forward and rotation actions. |
| [`simulator/navigation_backend.py`](../../dso/src/dso_hadr/simulator/navigation_backend.py) | Define the simulator-neutral execution interface. |
| [`simulator/ai2thor_backend.py`](../../dso/src/dso_hadr/simulator/ai2thor_backend.py) | Adapt that interface to AI2-THOR and translate coordinate conventions. |
| [`utils/record_utils/navigation_episode.py`](../../dso/src/dso_hadr/utils/record_utils/navigation_episode.py) | Persist episode JSON, trajectory records, RGB frames, and depth arrays. |

## Scene Representation And Planner Boundary

`SceneGraph` holds two planning resolutions:

- symbolic floors, regions, containment edges, and connectivity edges for
  room-level planning;
- a `TraversabilityMap` of 3D points and path-bearing edges for motion
  planning across one or multiple levels.

AI2-THOR ground truth is used only while extracting the current
`TraversabilityMap`. Candidate local and cross-component edges are checked
against the simulator navmesh, and accepted path geometry is stored in the
map. This is the current meaning of bridge discovery: it connects separated
reachable-point components when a verified navmesh path exists.

After extraction, Dijkstra and A* do not call the simulator. In particular,
`astar_search(traversability_map, start, goal)` operates only on the scene
representation. This boundary allows the ground-truth extractor to be
replaced later by an incrementally built traversability map without coupling
the planner to AI2-THOR.

The simulator is used again only when the waypoint follower executes the
planned actions and receives observations.

## Controller And Coordinates

Positions are `(x, y, z)` tuples in metres. `Pose` adds yaw in radians in the
shared navigation convention. Conversion to AI2-THOR's native yaw convention
is isolated in the simulator adapter and coordinate utilities.

The controller supports `move_forward`, `turn_left`, `turn_right`, and `stop`.
Movement distance, rotation resolution, heading tolerance, waypoint
tolerance, success distance, and step limit all come from the episode config.

## Recorded Artifacts

One episode directory contains:

- `scene-graph.json`: extracted symbolic graph and traversability map;
- `navigation-map.geojson`: floor and region geometry for inspection;
- `plan.json`: symbolic region plan and metric A* points;
- `trajectory.jsonl`: one aligned action, pose, collision state, RGB path, and
  depth path per observation;
- `rgb/`: lossless PNG observations;
- `depth/`: NumPy depth arrays at the configured simulator resolution;
- `summary.json`: execution success, termination reason, step and collision
  counts, distance, final error, pose, and backend metadata.

Generated episode data belongs under the ignored `data/` directory.

## Configuration

The single-episode template is
[`trajectory-following.json`](../../dso/configs/navigation/trajectory-following.json).
Its sections are:

- `paths`: corpus, manifest, scene directory, simulator config, and output;
- `episode`: scene ID, start room, goal room, and seed;
- `scene_graph`: rasterization resolution;
- `traversability_map`: reachable-point and edge-extraction parameters;
- `controller`: movement, rotation, tolerance, success, and step parameters.

Simulator rendering and executable settings remain in
[`ai2thor.json`](../../dso/configs/simulator/ai2thor.json), not in the scene
configuration.

## Verification

Run the lightweight repository checks with Pixi:

~~~bash
pixi run test
pixi run format
pixi run typecheck
~~~

Running the episode itself starts the local AI2-THOR executable and is an
integration run, not a lightweight unit check.
