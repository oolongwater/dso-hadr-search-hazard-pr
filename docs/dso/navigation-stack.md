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
  dso/configs/navigation/navigation.json
~~~

The episode configuration contains every path, scene selection, graph, and controller parameter used by the run. The loader does not
supply hidden defaults. `dso/configs/navigation/episodes.json` contains the
configured start and goal rooms for the 100-scene corpus.

## Execution Flow

```text
episode and simulator configs
            |
            v
load scene and export its runtime navmesh triangulation
            |
            v
extract floors, regions, connectivity, and triangle adjacency
            |
            v
Dijkstra search over symbolic region connectivity
            |
            v
one global A* search over the stored traversability map
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
| [`scenes/traversability.py`](../../dso/src/dso_hadr/scenes/traversability.py) | Convert the complete runtime navmesh triangulation and exact shared-edge adjacency into a traversability map, rejecting disconnected meshes. |
| [`planner/symbolic/dijkstra.py`](../../dso/src/dso_hadr/planner/symbolic/dijkstra.py) | Find a minimum-cost route through symbolic regions. |
| [`planner/motion/astar.py`](../../dso/src/dso_hadr/planner/motion/astar.py) | Find a metric route using only a `TraversabilityMap`, start point, and goal point. |
| [`controller/waypoint_follower.py`](../../dso/src/dso_hadr/controller/waypoint_follower.py) | Convert the metric route into configured forward and rotation actions. |
| [`simulator/navigation_backend.py`](../../dso/src/dso_hadr/simulator/navigation_backend.py) | Define the simulator-neutral execution interface. |
| [`simulator/ai2thor_backend.py`](../../dso/src/dso_hadr/simulator/ai2thor_backend.py) | Adapt that interface to AI2-THOR, request one runtime navmesh export per reset, and translate coordinate conventions. |
| [`NavMeshController.cs`](../../unity/Assets/DSOHADR/Runtime/Navigation/NavMeshController.cs) | Export the active Unity runtime navmesh triangulation and validated triangle adjacency without grid sampling. |
| [`utils/record_utils/navigation_episode.py`](../../dso/src/dso_hadr/utils/record_utils/navigation_episode.py) | Persist episode JSON, trajectory records, RGB frames, and depth arrays. |

## Scene Representation And Planner Boundary

`SceneGraph` holds two planning resolutions:

- symbolic floors, regions, containment edges, and connectivity edges for
  room-level planning;
- a `TraversabilityMap` of 3D points and path-bearing edges for motion
  planning across one or multiple levels.

AI2-THOR ground truth is used once per reset to export Unity's complete active
runtime navmesh. The export contains vertices, triangle indices, areas, exact
triangle adjacency, the bake radius, and the movement capsule radius. Two
triangles are adjacent only when their vertices, canonicalized to `1e-5` m,
share a complete physical edge. Neither Unity nor Python adds proximity,
doorway, landing, stair, or scene-semantic links.

Python verifies that every exported triangle belongs to one connected
component before planning. It then stores triangle-centroid nodes,
shared-edge portals, and dense edge geometry in the `TraversabilityMap`.
Scene semantics provide the symbolic room graph and region labels only; they
cannot repair or augment physical navmesh topology. Consequently, a broken
doorway or stair connection rejects the scene instead of being bridged in
code.

After extraction, Dijkstra and A* do not call the simulator. In particular,
`astar_search(traversability_map, start, goal)` operates only on the scene
representation. The episode runner performs one global start-to-goal A*
search, rather than concatenating independently grounded motion segments. A*
string-pulls each direct triangle corridor through its stored portals; portal
ends are inset using the existing movement resolution so the route remains in
the corridor interior, and the result is densified to that resolution.

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
- `video.mp4`: optional H.264 rendering of the recorded RGB sequence;
- `summary.json`: execution success, termination reason, step and collision
  counts, distance, final error, pose, and backend metadata.

Generated episode data belongs under the ignored `data/` directory.

## Configuration

The single-episode template is
[`navigation.json`](../../dso/configs/navigation/navigation.json).
Its sections are:

- `paths`: corpus, manifest, scene directory, simulator config, and output;
- `episode`: scene ID, start room, and goal room;
- `scene_graph`: rasterization resolution;
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

Run the complete configured corpus once with:

~~~bash
pixi run python dso/scripts/run_navigation_corpus.py \
  dso/configs/navigation/episodes.json \
  --output-directory data/navigation/all-scenes \
  --workers 6
~~~

The output directory must not already exist. The batch fails if any episode
fails, leaves the navmesh, or records a collision; it does not retry an
episode or substitute endpoints.

After a successful batch, encode and validate one video per episode with:

~~~bash
pixi run python dso/scripts/encode_navigation_videos.py \
  data/navigation/all-scenes \
  --fps 10 \
  --workers 6 \
  --expected-scene-count 100
~~~

The encoder refuses failed, colliding, off-navmesh, missing, or noncontiguous
trajectories. It verifies every output's codec, pixel format, frame rate, and
frame count, then writes `video-report.json` at the batch root.

To download the 100 complete demo folders without rerunning navigation:

~~~bash
python3 dso/scripts/download_assets.py demo
~~~

Each demo folder contains `video.mp4`, `trajectory.json`, `rgb/*.png`, and
`depth/*.npy`. The `data` selection separately mirrors the 107 human-readable
corpus and report JSON/GeoJSON files.
