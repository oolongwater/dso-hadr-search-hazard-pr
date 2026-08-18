# Changepoint kit

Portable, **stdlib-only** schema for decision nodes visited during hazard navigation demos. Copy this folder anywhere — no AI2-THOR, numpy, or OpenCV required.

## What is a changepoint?

A changepoint is a visited navigation decision node: world pose, door/room context, clutter cluster, traversable exits, controller decision, and artifact paths (`clip`, `payload_png`, `views`). Demos append records to `{label}.changepoints.json` as the agent reaches each node.

## JSON envelope

```json
{
  "schema_version": 1,
  "label": "scene_0040_cp",
  "house_json": "/path/to/scene.json",
  "updated_utc": "2026-08-16T07:00:00+00:00",
  "count": 1,
  "changepoints": [ { … } ]
}
```



## Field reference



### JSON envelope (file-level)


| Field            | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `schema_version` | Schema version number so readers know how to parse the file. |
| `label`          | Run/scene label for this episode (e.g. `scene_0040_cp`).     |
| `house_json`     | Path to the house/scene JSON this run was built from.        |
| `updated_utc`    | ISO timestamp of the last atomic write to this file.         |
| `count`          | Number of changepoint records in `changepoints`.             |
| `changepoints`   | Ordered list of visited decision-node snapshots.             |




### `Changepoint` fields


| Field                  | Description                                                                  |
| ---------------------- | ---------------------------------------------------------------------------- |
| `id`                   | SAM node id for this decision point (e.g. `cp_0`).                           |
| `world`                | Node position `{x, y, z}` in metres.                                         |
| `heading_deg`          | Node-frame yaw (which way “forward” is at this node).                        |
| `source`               | How the node was created (e.g. `doorway`, `cluster`, `manual`).              |
| `door_id`              | Doorway this changepoint sits at, if any.                                    |
| `room_ids`             | Rooms connected through this node/door.                                      |
| `passage_width_m`      | Estimated doorway clearance in metres.                                       |
| `clutter_score`        | Heuristic clutter density near the node.                                     |
| `block_score`          | Heuristic how likely this spot is to become blocked.                         |
| `cluster_object_ids`   | Object ids in the nearby clutter cluster.                                    |
| `cluster_object_types` | Object types matching `cluster_object_ids`.                                  |
| `cluster_type_summary` | Human-readable roll-up of cluster types.                                     |
| `connectivity`         | Text summary of room links, traversable exit count, and cluster.             |
| `decision`             | Controller action: `proceed`, `backtrack`, `reroute`, or `goal-unreachable`. |
| `decision_frame`       | Same decision expressed in the node’s local heading frame.                   |
| `blocked`              | Whether the passage at this node is considered blocked.                      |
| `exits`                | Outgoing SAM edges with traversability and path metrics (see below).         |
| `visit_index`          | Zero-based order this node was visited in the episode.                       |
| `phase`                | Lifecycle tag when recorded (e.g. `arrival`).                                |
| `agent`                | Agent pose `{x, y, z, yaw}` at visit time.                                   |
| `agent_path_m`         | Total distance the agent had walked when this was recorded.                  |
| `quake_active`         | Whether an earthquake was active at visit time.                              |
| `shake_elapsed_s`      | Seconds of shaking elapsed when recorded.                                    |
| `motion`               | Optional motion/context tag (often empty).                                   |
| `clip`                 | Path to a short video clip captured at this node.                            |
| `payload_png`          | Path to the decision-card / payload image.                                   |
| `views`                | Paths to multi-view stills (typically 4 yaw steps).                          |




### `ChangepointExit` fields (inside `exits[]`)


| Field         | Description                                                                   |
| ------------- | ----------------------------------------------------------------------------- |
| `src`         | Edge start node id.                                                           |
| `dst`         | Edge end node id.                                                             |
| `behaviour`   | Relative turn hint along this edge (`turn-left`, `go-forward`, `turn-right`). |
| `traversable` | Whether this edge is still passable after debris/blocking updates.            |
| `clearance_m` | Estimated clearance along this exit in metres.                                |
| `safety`      | Safety score for this exit (0–1).                                             |
| `visibility`  | Visibility score for this exit (0–1).                                         |




### Traversability

When debris or fallen objects paint the traversability mask and sever doorway edges, each exit’s `traversable` / `clearance_m` / `safety` / `visibility` drop (often to zero), which sets `blocked=true`, drives decisions like `backtrack` or `reroute`, and is reflected in `connectivity` (e.g. `0 traversable exit(s)`).

## Drop a changepoint at a coordinate

Requires a live AI2-THOR `controller` on a loaded ProcTHOR house (`pip install ai2thor`).

```python
from changepoint_kit import ChangepointLog
from changepoint_kit.thor import make_changepoint_at

log = ChangepointLog.open("run.changepoints.json")  # preserves existing records
cp = make_changepoint_at(
    controller,
    x=3.2,
    z=1.8,
    face=(4.0, 1.8),  # heading toward (tx, tz); or pass heading_deg=
    log=log,
)
print(cp.summary())
```

- `(x, z)` is snapped to the nearest **reachable** navmesh point (within 1 m by default); raises if none.
- Nearby movable objects within 1.5 m fill `cluster_object_ids` / `cluster_object_types`.
- Optional `room_ids` and `door_id` if you know them; otherwise left empty.
- Does **not** teleport the agent or capture images — record only.

Manual records use `source="manual"`, empty `exits`, and no SAM heuristic scores (`passage_width_m`, `clutter_score`, `block_score` stay at zero). Consumers should not expect traversability data on them.

## Integration rule (panel / JSON truth)

A changepoint record only reports blockage if the **scene graph is updated before the record is built**:

1. Sever doorway edges (`sever_edges_by_objects` / `_sever_target_doorway`)
2. Paint debris on the traversability mask
3. Recompute edge metrics and node connectivity
4. Set `node.blocked = True` and `decision = "backtrack"`
5. Then call `changepoint_from_sam_node` (or build `Changepoint` manually)

If you only set a local `passage_blocked` flag without updating the SAM node, the right panel and JSON will still show `proceed` / `blocked=False`.

