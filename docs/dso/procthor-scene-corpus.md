# ProcTHOR 100-Scene Corpus

## Current outcome

The current corpus is `dso-procthor-levels-1-3-100`. Its complete
generation contract lives in
`dso/configs/scenes/procthor-corpus.json`. It pins the ProcTHOR source
revision, AI2-THOR integration version and patch hash, generation seeds,
physical-clearance requirements, route-selection policy, and fixed split.

The corpus contains 31 one-level schema-1 houses, 34 two-level schema-2
houses, and 35 three-level schema-2 houses. Every accepted scene satisfies all
of these conditions:

- Unity house validation reports no warning.
- The complete exported physical navmesh has exactly one connected component.
- The navmesh is baked for a `0.28 m` agent radius while the runtime movement
  capsule reports a `0.201 m` collision sweep radius.
- No synthetic doorway, stair, landing, seam, or other navmesh link exists.
- The selected start-goal route has a geodesic distance of at least `10 m`.
- Multi-floor routes start and end on different floors.

For each scene, the selector evaluates every eligible pair of room
representatives and deterministically chooses the maximum finite geodesic.
Single-floor scenes consider every distinct room pair; multi-floor scenes
consider only cross-floor pairs. A candidate with disconnected topology or no
qualifying route is rejected as a whole. Navigation execution never resamples
the scene, endpoints, route, or waypoints.

The generated corpus has route distances from `10.7746897553 m` to
`56.0435536781 m`, with a mean of `27.7126347571 m`. All 69 multi-floor
episodes cross floors.

## Artifacts and fixed split

Generated houses remain outside Git under
`data/procthor/dso-procthor-levels-1-3-100/scenes/`. The checked-in
`dso/configs/scenes/procthor-scenes.json` manifest records every filename,
SHA-256 hash, schema, floor count, layout specification, and room/object
count. A changed, missing, malformed, or substituted house fails validation.

`generation-report.json` records the accepted generation seed, attempt count,
physical navmesh triangle/component counts, bake and movement radii, clearance
margin, rejected-candidate counts, and exact episode for every scene.

The configured split has 80 development scenes and 20 held-out
visualization/test scenes. The held-out subset is fixed and stratified into 6
one-level, 7 two-level, and 7 three-level houses. Use the configured IDs so
later experiments remain comparable.

## Download

For a normal checkout, mirror the human-readable data tree and validate the
scene bytes:

~~~bash
python3 dso/scripts/download_assets.py data
python3 dso/scripts/verify_runtime_assets.py --skip-build
~~~

The downloader uses rclone to copy plain JSON and GeoJSON files from
`Projects/HADR Navigation/data`. Download the 100 per-scene video,
trajectory, RGB, and depth tensor files with
`python3 dso/scripts/download_assets.py demo`. GitHub unit tests never invoke
the downloader.

## Regeneration from source

Build the integrated AI2-THOR player first, then run this from
`/media/run/Work/dso-hadr-search` with an empty output directory:

~~~bash
env PYTHONPATH="$PWD/dso/src" ../procthor/.venv/bin/python \
  dso/scripts/generate_procthor_corpus.py \
  dso/configs/scenes/procthor-corpus.json \
  --output-directory data/procthor/dso-procthor-levels-1-3-100/scenes \
  --manifest-output /tmp/scenes.json \
  --episodes-output /tmp/episodes.json
~~~

The process generates each configured scene once as an output task, while
trying deterministic candidate seeds up to the configured limit. It exits
nonzero and does not emit final manifests if any of the 100 tasks fails its
physical-connectivity, clearance, or long-horizon contract. Review the staged
manifests before replacing the checked-in manifest and episode files.

## Navigation verification

Verify scene bytes and the local player, then run all configured episodes once:

~~~bash
python3 dso/scripts/verify_runtime_assets.py
pixi run python dso/scripts/run_navigation_corpus.py \
  dso/configs/navigation/episodes.json \
  --output-directory data/navigation/all-scenes \
  --workers 6
pixi run python dso/scripts/encode_navigation_videos.py \
  data/navigation/all-scenes \
  --fps 10 \
  --workers 6 \
  --expected-scene-count 100
pixi run test
pixi run format
pixi run typecheck
~~~

The batch completed all 100 configured episodes in one run: 100 succeeded,
all 100 recorded trajectories stayed on the navmesh, and the total collision
count was zero. Executed distances range from `10.7746953369 m` to
`56.0478967807 m`, with a mean of `27.7010294769 m` across 18,796 primitive
steps. Generated RGB-D trajectories and reports stay under the ignored
`data/navigation/` tree. The corresponding video batch contains 100 validated
H.264 videos at 640 by 480 and 10 fps: 18,896 frames totaling 1,889.6 seconds.

## Interface boundary

The local executable, rendering settings, and quality are configured in
`dso/configs/simulator/ai2thor.json`. The simulator adapter loads a ProcTHOR
house, requests the complete Unity runtime navmesh, verifies that its bake
radius covers the runtime movement radius, and exposes primitive actions. It
does not choose endpoints or alternate actions after a failure.

`StepRecorder` writes one RGB image, one depth array, and one
`trajectory.jsonl` record per action. Each record contains scene and step IDs,
the issued action, action result, collision state, agent pose, camera horizon,
and observation paths. See the [navigation stack](navigation-stack.md) for the
planner and controller boundary.
