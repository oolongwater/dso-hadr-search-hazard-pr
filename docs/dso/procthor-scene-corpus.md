# ProcTHOR 100-Scene Corpus

## Stage 1 outcome

The Stage 1 corpus is dso-procthor-levels-1-3-100-v1. Its parameters live in
dso/configs/scenes/procthor-corpus.json rather than in Python code. The houses
were generated locally from /media/run/Work/procthor at revision
ec164a9b951213884841143fa69792370823d291 using its existing HouseGenerator and
patched schema-2 AI2-THOR executable.

Floor counts are randomly selected from 1, 2, and 3 with the configured seed.
The corpus has 31 one-level schema-1 houses, 34 two-level schema-2 houses, and
35 three-level schema-2 houses. No scene dataset was downloaded.

Generated houses remain outside Git under data/procthor/. The checked-in
dso/configs/scenes/procthor-scenes.json manifest records filenames, SHA-256
hashes, schemas, floor counts, layout specifications, and room/object counts.
A changed, missing, malformed, or substituted house fails validation.

## Configuration and fixed split

The corpus configuration owns:

- the dataset ID, generator repository, and generator revision;
- house seeds, random floor-count choices, and the floor-selection seed;
- the local Unity executable, output path, and generation resolution;
- the first scene index, scene count, filename template, and expected schemas;
- the visualization/test selection seed, floor targets, and complete ID list.

The configured split has 80 development scenes and 20 held-out
visualization/test scenes. The held-out subset is stratified into 6 one-level,
7 two-level, and 7 three-level houses.

Use the configured IDs rather than creating a new random split, so later
localization results are comparable and test scenes do not leak into
development.

## Commands

Run these commands from /media/run/Work/dso-hadr-search:

~~~bash
pixi run test
pixi run format
pixi run typecheck
~~~

Generation uses the local ProcTHOR checkout and local Unity executable; it does
not download scene JSON. If a generated house or selection parameter changes,
create and review a new dataset ID and manifest rather than silently replacing
this corpus.

## AI2-THOR interface

The local AI2-THOR executable, resolution, and quality are configured in
dso/configs/simulator/ai2thor.json. The interface in
dso_hadr.simulator.ai2thor owns the controller, loads a ProcTHOR JSON house,
and teleports the agent to the pose stored under metadata.agent. Callers supply
primitive AI2-THOR action dictionaries; the interface does not choose actions.

The StepRecorder in dso_hadr.simulator.recording writes one PNG under rgb/ and
one trajectory.jsonl record per action. Each record contains the scene ID, step
ID, action, action success, agent position and rotation, camera horizon, and RGB
path. Episode directories belong under the ignored data/ tree.

All 100 generated scenes were loaded through the simulator interface in the
local patched Unity build, teleported to their saved poses, and returned 300 by
300 RGB observations.

## Provenance and current boundary

Generator revision, house seed range, floor-selection seed, floor choices, and
Unity executable path are recorded in the corpus config. The exact generated
bytes remain the corpus identity and are locked by the manifest hashes.

This stage selects the 20 visualization/test houses and provides only the
simulator lifecycle and step-recording boundary. It does not select actions,
follow paths, or run localization. Evaluation through hadr_place_recognition
begins only after externally supplied movement trajectories are available.
