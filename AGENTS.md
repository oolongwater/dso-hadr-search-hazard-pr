# DSO HADR Agent Instructions

This repository is a DSO HADR Search project workspace built from an AI2-THOR source snapshot. Its purpose is to support research on hazard-aware search and navigation in simulated household environments.

## Current Milestone

The current milestone is repository setup and then implementation planning for parameterized fire, smoke, and earthquake events, Unity-rendered hazard effects, Unity-to-Python hazard metadata, symbolic scene-graph extraction and update, valid start-goal sampling, graph search, waypoint following, route validation, and replanning.

Do not implement those research systems unless the task explicitly asks for implementation work.

## Ownership Boundaries

- Yiqing owns research direction, milestones, interfaces, evaluation design, tracking, and reporting.
- Ewen owns Unity simulation effects, hazard visuals, physical scene changes, and Unity hazard metadata.
- Tianrun owns scene-graph design, graph extraction and updates, graph search, waypoint following, validation, and replanning.

## Upstream Versus DSO Code

- Treat `ai2thor/`, `doc/`, and most of `unity/Assets/` as upstream AI2-THOR code or assets.
- Put project-specific Python code under `dso/`.
- Put project-specific Unity work under `unity/Assets/DSOHADR/` whenever possible.
- Minimize changes to upstream AI2-THOR files. If an upstream file must change, document the reason, behavior impact, and test evidence in the pull request.

## Editing Rules

- Inspect relevant files before editing.
- Do not move, delete, rename, or reorganize upstream AI2-THOR files without explicit approval.
- Do not claim Unity behavior was tested unless Unity actually ran.
- Do not invent unresolved research decisions. Mark them `TBD` and link the relevant decision document.
- Keep generated Unity directories, builds, datasets, videos, checkpoints, credentials, private reports, DSO proposal PDFs, and review reports out of Git.

## Testing And Reporting

- Run lightweight Python checks for `dso/` changes when possible.
- Do not run Unity, graphical tests, or simulator asset downloads unless the task explicitly requires them.
- Report exact commands, results, and any checks that were not run.
