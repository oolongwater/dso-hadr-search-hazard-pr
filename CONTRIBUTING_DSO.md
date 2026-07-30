# Contributing To DSO HADR

Use one GitHub issue per task and one branch per issue. Do not push directly to `main`.

DSO-specific Python code belongs under `dso/`. DSO-specific Unity work should normally live under `unity/Assets/DSOHADR/`. Upstream AI2-THOR files should be changed only when necessary, and those changes must be documented in the pull request.

Run lightweight checks for the area you changed. Do not claim Unity behavior was tested unless Unity actually ran in the recorded editor version and environment.

Project-level decisions belong in `docs/dso/decisions/`. Unresolved research decisions should be marked `TBD` rather than silently decided in code.
