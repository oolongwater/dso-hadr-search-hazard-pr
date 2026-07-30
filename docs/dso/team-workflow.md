# Team Workflow

Use one GitHub issue per task and one branch per issue. Do not push directly to `main`.

Pull requests should link the issue, identify the workstream, list tests or demos run, and call out interface, graph-specification, or upstream AI2-THOR changes.

## Review Responsibilities

- Project planning and DSO documentation: Yiqing.
- Unity hazard code and assets: Ewen.
- Graph and navigation code: Tianrun.
- Unity-Python interface files: Ewen and Tianrun.

Shared interface changes require review from the owners of both sides of the interface. Graph-specification changes require review by Tianrun, and Ewen when Unity metadata semantics are affected.

Visual Unity effects should be demonstrated with screenshots or recordings and the Unity editor version. Task completion should be verified against the issue acceptance criteria and definition of done.

Project-level decisions should be recorded under `docs/dso/decisions/`. Blockers should be reported in the issue with the blocked dependency, owner, and next decision needed.
