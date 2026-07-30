# DSO HADR Search Project

## Objective

Build a research simulator and benchmark workflow for hazard-aware household search and navigation using AI2-THOR as the simulator base.

## Current Milestone

The current milestone is Milestone 1: parameterized hazard events, visible and stateful hazard effects, Unity-to-Python hazard metadata, symbolic graph extraction and updates, start-goal sampling, graph search, waypoint following, route validation, replanning, and an integrated demonstration.

## Work Packages

| Work package | Owner | Expected deliverable | Key dependencies |
| --- | --- | --- | --- |
| Repository and development environment | Yiqing | DSO scaffold, workflow docs, lightweight Python checks | None |
| Shared graph and hazard terminology | Yiqing, Tianrun, Ewen | Reviewed terminology and TBD decision list | Repository scaffold |
| Unity hazard framework | Ewen | Common Unity hazard event structure | Interface agreement |
| Smoke effect and visibility update | Ewen, Tianrun | Smoke visuals and visibility metadata/update path | Hazard framework, graph terminology |
| Fire effect and accessibility update | Ewen, Tianrun | Fire visuals and accessibility metadata/update path | Hazard framework, graph terminology |
| Earthquake effect and traversability/connectivity update | Ewen, Tianrun | Earthquake scene changes and graph update path | Hazard framework, graph terminology |
| Static graph extraction | Tianrun | Ground-truth graph extractor | Graph specification |
| Dynamic graph update | Tianrun | Hazard-conditioned graph update logic | Hazard metadata, graph extractor |
| Start-goal sampling | Tianrun | Valid random start-goal sampler | Graph extractor |
| Search, waypoint following, validation, replanning | Tianrun | Navigation loop components | Graph update logic |
| Evaluation and reporting | Yiqing | Benchmark protocol and milestone report | Integrated demo outputs |

## Definitions Of Done

| Area | Definition of done |
| --- | --- |
| Documentation | Scope, owner, open decisions, and test expectations are recorded. |
| Unity work | Behavior is demonstrated in the stated Unity editor version or clearly marked untested. |
| Python work | Lightweight tests pass without requiring Unity or simulator asset downloads unless explicitly scoped. |
| Interface work | Ewen, Tianrun, and Yiqing review shared semantics before dependent work proceeds. |
| Milestone demo | Hazard event, graph update, route validation, and replanning are demonstrated together. |

## Current Tasks

| Task | Owner | Status |
| --- | --- | --- |
| Create repository scaffold | Yiqing | In progress |
| Review graph and hazard terminology | Yiqing, Tianrun, Ewen | Not started |
| Draft Unity-Python interface decisions | Yiqing, Ewen, Tianrun | Not started |

## Unresolved Project-Level Decisions

- TBD: Final graph node and edge types.
- TBD: Final hazard event parameter sets.
- TBD: Unity-Python protocol and versioning.
- TBD: Episode serialization format.
- TBD: Evaluation metrics and success criteria.

## Links

- [Upstream snapshot notes](docs/dso/upstream.md)
- [Target architecture](docs/dso/architecture.md)
- [Scene graph specification](docs/dso/scene-graph-specification.md)
- [Hazard model](docs/dso/hazard-model.md)
- [Unity-Python interface](docs/dso/unity-python-interface.md)
- [Navigation episode format](docs/dso/navigation-episode-format.md)
- [Team workflow](docs/dso/team-workflow.md)
- [Milestone 1](docs/dso/milestone-1.md)
