# Milestone 1

| # | Work package | Owner | Objective | Expected deliverable | Dependencies | Definition of done | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Repository and development environment | Yiqing | Establish shared scaffold and workflow. | DSO docs, workspace, templates, lightweight checks. | None | Scaffold exists and checks pass. | Not started |
| 2 | Shared graph and hazard terminology | Yiqing, Tianrun, Ewen | Align terms across Unity, Python, and planning. | Reviewed terminology and TBD list. | Work package 1 | Terms are documented and reviewed. | Not started |
| 3 | Common Unity hazard framework | Ewen | Create common structure for hazard events. | Unity hazard framework design and implementation. | Work packages 1-2 | Framework is reviewed and demonstrated. | Not started |
| 4 | Smoke effect and visibility update | Ewen, Tianrun | Model smoke visuals and visibility metadata/update. | Smoke event path and graph visibility update. | Work packages 2-3 | Smoke affects authoritative visibility metadata and graph update. | Not started |
| 5 | Fire effect and accessibility update | Ewen, Tianrun | Model fire visuals, hazard state, and accessibility update. | Fire event path and accessibility update. | Work packages 2-3 | Fire affects authoritative metadata and graph accessibility update. | Not started |
| 6 | Earthquake effect and traversability/connectivity update | Ewen, Tianrun | Model earthquake physical effects and graph update. | Earthquake event path and traversability/connectivity update. | Work packages 2-3 | Earthquake effects are represented in Unity metadata and graph update. | Not started |
| 7 | Static ground-truth graph extraction | Tianrun | Extract initial graph from simulator ground truth. | Static graph extractor. | Work package 2 | Extractor produces reviewed graph representation. | Not started |
| 8 | Dynamic graph update | Tianrun | Apply hazard-conditioned updates to the graph. | Dynamic graph update module. | Work packages 4-7 | Updates preserve separate visibility, traversability, accessibility, and connectivity semantics. | Not started |
| 9 | Start-goal sampling | Tianrun | Sample valid random start-goal pairs. | Start-goal sampler. | Work package 7 | Sampler validates candidates against graph and episode constraints. | Not started |
| 10 | Graph search and waypoint following | Tianrun | Plan and execute graph routes. | Search and waypoint-following components. | Work packages 7-9 | Route can be planned and followed in scoped test cases. | Not started |
| 11 | Route validation and replanning | Tianrun | Detect invalid routes and replan after graph changes. | Validation and replanning components. | Work packages 8-10 | Route invalidation and replanning are demonstrated. | Not started |
| 12 | Integrated milestone demonstration | Yiqing, Ewen, Tianrun | Demonstrate end-to-end milestone behavior. | Demo script, evidence, and report notes. | Work packages 1-11 | Demo shows hazard, graph update, validation, and replanning with documented limitations. | Not started |
