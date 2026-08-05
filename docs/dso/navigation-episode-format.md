# Navigation Episode Format Template

This document is a design template for the future hazard-aware benchmark
episode and does not finalize that serialization. The current non-hazard
navigation runner and its concrete artifacts are documented in
[the navigation stack](navigation-stack.md).

## Candidate Fields

- Scene identifier: TBD.
- Start state: TBD.
- Goal state: TBD.
- Random seed: TBD.
- Hazard schedule: TBD.
- Initial graph: TBD.
- Graph updates: TBD.
- Original route: TBD.
- Replanned route: TBD.
- Success or failure: TBD.
- Metrics: TBD.
- Replay information: TBD.

## Open Decisions

- TBD: File format and schema ownership.
- TBD: Required versus optional fields.
- TBD: How Unity state references map to graph references.
- TBD: How failed route validation and replanning attempts are recorded.
- TBD: Which metrics are part of the milestone versus later evaluation.
