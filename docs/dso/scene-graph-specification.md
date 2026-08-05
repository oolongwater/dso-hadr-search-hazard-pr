# Scene Graph Specification

This document records candidate scientific and software design choices for the
hazard-aware symbolic scene graph. It does not finalize that graph structure.
Final decisions are TBD and should be jointly reviewed by Yiqing, Tianrun, and
Ewen. The implemented non-hazard baseline is documented separately in
[the navigation stack](navigation-stack.md).

## Purpose Of The Graph

TBD: Define the graph abstraction used for hazard-aware search and navigation.

## Intended Use In HADR Search And Navigation

Candidate use: represent rooms, regions, passages, objects, and hazards so the planner can reason about goals, routes, visibility, traversability, accessibility, and changes caused by hazards.

## Candidate Node Types

TBD candidate categories include room, region, passage, object, receptacle or support surface, agent state, goal, and hazard region.

## Candidate Edge Types

TBD candidate categories include adjacency, containment, support, passage connectivity, visibility, reachability, traversability, and task-specific accessibility.

## Hierarchy And Granularity

TBD: Decide how room-level, region-level, object-level, and waypoint-level graph elements relate to one another.

## Rooms, Regions, Passages, Objects, And Hazards

TBD: Define how simulator identifiers map to graph identifiers, and how hazard-affected entities are referenced.

## Static Versus Dynamic Graph State

Candidate static state includes geometry-derived connectivity and persistent object identity. Candidate dynamic state includes object poses, hazard state, visibility, traversability, accessibility, and support relations that can change during an episode.

## Structural Connectivity

TBD: Define when the structure of the environment itself changes, such as a passage becoming structurally disconnected in severe modeled earthquake cases.

## Physical Traversability

TBD: Define whether the agent can physically pass through a region, edge, or waypoint sequence.

## Operational Accessibility

TBD: Define whether a target, object, control, or passage can be practically used under hazard conditions, separate from geometric traversal.

## Visibility

TBD: Define line-of-sight, perception quality, and smoke-related observability semantics.

## Support Relations

TBD: Define object-on-object, object-on-surface, and stability/support changes after physical scene changes.

## Hazard-Related Node And Edge Updates

TBD: Specify how smoke, fire, and earthquake events update node and edge attributes without treating visual effects as authoritative state.

## Graph Extraction From Simulator Ground Truth

TBD: Define which Unity metadata and AI2-THOR ground-truth fields are required for extraction.

## Open Design Questions

- TBD: Final node taxonomy.
- TBD: Final edge taxonomy.
- TBD: Graph granularity.
- TBD: Attribute names and allowed values.
- TBD: How uncertainty or partial observability is represented.
- TBD: How route validation reports graph-state failures.

## Decisions Still Marked TBD

All sections above remain TBD until reviewed by Yiqing, Tianrun, and Ewen.
