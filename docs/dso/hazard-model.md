# Hazard Model

This document is a design scaffold. It does not claim physical realism and does not define implemented simulator behavior.

## Common Hazard-Event Concepts

TBD: Hazard events should have an identifier, affected location or entities, controllable parameters, timing, deterministic replay settings, visual effects, simulator-state effects, and metadata returned to Python.

Unity owns realized simulator state. Python owns experiment configuration, planning, evaluation, and result serialization.

## Fire

Candidate fire effects include visible flames, heat or danger state, reduced safe access, reduced visibility near the event, and operational accessibility changes. TBD: final parameters and affected state fields.

## Smoke

Candidate smoke effects include visible smoke volumes and perception or visibility changes. Smoke normally affects visibility and perception. TBD: final parameters and visibility semantics.

## Earthquake

Candidate earthquake effects include object pose changes, support relation changes, debris, physical traversability changes, and explicitly modeled severe structural connectivity changes. TBD: final severity levels and state changes.

## Controllable Parameters

TBD candidates include location, radius or affected region, intensity, duration, growth or decay model, start time, random seed, and affected object filters.

## Visual Effects

Visual rendering should communicate hazard state to users and recordings, but visual effects must not be treated as the authoritative source of graph state.

## Physical Or Perceptual Effects

TBD: Separate visual appearance from simulator metadata that represents traversability, accessibility, visibility, support relations, and hazard state.

## Expected Scene-State Changes

TBD: Define the Unity-owned state fields needed to describe affected rooms, passages, objects, and agent-relevant constraints.

## Expected Graph Changes

Smoke normally affects visibility. Fire normally affects visibility, hazard state, safety, and operational accessibility. Earthquake may affect object poses, support relations, physical traversability, and severe-case structural connectivity.

## Deterministic Replay And Random Seeds

TBD: Define seed handling, event order, time stepping, and replay metadata.

## Open Design Questions

- TBD: Final hazard parameter sets.
- TBD: State fields returned by Unity.
- TBD: Deterministic replay requirements.
- TBD: Which changes are authoritative metadata versus visual-only effects.
