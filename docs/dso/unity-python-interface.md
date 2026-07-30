# Unity-Python Interface Template

This document is a design template for a future interface. It does not define final JSON, Python, or C# data models.

## Hazard Triggering

- TBD: How does Python trigger a fire, smoke, or earthquake event?
- TBD: Is the trigger an AI2-THOR action, a separate controller call, or another bridge mechanism?

## Parameters Sent To Unity

- TBD: Which fields identify hazard type, location, affected entities, timing, intensity, duration, random seed, and progression?
- TBD: How are invalid or unsupported parameters rejected?

## Unity State Returned To Python

- TBD: Which hazard state, object state, room state, passage state, and graph-relevant metadata are returned?
- TBD: Which returned fields are authoritative for graph updates?

## References To Rooms, Passages, And Objects

- TBD: How are affected rooms, passages, regions, objects, receptacles, and support surfaces referenced consistently across Unity and Python?

## Time And Event Progression

- TBD: How are event start time, duration, update ticks, and end state represented?

## Error Reporting

- TBD: How are unsupported hazards, invalid targets, parameter errors, Unity failures, and partial failures reported?

## Protocol Versioning

- TBD: How are interface changes versioned, reviewed, and tested across Unity and Python?
