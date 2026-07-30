# Target Architecture

This document describes the intended architecture, not an implemented system.

The target flow is:

```text
hazard configuration
-> Python request
-> Unity hazard event
-> visual and physical/perceptual scene change
-> Unity metadata
-> dynamic ground-truth scene graph
-> route validation
-> replanning and waypoint execution
```

Python is expected to own experiment configuration, graph representation, planning, evaluation, and result serialization. Unity is expected to own the realized simulator state, hazard event execution, visible effects, physical scene changes, and authoritative hazard metadata exposed to Python.

Visual effects are not the authoritative source of graph state. Graph updates should be derived from simulator state and metadata designed for that purpose.

Structural connectivity, physical traversability, operational accessibility, and visibility are separate concepts and should not be collapsed into a single blocked/unblocked flag.
