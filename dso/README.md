# DSO Python Workspace

This workspace contains DSO-specific Python package boundaries, tests, configuration, scripts, and examples.

The package currently contains scaffolding only. It does not implement simulator hazards, graph extraction, graph search, waypoint following, route validation, replanning, or AI2-THOR integration.

## Lightweight Checks

From this directory:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy src
```
