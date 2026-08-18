"""Changepoint record schema — stdlib only, safe to import without AI2-THOR or cv2."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class ChangepointExit:
    dst: str
    behaviour: str
    traversable: bool
    clearance_m: float = 0.0
    src: str = ""
    safety: float = 1.0
    visibility: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "behaviour": self.behaviour,
            "traversable": self.traversable,
            "clearance_m": round(self.clearance_m, 3),
            "safety": round(self.safety, 3),
            "visibility": round(self.visibility, 3),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChangepointExit:
        return cls(
            dst=str(data.get("dst") or ""),
            behaviour=str(data.get("behaviour") or ""),
            traversable=bool(data.get("traversable", True)),
            clearance_m=float(data.get("clearance_m") or 0.0),
            src=str(data.get("src") or ""),
            safety=float(data.get("safety") if data.get("safety") is not None else 1.0),
            visibility=float(data.get("visibility") if data.get("visibility") is not None else 1.0),
        )


@dataclass
class Changepoint:
    id: str
    world: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    heading_deg: float = 0.0
    source: str = "cluster"
    door_id: str | None = None
    room_ids: list[str] = field(default_factory=list)
    passage_width_m: float = 0.0
    clutter_score: float = 0.0
    block_score: float = 0.0
    cluster_object_ids: list[str] = field(default_factory=list)
    cluster_object_types: list[str] = field(default_factory=list)
    cluster_type_summary: str = ""
    connectivity: str = ""
    decision: str = ""
    decision_frame: str = ""
    blocked: bool = False
    exits: list[ChangepointExit] = field(default_factory=list)
    visit_index: int = 0
    phase: str = ""
    agent: dict[str, float] = field(default_factory=dict)
    agent_path_m: float = 0.0
    quake_active: bool = False
    shake_elapsed_s: float = 0.0
    motion: str = ""
    clip: str = ""
    payload_png: str = ""
    views: list[str] = field(default_factory=list)

    def cluster_counts(self) -> dict[str, int]:
        return dict(Counter(self.cluster_object_types))

    @property
    def cluster_size(self) -> int:
        return len(self.cluster_object_ids)

    def is_blocked(self) -> bool:
        return bool(self.blocked)

    def distance_to(self, x: float, z: float) -> float:
        wx = float(self.world.get("x", 0.0))
        wz = float(self.world.get("z", 0.0))
        return math.hypot(wx - x, wz - z)

    def traversable_exits(self) -> list[ChangepointExit]:
        return [e for e in self.exits if e.traversable]

    def summary(self) -> str:
        rooms = " <-> ".join(self.room_ids) if self.room_ids else "local"
        cluster = self.cluster_type_summary or ", ".join(
            f"{c}x {t}" if c > 1 else t for t, c in self.cluster_counts().most_common(3)
        )
        return (
            f"{self.id} @ ({self.world.get('x', 0):.1f}, {self.world.get('z', 0):.1f}) "
            f"rooms={rooms} cluster={cluster or 'none'} decision={self.decision or 'proceed'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "world": {k: round(float(v), 3) for k, v in self.world.items()},
            "heading_deg": round(self.heading_deg, 1),
            "source": self.source,
            "door_id": self.door_id,
            "room_ids": list(self.room_ids),
            "passage_width_m": round(self.passage_width_m, 3),
            "clutter_score": round(self.clutter_score, 3),
            "block_score": round(self.block_score, 3),
            "cluster_object_ids": list(self.cluster_object_ids),
            "cluster_object_types": list(self.cluster_object_types),
            "cluster_type_summary": self.cluster_type_summary,
            "connectivity": self.connectivity,
            "decision": self.decision,
            "decision_frame": self.decision_frame,
            "blocked": self.blocked,
            "exits": [e.to_dict() for e in self.exits],
            "visit_index": self.visit_index,
            "phase": self.phase,
            "agent": dict(self.agent),
            "agent_path_m": round(self.agent_path_m, 3),
            "quake_active": self.quake_active,
            "shake_elapsed_s": round(self.shake_elapsed_s, 2),
            "motion": self.motion,
            "clip": self.clip,
            "payload_png": self.payload_png,
            "views": list(self.views),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Changepoint:
        world = data.get("world") or {}
        agent = data.get("agent") or {}
        exits_raw = data.get("exits") or []
        return cls(
            id=str(data.get("id") or ""),
            world={
                "x": float(world.get("x", 0.0)),
                "y": float(world.get("y", 0.0)),
                "z": float(world.get("z", 0.0)),
            },
            heading_deg=float(data.get("heading_deg") or 0.0),
            source=str(data.get("source") or "cluster"),
            door_id=data.get("door_id"),
            room_ids=[str(r) for r in (data.get("room_ids") or [])],
            passage_width_m=float(data.get("passage_width_m") or 0.0),
            clutter_score=float(data.get("clutter_score") or 0.0),
            block_score=float(data.get("block_score") or 0.0),
            cluster_object_ids=[str(x) for x in (data.get("cluster_object_ids") or [])],
            cluster_object_types=[str(x) for x in (data.get("cluster_object_types") or [])],
            cluster_type_summary=str(data.get("cluster_type_summary") or ""),
            connectivity=str(data.get("connectivity") or ""),
            decision=str(data.get("decision") or ""),
            decision_frame=str(data.get("decision_frame") or ""),
            blocked=bool(data.get("blocked", False)),
            exits=[ChangepointExit.from_dict(e) for e in exits_raw],
            visit_index=int(data.get("visit_index") or 0),
            phase=str(data.get("phase") or ""),
            agent={k: float(v) for k, v in agent.items()},
            agent_path_m=float(data.get("agent_path_m") or 0.0),
            quake_active=bool(data.get("quake_active", False)),
            shake_elapsed_s=float(data.get("shake_elapsed_s") or 0.0),
            motion=str(data.get("motion") or ""),
            clip=str(data.get("clip") or ""),
            payload_png=str(data.get("payload_png") or ""),
            views=[str(v) for v in (data.get("views") or [])],
        )


class ChangepointLog:
    """Append-only changepoint log with atomic flush."""

    def __init__(
        self,
        path: str | Path,
        *,
        label: str = "",
        house_json: str = "",
    ) -> None:
        self.path = Path(path)
        self.label = label
        self.house_json = house_json
        self._records: list[Changepoint] = []

    def append(self, cp: Changepoint) -> None:
        cp.visit_index = len(self._records)
        self._records.append(cp)
        self.flush()

    @property
    def records(self) -> list[Changepoint]:
        return list(self._records)

    @classmethod
    def open(cls, path: str | Path) -> ChangepointLog:
        """Reopen an existing log, preserving prior records (empty log if absent)."""
        p = Path(path)
        if not p.is_file():
            return cls(p)
        data = json.loads(p.read_text(encoding="utf-8"))
        log = cls(
            p,
            label=str(data.get("label") or ""),
            house_json=str(data.get("house_json") or ""),
        )
        log._records = [
            Changepoint.from_dict(item) for item in (data.get("changepoints") or [])
        ]
        return log

    def flush(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "label": self.label,
            "house_json": self.house_json,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "count": len(self._records),
            "changepoints": [cp.to_dict() for cp in self._records],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


def load_changepoints(path: str | Path) -> list[Changepoint]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Changepoint.from_dict(item) for item in (data.get("changepoints") or [])]


def changepoint_fields() -> tuple[str, ...]:
    return tuple(f.name for f in fields(Changepoint))
