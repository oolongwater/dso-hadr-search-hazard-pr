"""Render canonical scene-graph timelines as standalone MP4 frames."""

from __future__ import annotations

from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D

from scene_graph.export import replay_timeline
from scene_graph.schema import EdgeType, Level, PassageState, SceneGraph

# BGR tuples from scene_graph.py converted to matplotlib RGB hex.
_STATE_RGB = {
    "broken": "#dc0000",
    "hot": "#ff7800",
    "cooked": "#b4b400",
    "moving": "#00c800",
    "open": "#0078c8",
    "normal": "#dcdcdc",
}
_HAZARD_RGB = {
    "none": "#404040",
    "smoke": "#888888",
    "fire": "#ff4400",
    "debris": "#aa6633",
    "structural": "#884400",
}
_PASSAGE_RGB = {
    PassageState.OPEN.value: "#00b450",
    PassageState.CLOSED.value: "#787878",
    PassageState.BLOCKED.value: "#dc5014",
    PassageState.CONSTRAINED.value: "#00a0dc",
}
_EDGE_STYLE = {
    EdgeType.CONTAINS.value: {"color": "#666666", "width": 0.6, "alpha": 0.35, "style": "solid"},
    EdgeType.SUPPORTS.value: {"color": "#b4b4b4", "width": 1.0, "alpha": 0.55, "style": "solid"},
    EdgeType.CONNECTED_BY.value: {"color": "#5096ff", "width": 1.4, "alpha": 0.8, "style": "solid"},
    EdgeType.BLOCKS.value: {"color": "#ff3232", "width": 1.6, "alpha": 0.9, "style": "dashed"},
    EdgeType.ADJACENT_TO.value: {"color": "#888888", "width": 0.8, "alpha": 0.4, "style": "dotted"},
    EdgeType.REACHABLE_FROM.value: {"color": "#888888", "width": 0.8, "alpha": 0.4, "style": "dotted"},
    EdgeType.SAME_REGION_AS.value: {"color": "#888888", "width": 0.8, "alpha": 0.4, "style": "dotted"},
}


def compute_graph_layout(sg: SceneGraph) -> dict[str, tuple[float, float]]:
    """Fixed shell layout by node level; reuse across all frames."""
    g = sg.to_networkx()
    shells: list[list[str]] = []
    for level in (Level.FLOOR.value, Level.REGION.value, Level.CONNECTOR.value, Level.OBJECT.value):
        shell = [nid for nid, attrs in g.nodes(data=True) if attrs.get("level") == level]
        if shell:
            shells.append(shell)
    if not shells:
        return nx.spring_layout(g, seed=0)
    return nx.shell_layout(g, nlist=shells)


def _node_color(node: Any) -> str:
    level = node.level.value if hasattr(node.level, "value") else str(node.level)
    if level == Level.FLOOR.value:
        state = node.obs.hazard_state.value if hasattr(node.obs.hazard_state, "value") else str(node.obs.hazard_state)
        return _HAZARD_RGB.get(state, _HAZARD_RGB["none"])
    if level == Level.REGION.value:
        severity = float(getattr(node.obs, "hazard_severity", 0.0) or 0.0)
        return mcolors.to_hex(cm.YlOrRd(min(1.0, max(0.0, severity))))
    if level == Level.CONNECTOR.value:
        state = node.obs.passage_state.value if hasattr(node.obs.passage_state, "value") else str(node.obs.passage_state)
        return _PASSAGE_RGB.get(state, _PASSAGE_RGB[PassageState.OPEN.value])
    if getattr(node.obs, "fallen", False) or getattr(node.obs, "state", None) == "broken":
        return _STATE_RGB["broken"]
    if getattr(node.obs, "state", None) == "hot":
        return _STATE_RGB["hot"]
    return _STATE_RGB["normal"]


def _node_label(node: Any) -> str:
    level = node.level.value if hasattr(node.level, "value") else str(node.level)
    if level == Level.FLOOR.value:
        return "floor"
    if level == Level.REGION.value:
        return str(node.gt.semantic_type)[:10]
    if level == Level.CONNECTOR.value:
        ctype = node.gt.connector_type.value if hasattr(node.gt.connector_type, "value") else str(node.gt.connector_type)
        return ctype[:8]
    return str(node.gt.category)[:8]


def _node_size(node: Any) -> float:
    level = node.level.value if hasattr(node.level, "value") else str(node.level)
    if level == Level.FLOOR.value:
        return 900.0
    if level == Level.REGION.value:
        return 700.0
    if level == Level.CONNECTOR.value:
        return 550.0
    return 120.0


def render_graph_frame(
    sg: SceneGraph,
    pos: dict[str, tuple[float, float]],
    *,
    title: str,
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    """Render one canonical graph frame as a BGR uint8 image."""
    g = sg.to_networkx()
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#181818")
    ax.set_facecolor("#181818")
    ax.axis("off")

    for edge_type, style in _EDGE_STYLE.items():
        edgelist = [
            (u, v)
            for u, v, attrs in g.edges(data=True)
            if attrs.get("type") == edge_type
        ]
        if not edgelist:
            continue
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=edgelist,
            ax=ax,
            edge_color=style["color"],
            width=style["width"],
            alpha=style["alpha"],
            style=style["style"],
            arrows=False,
        )

    for level in (Level.FLOOR.value, Level.REGION.value, Level.CONNECTOR.value, Level.OBJECT.value):
        nodes = [n for n in sg.iter_nodes() if n.level.value == level]
        if not nodes:
            continue
        nids = [n.id for n in nodes]
        node_colors = [_node_color(n) for n in nodes]
        node_sizes = [_node_size(n) for n in nodes]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=nids,
            node_color=node_colors,
            node_size=node_sizes,
            ax=ax,
            linewidths=0.5,
            edgecolors="#ffffff",
        )
        if level != Level.OBJECT.value:
            labels = {n.id: _node_label(n) for n in nodes}
            nx.draw_networkx_labels(
                g,
                pos,
                labels=labels,
                font_size=8,
                font_color="white",
                ax=ax,
            )

    ax.set_title(title, color="white", fontsize=12, pad=12)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_STATE_RGB["broken"], markersize=8, label="broken/fallen"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_STATE_RGB["hot"], markersize=8, label="hot"),
        Line2D([0], [0], color=_EDGE_STYLE[EdgeType.SUPPORTS.value]["color"], lw=2, label="supports"),
        Line2D([0], [0], color=_EDGE_STYLE[EdgeType.BLOCKS.value]["color"], lw=2, linestyle="--", label="blocks"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, facecolor="#222222", labelcolor="white")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    rgb = rgba[..., :3].copy()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bgr.shape[1] != width or bgr.shape[0] != height:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return bgr


def load_scenegraph_payload(path) -> dict[str, Any]:
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))


def payload_to_scenegraph(data: dict[str, Any], key: str) -> SceneGraph:
    return SceneGraph.model_validate(data[key])


def build_tick_graphs(payload: dict[str, Any]) -> tuple[list[SceneGraph], bool]:
    """Return per-tick graphs and whether a full timeline was available."""
    initial = payload_to_scenegraph(payload, "initial")
    timeline = payload.get("timeline")
    if timeline:
        return replay_timeline(initial, timeline), True
    final = payload_to_scenegraph(payload, "final")
    return [initial, final], False
