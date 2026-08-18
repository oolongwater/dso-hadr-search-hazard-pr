"""Structured validators for canonical scene graphs. Never raises — returns ValidationReport."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from scene_graph.schema import ConnectorNode, EdgeType, Level, SceneGraph

_ID_PATTERN = re.compile(r"^(?:floor|room|conn|obj)_(?:[a-z0-9]+_)*\d+$")


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(BaseModel):
    code: str
    severity: Severity
    message: str
    node_id: str | None = None
    edge: dict[str, Any] | None = None


class ValidationReport(BaseModel):
    ok: bool = True
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    def add_error(self, code: str, message: str, *, node_id: str | None = None, edge: dict[str, Any] | None = None) -> None:
        self.ok = False
        self.errors.append(
            ValidationIssue(code=code, severity=Severity.ERROR, message=message, node_id=node_id, edge=edge)
        )

    def add_warning(self, code: str, message: str, *, node_id: str | None = None, edge: dict[str, Any] | None = None) -> None:
        self.warnings.append(
            ValidationIssue(code=code, severity=Severity.WARNING, message=message, node_id=node_id, edge=edge)
        )


def validate_scene_graph(sg: SceneGraph) -> ValidationReport:
    """Run all structural checks and return a structured report."""
    report = ValidationReport()
    node_ids = sg.node_ids()
    connectors_by_id: dict[str, ConnectorNode] = {c.id: c for c in sg.nodes.connector}
    region_ids = {r.id for r in sg.nodes.region}
    floor_ids = {f.id for f in sg.nodes.floor}

    _check_referential_integrity(sg, report, node_ids, connectors_by_id)
    _check_connected_by(sg, report, node_ids, connectors_by_id)
    _check_region_connectivity(sg, report, region_ids)
    _check_multi_floor(sg, report, floor_ids, connectors_by_id)
    _check_id_convention(sg, report)
    return report


def _check_referential_integrity(
    sg: SceneGraph,
    report: ValidationReport,
    node_ids: set[str],
    connectors_by_id: dict[str, ConnectorNode],
) -> None:
    for edge in sg.edges:
        if edge.src not in node_ids:
            report.add_error(
                "missing_node",
                f"edge src '{edge.src}' not found",
                edge={"src": edge.src, "dst": edge.dst, "type": edge.type.value},
            )
        if edge.dst not in node_ids:
            report.add_error(
                "missing_node",
                f"edge dst '{edge.dst}' not found",
                edge={"src": edge.src, "dst": edge.dst, "type": edge.type.value},
            )

    for conn in sg.nodes.connector:
        for endpoint in conn.gt.endpoints:
            if endpoint not in node_ids:
                report.add_error(
                    "missing_endpoint",
                    f"connector '{conn.id}' endpoint '{endpoint}' not found",
                    node_id=conn.id,
                )

    for region in sg.nodes.region:
        if region.gt.floor not in node_ids:
            report.add_error(
                "missing_floor_ref",
                f"region '{region.id}' references missing floor '{region.gt.floor}'",
                node_id=region.id,
            )

    for obj in sg.nodes.object:
        if obj.gt.region not in node_ids:
            report.add_error(
                "missing_region_ref",
                f"object '{obj.id}' references missing region '{obj.gt.region}'",
                node_id=obj.id,
            )
        if obj.gt.support_parent and obj.gt.support_parent not in node_ids:
            report.add_error(
                "missing_support_parent",
                f"object '{obj.id}' support_parent '{obj.gt.support_parent}' not found",
                node_id=obj.id,
            )


def _check_connected_by(
    sg: SceneGraph,
    report: ValidationReport,
    node_ids: set[str],
    connectors_by_id: dict[str, ConnectorNode],
) -> None:
    for edge in sg.edges:
        if edge.type != EdgeType.CONNECTED_BY:
            continue
        if not edge.via:
            report.add_error(
                "missing_via",
                f"connected_by edge {edge.src}->{edge.dst} missing via",
                edge={"src": edge.src, "dst": edge.dst, "type": edge.type.value},
            )
            continue
        if edge.via not in node_ids:
            report.add_error(
                "missing_via_node",
                f"connected_by via '{edge.via}' not found",
                edge={"src": edge.src, "dst": edge.dst, "type": edge.type.value, "via": edge.via},
            )
            continue
        conn = connectors_by_id.get(edge.via)
        if conn is None:
            report.add_error(
                "via_not_connector",
                f"connected_by via '{edge.via}' is not a connector node",
                edge={"src": edge.src, "dst": edge.dst, "type": edge.type.value, "via": edge.via},
            )
            continue
        expected = set(conn.gt.endpoints)
        actual = {edge.src, edge.dst}
        if expected != actual:
            report.add_error(
                "endpoint_mismatch",
                f"connected_by via '{edge.via}' endpoints {sorted(expected)} != edge {sorted(actual)}",
                edge={"src": edge.src, "dst": edge.dst, "type": edge.type.value, "via": edge.via},
            )


def _check_region_connectivity(sg: SceneGraph, report: ValidationReport, region_ids: set[str]) -> None:
    if len(region_ids) <= 1:
        return

    g = nx.Graph()
    for rid in region_ids:
        g.add_node(rid)
    for edge in sg.edges:
        if edge.type == EdgeType.CONNECTED_BY and edge.src in region_ids and edge.dst in region_ids:
            g.add_edge(edge.src, edge.dst)

    components = list(nx.connected_components(g))
    if len(components) <= 1:
        return

    primary = max(components, key=len)
    for comp in components:
        if comp == primary:
            continue
        for rid in sorted(comp):
            report.add_error(
                "disconnected_regions",
                f"disconnected regions: {rid} has no connector to the rest of the house",
                node_id=rid,
            )


def _check_multi_floor(
    sg: SceneGraph,
    report: ValidationReport,
    floor_ids: set[str],
    connectors_by_id: dict[str, ConnectorNode],
) -> None:
    if len(floor_ids) <= 1:
        return

    has_inter_floor = False
    for conn in connectors_by_id.values():
        if conn.gt.connector_type.value not in ("staircase", "passage"):
            continue
        ep0, ep1 = conn.gt.endpoints
        if ep0 in floor_ids and ep1 in floor_ids and ep0 != ep1:
            has_inter_floor = True
            break

    if not has_inter_floor:
        report.add_error(
            "missing_inter_floor_connector",
            "multi-floor scene requires a staircase or passage connector linking two floor nodes",
        )


def _check_id_convention(sg: SceneGraph, report: ValidationReport) -> None:
    for node in sg.iter_nodes():
        if not _ID_PATTERN.match(node.id):
            report.add_warning(
                "id_convention",
                f"node id '{node.id}' does not match <leveltag>_<name>_<idx> convention",
                node_id=node.id,
            )
