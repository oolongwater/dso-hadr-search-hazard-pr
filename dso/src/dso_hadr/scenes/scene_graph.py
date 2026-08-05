"""Extract the local scene graph from a ProcTHOR house description."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from dso_hadr.graph.model import (
    ConnectivityEdge,
    ConnectivityKind,
    ContainsEdge,
    FloorGrid,
    FloorNode,
    GraphEvidence,
    RegionKind,
    RegionNode,
    SceneGraph,
    SceneGraphTask,
    SymbolicPlan,
    TraversabilityMap,
)
from dso_hadr.utils.coordinates import native_yaw_to_navigation
from dso_hadr.types.navigation import Point3, Pose

_SINGLE_FLOOR_ID = "procthor_floor_0"


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return {str(key): entry for key, entry in value.items()}


def _records(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        raise TypeError("expected a JSON array")
    return tuple(_mapping(entry) for entry in value)


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError("expected a JSON number")
    return float(value)


def _point(value: object) -> Point3:
    document = _mapping(value)
    return _number(document["x"]), _number(document["y"]), _number(document["z"])


def _leaf_rooms(value: object) -> tuple[dict[str, object], ...]:
    leaves: list[dict[str, object]] = []
    for room in _records(value):
        children = room.get("rooms")
        if children:
            leaves.extend(_leaf_rooms(children))
        else:
            leaves.append(room)
    return tuple(leaves)


def _room_polygon(room: dict[str, object]) -> tuple[Point3, ...]:
    return tuple(_point(point) for point in _records(room["floorPolygon"]))


def _polygon_mask(
    polygon: tuple[Point3, ...],
    cell_x: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    cell_z: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.bool_]]:
    inside: np.ndarray[tuple[int, int], np.dtype[np.bool_]] = np.zeros(cell_x.shape, dtype=np.bool_)
    previous = polygon[-1]
    for current in polygon:
        x0, z0 = previous[0], previous[2]
        x1, z1 = current[0], current[2]
        if z0 != z1:
            crosses_row = (z0 > cell_z) != (z1 > cell_z)
            boundary_x = x0 + (cell_z - z0) * (x1 - x0) / (z1 - z0)
            inside ^= crosses_row & (cell_x < boundary_x)
        previous = current
    return inside


def _floor_grid(
    floor_id: str,
    room_ids: tuple[str, ...],
    polygons: tuple[tuple[Point3, ...], ...],
    surface_polygons: tuple[tuple[Point3, ...], ...],
    meters_per_pixel: float,
) -> FloorGrid:
    floor_height = polygons[0][0][1]
    minimum_x = min(point[0] for polygon in polygons for point in polygon)
    maximum_x = max(point[0] for polygon in polygons for point in polygon)
    minimum_z = min(point[2] for polygon in polygons for point in polygon)
    maximum_z = max(point[2] for polygon in polygons for point in polygon)
    width = max(1, math.ceil((maximum_x - minimum_x) / meters_per_pixel))
    height = max(1, math.ceil((maximum_z - minimum_z) / meters_per_pixel))
    origin_x = minimum_x + meters_per_pixel * 0.5
    origin_z = minimum_z + meters_per_pixel * 0.5
    columns = origin_x + np.arange(width, dtype=np.float64) * meters_per_pixel
    rows = origin_z + np.arange(height, dtype=np.float64) * meters_per_pixel
    cell_x, cell_z = np.meshgrid(columns, rows)

    semantic_regions = np.zeros((height, width), dtype=np.int32)
    for encoded, polygon in enumerate(polygons, start=1):
        semantic_regions[_polygon_mask(polygon, cell_x, cell_z)] = encoded
    traversable = semantic_regions > 0
    if surface_polygons:
        physical_surface = np.zeros((height, width), dtype=np.bool_)
        for polygon in surface_polygons:
            physical_surface |= _polygon_mask(polygon, cell_x, cell_z)
        traversable &= physical_surface
    if len(room_ids) != int(semantic_regions.max()):
        raise ValueError(f"floor {floor_id!r} has a room with no raster cells")
    return FloorGrid(
        floor_id=floor_id,
        traversable=traversable,
        semantic_regions=semantic_regions,
        meters_per_pixel=meters_per_pixel,
        origin_xz=(origin_x, origin_z),
        floor_height=floor_height,
    )


def _nearest_region_pose(
    grid: FloorGrid,
    semantic_value: int,
    target_xz: tuple[float, float],
) -> Pose:
    cells = np.argwhere((grid.semantic_regions == semantic_value) & grid.traversable)
    target_x, target_z = target_xz
    selected = min(
        ((int(cell[0]), int(cell[1])) for cell in cells),
        key=lambda cell: (
            (grid.grid_to_world(cell)[0] - target_x) ** 2
            + (grid.grid_to_world(cell)[2] - target_z) ** 2,
            cell,
        ),
    )
    return Pose(*grid.grid_to_world(selected), 0.0)


def _wall_segment(wall: dict[str, object]) -> tuple[Point3, Point3]:
    points = tuple(_point(point) for point in _records(wall["polygon"]))
    return points[0], points[1]


def _segment_key(
    segment: tuple[Point3, Point3],
) -> tuple[tuple[float, float], tuple[float, float]]:
    start, end = segment
    values = sorted(((start[0], start[2]), (end[0], end[2])))
    return values[0], values[1]


def _door_portal(
    door: dict[str, object],
    wall: dict[str, object],
) -> tuple[float, float]:
    start, end = _wall_segment(wall)
    wall_length = math.hypot(end[0] - start[0], end[2] - start[2])
    hole = tuple(_point(point) for point in _records(door["holePolygon"]))
    local_center = (min(point[0] for point in hole) + max(point[0] for point in hole)) * 0.5
    return (
        start[0] + (end[0] - start[0]) * local_center / wall_length,
        start[2] + (end[2] - start[2]) * local_center / wall_length,
    )


def _portal_edge(
    *,
    edge_id: str,
    label: str,
    node_a: str,
    node_b: str,
    kind: ConnectivityKind,
    portal_xz: tuple[float, float],
    supporting_entity_id: str,
    evidence_detail: str,
    grid: FloorGrid,
    encoded_by_room: dict[str, int],
    representative_by_room: dict[str, Pose],
) -> ConnectivityEdge:
    pose_a = _nearest_region_pose(grid, encoded_by_room[node_a], portal_xz)
    pose_b = _nearest_region_pose(grid, encoded_by_room[node_b], portal_xz)
    yaw = math.atan2(-(pose_b.x - pose_a.x), -(pose_b.z - pose_a.z))
    pose_a = Pose(*pose_a.position, yaw)
    pose_b = Pose(*pose_b.position, yaw)
    cost = (
        math.dist(representative_by_room[node_a].position, pose_a.position)
        + math.dist(pose_a.position, pose_b.position)
        + math.dist(pose_b.position, representative_by_room[node_b].position)
    )
    return ConnectivityEdge(
        id=edge_id,
        label=label,
        node_a=node_a,
        node_b=node_b,
        kind=kind,
        pose_a=pose_a,
        pose_b=pose_b,
        cost=max(cost, grid.meters_per_pixel),
        evidence=GraphEvidence.DATASET_SEMANTICS,
        evidence_detail=evidence_detail,
        supporting_entity_id=supporting_entity_id,
    )


def _horizontal_edges(
    walls: tuple[dict[str, object], ...],
    doors: tuple[dict[str, object], ...],
    room_ids: tuple[str, ...],
    grid: FloorGrid,
    encoded_by_room: dict[str, int],
    representative_by_room: dict[str, Pose],
) -> tuple[ConnectivityEdge, ...]:
    known_rooms = set(room_ids)
    wall_by_id = {str(wall["id"]): wall for wall in walls}
    edges: list[ConnectivityEdge] = []
    for door in sorted(doors, key=lambda item: str(item["id"])):
        door_id = str(door["id"])
        room_a = str(door["room0"])
        room_b = str(door["room1"])
        if room_a == room_b:
            continue
        node_a, node_b = sorted((room_a, room_b))
        edges.append(
            _portal_edge(
                edge_id=f"procthor_door_{door_id}",
                label=f"ProcTHOR door {room_a} ↔ {room_b}",
                node_a=node_a,
                node_b=node_b,
                kind=ConnectivityKind.DOOR,
                portal_xz=_door_portal(door, wall_by_id[str(door["wall0"])]),
                supporting_entity_id=door_id,
                evidence_detail=f"internal door {door_id!r} in generated house description",
                grid=grid,
                encoded_by_room=encoded_by_room,
                representative_by_room=representative_by_room,
            )
        )

    empty_walls: dict[
        tuple[tuple[float, float], tuple[float, float]],
        list[tuple[str, str]],
    ] = defaultdict(list)
    for wall in walls:
        room_id = str(wall["roomId"])
        if wall.get("empty") is True and room_id in known_rooms:
            empty_walls[_segment_key(_wall_segment(wall))].append((room_id, str(wall["id"])))
    for index, (segment, entries) in enumerate(sorted(empty_walls.items())):
        rooms = sorted({room_id for room_id, _wall_id in entries})
        if len(rooms) != 2:
            continue
        node_a, node_b = rooms
        wall_ids = sorted(wall_id for _room_id, wall_id in entries)
        edges.append(
            _portal_edge(
                edge_id=f"procthor_open_{index}_{node_a}_{node_b}",
                label=f"ProcTHOR open boundary {node_a} ↔ {node_b}",
                node_a=node_a,
                node_b=node_b,
                kind=ConnectivityKind.ADJACENT,
                portal_xz=(
                    (segment[0][0] + segment[1][0]) * 0.5,
                    (segment[0][1] + segment[1][1]) * 0.5,
                ),
                supporting_entity_id=wall_ids[0],
                evidence_detail="paired empty walls: " + ", ".join(wall_ids),
                grid=grid,
                encoded_by_room=encoded_by_room,
                representative_by_room=representative_by_room,
            )
        )
    return tuple(edges)


def _landing_center(landing: dict[str, object]) -> tuple[float, float]:
    polygon = tuple(_point(point) for point in _records(landing["polygon"]))
    return (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[2] for point in polygon) / len(polygon),
    )


def _vertical_edges(
    connectors: tuple[dict[str, object], ...],
    grids_by_floor: dict[str, FloorGrid],
    encoded_by_room: dict[str, int],
) -> tuple[ConnectivityEdge, ...]:
    edges: list[ConnectivityEdge] = []
    for connector in connectors:
        connector_id = str(connector["id"])
        lower_floor_id = str(connector["lowerFloorId"])
        upper_floor_id = str(connector["upperFloorId"])
        lower_room_id = str(connector["lowerRoomId"])
        upper_room_id = str(connector["upperRoomId"])
        landings = _records(connector["landingPolygons"])
        lower = next(landing for landing in landings if landing["floorId"] == lower_floor_id)
        upper = next(landing for landing in landings if landing["floorId"] == upper_floor_id)
        pose_a = _nearest_region_pose(
            grids_by_floor[lower_floor_id],
            encoded_by_room[lower_room_id],
            _landing_center(lower),
        )
        pose_b = _nearest_region_pose(
            grids_by_floor[upper_floor_id],
            encoded_by_room[upper_room_id],
            _landing_center(upper),
        )
        yaw = math.atan2(-(pose_b.x - pose_a.x), -(pose_b.z - pose_a.z))
        pose_a = Pose(*pose_a.position, yaw)
        pose_b = Pose(*pose_b.position, yaw)
        edges.append(
            ConnectivityEdge(
                id=f"procthor_stair_{connector_id}",
                label=f"ProcTHOR stair {lower_room_id} ↔ {upper_room_id}",
                node_a=lower_room_id,
                node_b=upper_room_id,
                kind=ConnectivityKind.CROSS_FLOOR,
                pose_a=pose_a,
                pose_b=pose_b,
                cost=math.dist(pose_a.position, pose_b.position),
                evidence=GraphEvidence.DATASET_SEMANTICS,
                evidence_detail=f"vertical connector {connector_id!r}",
                supporting_entity_id=str(connector["assetId"]),
            )
        )
    return tuple(edges)


def extract_scene_graph_task(
    scene_path: Path,
    traversability_map: TraversabilityMap,
    *,
    start_room_id: str,
    goal_room_id: str,
    meters_per_pixel: float,
    navigation_map_path: Path,
) -> SceneGraphTask:
    """Build one local graph and symbolic task from a ProcTHOR scene."""

    resolved_scene = scene_path.expanduser().resolve(strict=True)
    house = _mapping(json.loads(resolved_scene.read_text(encoding="utf-8")))
    metadata = _mapping(house["metadata"])
    schema = str(metadata["schema"])
    rooms = _leaf_rooms(house["rooms"])
    room_ids = tuple(str(room["id"]) for room in rooms)
    polygons = tuple(_room_polygon(room) for room in rooms)

    if schema == "2.0.0":
        floors = _records(house["floors"])
        floor_by_room = {str(room["id"]): str(room["floorId"]) for room in rooms}
        navigation_map_path.write_text(
            json.dumps(scene_navigation_map(resolved_scene), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        floors = ({"id": _SINGLE_FLOOR_ID, "index": 0, "baseY": polygons[0][0][1]},)
        floor_by_room = {room_id: _SINGLE_FLOOR_ID for room_id in room_ids}

    grids_by_floor: dict[str, FloorGrid] = {}
    encoded_by_room: dict[str, int] = {}
    representative_by_room: dict[str, Pose] = {}
    for floor in floors:
        floor_id = str(floor["id"])
        selected = tuple(
            (room_id, polygon)
            for room_id, polygon in zip(room_ids, polygons)
            if floor_by_room[room_id] == floor_id
        )
        selected_ids = tuple(room_id for room_id, _polygon in selected)
        selected_polygons = tuple(polygon for _room_id, polygon in selected)
        surfaces = (
            tuple(
                tuple(_point(point) for point in _records(surface["polygon"]))
                for surface in _records(floor["floorSurfaces"])
            )
            if schema == "2.0.0"
            else ()
        )
        grid = _floor_grid(
            floor_id,
            selected_ids,
            selected_polygons,
            surfaces,
            meters_per_pixel,
        )
        grids_by_floor[floor_id] = grid
        for encoded, (room_id, polygon) in enumerate(selected, start=1):
            encoded_by_room[room_id] = encoded
            representative_by_room[room_id] = _nearest_region_pose(
                grid,
                encoded,
                (
                    sum(point[0] for point in polygon) / len(polygon),
                    sum(point[2] for point in polygon) / len(polygon),
                ),
            )

    regions = tuple(
        RegionNode(
            id=room_id,
            label=str(room["roomType"]),
            category=str(room["roomType"]),
            kind=(
                RegionKind.HALLWAY
                if str(room["roomType"]).casefold() == "hallway"
                else RegionKind.ROOM
            ),
            floor_id=floor_by_room[room_id],
            navigation_pose=representative_by_room[room_id],
            bounds_xz=(
                min(point[0] for point in polygon),
                min(point[2] for point in polygon),
                max(point[0] for point in polygon),
                max(point[2] for point in polygon),
            ),
            semantic_region_value=encoded_by_room[room_id],
            evidence=GraphEvidence.DATASET_SEMANTICS,
        )
        for room_id, room, polygon in zip(room_ids, rooms, polygons)
    )
    containment = tuple(
        ContainsEdge(
            id=f"{floor_by_room[room_id]}_contains_{room_id}",
            floor_id=floor_by_room[room_id],
            region_id=room_id,
            evidence=GraphEvidence.DATASET_SEMANTICS,
        )
        for room_id in room_ids
    )
    all_walls = _records(house["walls"])
    all_doors = _records(house["doors"])
    connectivity: list[ConnectivityEdge] = []
    for floor in floors:
        floor_id = str(floor["id"])
        floor_room_ids = tuple(
            room_id for room_id in room_ids if floor_by_room[room_id] == floor_id
        )
        connectivity.extend(
            _horizontal_edges(
                tuple(wall for wall in all_walls if wall.get("floorId", floor_id) == floor_id),
                tuple(door for door in all_doors if door.get("floorId", floor_id) == floor_id),
                floor_room_ids,
                grids_by_floor[floor_id],
                encoded_by_room,
                representative_by_room,
            )
        )
    connectivity.extend(
        _vertical_edges(
            _records(house.get("verticalConnectors", [])),
            grids_by_floor,
            encoded_by_room,
        )
    )
    graph = SceneGraph(
        scene_id=resolved_scene.stem,
        floors=tuple(
            FloorNode(
                id=str(floor["id"]),
                label=(
                    f"ProcTHOR floor {int(_number(floor['index']))} (y={_number(floor['baseY']):.3f} m)"
                ),
                scene_model=str(resolved_scene),
                level_index=int(_number(floor["index"])),
                evidence=GraphEvidence.DATASET_SEMANTICS,
            )
            for floor in floors
        ),
        regions=regions,
        containment_edges=containment,
        connectivity_edges=tuple(sorted(connectivity, key=lambda edge: edge.id)),
        traversability_map=traversability_map,
    )
    region_by_id = {region.id: region for region in regions}
    region_by_id[start_room_id]
    region_by_id[goal_room_id]
    return SceneGraphTask(
        graph=graph,
        floor_grids=tuple(grids_by_floor.values()),
        start_region_id=start_room_id,
        goal_region_id=goal_room_id,
    )


def _closed_xz_ring(polygon: list[dict[str, float]]) -> list[list[float]]:
    ring = [[point["x"], point["z"]] for point in polygon]
    ring.append(list(ring[0]))
    return ring


def scene_navigation_map(scene_path: Path) -> dict[str, object]:
    """Build a review GeoJSON file from physical floor surfaces."""

    house = json.loads(scene_path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    features = [
        {
            "type": "Feature",
            "properties": {
                "kind": "logical_walkable",
                "floorId": surface["floorId"],
                "roomId": surface["roomId"],
                "surfaceId": surface["id"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [_closed_xz_ring(surface["polygon"])],
            },
        }
        for floor in house["floors"]
        for surface in floor["floorSurfaces"]
    ]
    return {"type": "FeatureCollection", "features": features}


def scene_agent_pose(scene_path: Path, start_room_id: str) -> Pose:
    """Read the stored agent pose in navigation coordinates."""

    house = json.loads(scene_path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    room = next(room for room in _leaf_rooms(house["rooms"]) if room["id"] == start_room_id)
    floor_height = (
        next(float(floor["baseY"]) for floor in house["floors"] if floor["id"] == room["floorId"])
        if house["metadata"]["schema"] == "2.0.0"
        else _number(_records(room["floorPolygon"])[0]["y"])
    )
    agent = house["metadata"]["agent"]
    return Pose(
        x=float(agent["position"]["x"]),
        y=floor_height,
        z=float(agent["position"]["z"]),
        yaw=native_yaw_to_navigation(float(agent["rotation"]["y"])),
    )


def scene_graph_to_dict(task: SceneGraphTask) -> dict[str, object]:
    """Return a JSON-compatible scene-graph record."""

    graph = task.graph
    return {
        "scene_id": graph.scene_id,
        "floors": [
            {
                "id": floor.id,
                "label": floor.label,
                "level_index": floor.level_index,
                "scene_model": floor.scene_model,
                "evidence": floor.evidence.value,
            }
            for floor in graph.floors
        ],
        "regions": [
            {
                "id": region.id,
                "label": region.label,
                "category": region.category,
                "kind": region.kind.value,
                "floor_id": region.floor_id,
                "navigation_pose": region.navigation_pose.as_list(),
                "bounds_xz": list(region.bounds_xz),
                "semantic_region_value": region.semantic_region_value,
                "evidence": region.evidence.value,
            }
            for region in graph.regions
        ],
        "containment_edges": [
            {
                "id": edge.id,
                "floor_id": edge.floor_id,
                "region_id": edge.region_id,
                "evidence": edge.evidence.value,
            }
            for edge in graph.containment_edges
        ],
        "connectivity_edges": [
            {
                "id": edge.id,
                "label": edge.label,
                "node_a": edge.node_a,
                "node_b": edge.node_b,
                "kind": edge.kind.value,
                "pose_a": edge.pose_a.as_list(),
                "pose_b": edge.pose_b.as_list(),
                "cost": edge.cost,
                "evidence": edge.evidence.value,
                "evidence_detail": edge.evidence_detail,
                "supporting_entity_id": edge.supporting_entity_id,
            }
            for edge in graph.connectivity_edges
        ],
        "traversability_map": {
            "source": graph.traversability_map.source.value,
            "nodes": [list(point) for point in graph.traversability_map.nodes],
            "edges": [
                {
                    "node_a": edge.node_a,
                    "node_b": edge.node_b,
                    "path": [list(point) for point in edge.path],
                    "cost": edge.cost,
                }
                for edge in graph.traversability_map.edges
            ],
        },
    }


def symbolic_plan_to_dict(plan: SymbolicPlan) -> dict[str, object]:
    """Return a JSON-compatible symbolic route."""

    return {
        "start_region_id": plan.start_region_id,
        "goal_region_id": plan.goal_region_id,
        "region_ids": list(plan.region_ids),
        "edge_ids": list(plan.edge_ids),
        "total_cost": plan.total_cost,
        "subgoals": [
            {
                "region_id": subgoal.region_id,
                "target_pose": subgoal.target_pose.as_list(),
                "incoming_edge_id": subgoal.incoming_edge_id,
            }
            for subgoal in plan.subgoals
        ],
    }


__all__ = [
    "extract_scene_graph_task",
    "scene_agent_pose",
    "scene_graph_to_dict",
    "scene_navigation_map",
    "symbolic_plan_to_dict",
]
