from dso_hadr.graph.model import (
    ConnectivityEdge,
    ConnectivityKind,
    FloorNode,
    GraphEvidence,
    RegionKind,
    RegionNode,
    SceneGraph,
    TraversabilityMap,
    TraversabilitySource,
)
from dso_hadr.planner.symbolic.dijkstra import dijkstra_search
from dso_hadr.types.navigation import Pose


def _region(region_id: str, x: float) -> RegionNode:
    return RegionNode(
        id=region_id,
        label=region_id,
        category="room",
        kind=RegionKind.ROOM,
        floor_id="floor|0",
        navigation_pose=Pose(x, 0.0, 0.0, 0.0),
        bounds_xz=(x, 0.0, x + 1.0, 1.0),
        semantic_region_value=int(x) + 1,
        evidence=GraphEvidence.DATASET_SEMANTICS,
    )


def _edge(
    edge_id: str,
    node_a: str,
    node_b: str,
    x_a: float,
    x_b: float,
    cost: float,
) -> ConnectivityEdge:
    return ConnectivityEdge(
        id=edge_id,
        label=edge_id,
        node_a=node_a,
        node_b=node_b,
        kind=ConnectivityKind.DOOR,
        pose_a=Pose(x_a, 0.0, 0.0, 0.0),
        pose_b=Pose(x_b, 0.0, 0.0, 0.0),
        cost=cost,
        evidence=GraphEvidence.DATASET_SEMANTICS,
        evidence_detail=edge_id,
        supporting_entity_id=edge_id,
    )


def test_dijkstra_search_uses_local_scene_graph_types() -> None:
    graph = SceneGraph(
        scene_id="scene",
        floors=(
            FloorNode(
                id="floor|0",
                label="floor",
                scene_model="scene.json",
                level_index=0,
                evidence=GraphEvidence.DATASET_SEMANTICS,
            ),
        ),
        regions=(
            _region("room|a", 0.0),
            _region("room|b", 1.0),
            _region("room|c", 2.0),
        ),
        containment_edges=(),
        connectivity_edges=(
            _edge("edge|a-b", "room|a", "room|b", 0.4, 0.6, 1.0),
            _edge("edge|b-c", "room|b", "room|c", 1.4, 1.6, 1.0),
            _edge("edge|a-c", "room|a", "room|c", 0.4, 1.6, 3.0),
        ),
        traversability_map=TraversabilityMap(
            source=TraversabilitySource.AI2THOR_NAVMESH_GROUND_TRUTH,
            nodes=(),
            edges=(),
        ),
    )

    plan = dijkstra_search(graph, "room|a", "room|c")

    assert plan.region_ids == ("room|a", "room|b", "room|c")
    assert plan.edge_ids == ("edge|a-b", "edge|b-c")
    assert plan.total_cost == 2.0
    assert tuple(subgoal.region_id for subgoal in plan.subgoals) == plan.region_ids
    assert tuple(subgoal.target_pose.x for subgoal in plan.subgoals) == (
        0.0,
        0.6,
        1.6,
    )
