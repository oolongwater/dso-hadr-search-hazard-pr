"""Run and record one configured navigation episode."""

import json
import sys
from pathlib import Path

from dso_hadr.config.nav_config import load_navigation_episode_config
from dso_hadr.controller.waypoint_follower import WaypointFollower
from dso_hadr.planner.grounding import select_goal_point
from dso_hadr.planner.motion.astar import astar_search
from dso_hadr.planner.symbolic.dijkstra import dijkstra_search
from dso_hadr.scenes.procthor import load_corpus_config, load_manifest
from dso_hadr.scenes.scene_graph import (
    extract_scene_graph_task,
    scene_agent_pose,
    scene_graph_to_dict,
    symbolic_plan_to_dict,
)
from dso_hadr.scenes.traversability import direct_navmesh_traversability
from dso_hadr.simulator.ai2thor import ProcTHORSimulator, load_simulator_config
from dso_hadr.simulator.ai2thor_backend import (
    AI2THORNavigationBackend,
    AI2THORNavigationConfig,
)
from dso_hadr.types.navigation import (
    NavigationAction,
    Observation,
)
from dso_hadr.utils.record_utils.navigation_episode import (
    prepare_navigation_episode_output,
    record_observation,
    write_json,
)

config = load_navigation_episode_config(Path(sys.argv[1]))
corpus_config = load_corpus_config(config.corpus_config_path)
manifest = load_manifest(config.scene_manifest_path, corpus_config)
records = {record.scene_id: record for record in manifest.scenes}
scene_paths = {
    scene_id: config.scenes_directory / record.filename for scene_id, record in records.items()
}

output_directory, trajectory_path = prepare_navigation_episode_output(config.output_directory)

simulator = ProcTHORSimulator(load_simulator_config(config.simulator_config_path))
backend = AI2THORNavigationBackend(
    simulator,
    scene_paths,
    AI2THORNavigationConfig(
        move_magnitude=config.move_magnitude,
        rotation_degrees=config.rotation_degrees,
    ),
)
with backend:
    initial_observation = backend.reset(
        config.scene_id,
        scene_agent_pose(scene_paths[config.scene_id], config.start_room_id),
        seed=config.seed,
    )
    task = extract_scene_graph_task(
        scene_paths[config.scene_id],
        direct_navmesh_traversability(
            backend.navmesh,
            move_magnitude=config.move_magnitude,
        ),
        start_room_id=config.start_room_id,
        goal_room_id=config.goal_room_id,
        meters_per_pixel=config.meters_per_pixel,
        navigation_map_path=output_directory / "navigation-map.geojson",
    )
    plan = dijkstra_search(task.graph, task.start_region_id, task.goal_region_id)
    goal_point = select_goal_point(
        task,
        plan.subgoals[-1].target_pose.position,
        config.navigable_tolerance,
    )
    write_json(output_directory / "scene-graph.json", scene_graph_to_dict(task))
    record_observation(
        output_directory=output_directory,
        trajectory_path=trajectory_path,
        scene_id=config.scene_id,
        step_id=0,
        action=None,
        observation=initial_observation,
    )
    motion_plan = astar_search(
        task.graph.traversability_map,
        backend.get_agent_pose().position,
        goal_point,
    )
    waypoints = motion_plan.points
    motion_distance = motion_plan.geodesic_distance
    write_json(
        output_directory / "plan.json",
        {
            "symbolic": symbolic_plan_to_dict(plan),
            "metric": {
                "planner": "astar",
                "geodesic_distance": motion_distance,
                "points": [list(point) for point in waypoints],
                "goal_point": list(goal_point),
            },
        },
    )

    def record_step(
        step_id: int,
        observation: Observation,
        action: NavigationAction,
    ) -> None:
        record_observation(
            output_directory=output_directory,
            trajectory_path=trajectory_path,
            scene_id=config.scene_id,
            step_id=step_id,
            action=action,
            observation=observation,
        )

    result = WaypointFollower(
        waypoint_tolerance=config.waypoint_tolerance,
        heading_tolerance_degrees=config.heading_tolerance_degrees,
        rotation_step_degrees=config.rotation_degrees,
    ).follow(
        backend,
        waypoints,
        success_distance=config.success_distance,
        max_steps=config.max_steps,
        on_step=record_step,
    )
    trajectory_on_navmesh = result.collisions == 0

summary: dict[str, object] = {
    "scene_id": config.scene_id,
    "start_room_id": config.start_room_id,
    "goal_room_id": config.goal_room_id,
    "backend": backend.get_scene_metadata(),
    "execution": {
        "success": result.success,
        "trajectory_on_navmesh": trajectory_on_navmesh,
        "termination_reason": result.termination_reason,
        "stop_called": result.stop_called,
        "steps": result.steps,
        "collisions": result.collisions,
        "traveled_distance": result.traveled_distance,
        "final_distance": result.final_distance,
        "final_pose": result.final_pose.as_list(),
    },
}
write_json(output_directory / "summary.json", summary)
print(json.dumps(summary, indent=2, sort_keys=True))
