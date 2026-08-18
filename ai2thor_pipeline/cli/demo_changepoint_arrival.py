#!/usr/bin/env python3
"""Single-changepoint arrival demo: FPV walk + gated decision card on scene_0040."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.demo_earthquake_multiroom import (
    _aabb_overlaps_rect,
    _obj_xz_bounds,
    _rect_xz_bounds,
    add_overhead_camera_at_floor,
    paint_debris_on_mask,
)
from cli.demo_scene_action_map import (
    CP_ARRIVE_M,
    PANEL_HEIGHT,
    _cp_for_door,
    _door_by_id,
    _ground_floor_base_y,
    _sever_target_doorway,
    _settle_scene_physics,
)
from cli.hazard_scenes import SCENE_CAPTURE_PROFILES
from core.changepoint import Changepoint, ChangepointLog, load_changepoints
from core.procthor_house import (
    default_local_executable,
    door_world_center,
    door_world_frame,
    doorway_corridor_rect,
    house_schema,
    load_house_json,
    make_procedural_controller,
    map_projection_from_props,
    reachable_on_floor,
    room_containing,
    stage_doorway_blockers,
)
from core.scene_action_map import (
    MIN_CLUSTER_OBJECTS,
    ControllerLabel,
    agent_radius_px,
    build_reachable_free_mask,
    build_scene_action_map,
    changepoint_from_sam_node,
    decision_frame_text,
    recompute_edges_from_mask,
    render_decision_card,
    render_node_payload_png,
)
from core.thor import (
    agent_pose,
    distance_xz,
    follow_path_discrete,
    get_reachable_positions,
    get_shortest_path,
    rotate_toward,
    snap_to_reachable,
    teleport,
    yaw_toward,
)
from core.video import finalize_mp4, probe_video, rgb_to_bgr_uint8
from hazard.functions import EarthquakeLatents, HazardConfig
from hazard.functions.earthquake import (
    earthquake_params_from_latents,
    start_earthquake,
    stop_earthquake,
)
from hazard.utils import (
    advance_physics,
    find_objects,
    hazard_output_dir,
    object_state_snapshot,
    objects_fallen,
    pause_physics,
    unpause_physics,
)

DEFAULT_HOUSE = (
    Path(__file__).resolve().parents[1] / "assets" / "houses" / "scene_0040.json"
)
DEFAULT_DOOR = "door|floor-0|2|3"
DEFAULT_START_ROOM = "room|2"
OUT_SUBDIR = "sam"
CARD_WIDTH = 700
SCAN_YAW_STEP = 90.0
SCAN_VIEWS = 4
PAUSE_SECONDS = 3.0
CORRIDOR_DEPTH_M = 0.45
PHYSICS_TIMESTEP = 1.0 / 30.0
QUAKE_STEPS = 450
SETTLE_STEPS = 90
PHYSICS_STEPS_PER_NAV_ACTION = 15
NAV_MOVE_M = 0.25
NAV_APPROACH_M = 0.05
NAV_ROTATE_DEG = 15.0
NAV_APPROACH_ROTATE_DEG = 5.0
SUSTAINED_BLOCK_STEPS = 15
MIN_OBJECTS_FALLEN = 5
REFERENCE_STAGED_IDS = ["3|5|1", "3|6", "2|4|1"]
STAGE_OFFSET_M = 0.35
STAGE_COUNT = 3
PRE_QUAKE_STANDOFF_M = 2.5


def _doorway_occupants(
    event,
    corridor: list[tuple[float, float]],
    staged_ids: list[str],
) -> list[str]:
    id_set = set(staged_ids)
    rect_bounds = _rect_xz_bounds(corridor)
    occupants: list[str] = []
    for obj in event.metadata.get("objects") or []:
        oid = str(obj.get("objectId") or "")
        if oid not in id_set:
            continue
        if obj.get("isMoving"):
            continue
        if _aabb_overlaps_rect(_obj_xz_bounds(obj), rect_bounds):
            occupants.append(oid)
    return occupants


def _doorway_blocking_ids(
    fallen_ids: list[str],
    staged_ids: list[str],
    corridor: list[tuple[float, float]],
    event,
) -> list[str]:
    staged_overlap = sorted(set(staged_ids) & set(fallen_ids))
    return staged_overlap or _doorway_occupants(event, corridor, fallen_ids)


class _ReachedStandoff(Exception):
    pass


class _BlockingDetector:
    def __init__(
        self,
        staged_ids: list[str],
        corridor: list[tuple[float, float]],
        *,
        sustained_steps: int = SUSTAINED_BLOCK_STEPS,
    ) -> None:
        self.staged_ids = staged_ids
        self.corridor = corridor
        self.sustained_steps = sustained_steps
        self._occupied_steps = 0
        self.blocked = False
        self.blocked_step: int | None = None

    def observe(self, step: int, event) -> bool:
        if self.blocked:
            return True
        occupants = _doorway_occupants(event, self.corridor, self.staged_ids)
        if occupants:
            self._occupied_steps += 1
        else:
            self._occupied_steps = 0
        if self._occupied_steps >= self.sustained_steps:
            self.blocked = True
            self.blocked_step = step
        return self.blocked


def _count_moving(event) -> int:
    return len(find_objects(event, predicate=lambda o: o.get("isMoving")))


def _nav_step_toward(
    controller,
    target_x: float,
    target_z: float,
    *,
    move_m: float = NAV_MOVE_M,
    rotate_deg: float = NAV_ROTATE_DEG,
) -> None:
    ax, az, ayaw = agent_pose(controller.last_event)
    desired = yaw_toward(ax, az, target_x, target_z)
    delta = (desired - ayaw + 180.0) % 360.0 - 180.0
    if abs(delta) > 1.0:
        if abs(delta) <= rotate_deg:
            rotate_toward(controller, desired, rotate_deg=rotate_deg)
        else:
            action = "RotateRight" if delta > 0 else "RotateLeft"
            controller.step(action=action, degrees=rotate_deg)
    controller.step(action="MoveAhead", moveMagnitude=move_m)


def _compose_two_panel(event, card: np.ndarray, *, card_width: int = CARD_WIDTH) -> np.ndarray | None:
    if event.frame is None:
        return None
    fpv_bgr = rgb_to_bgr_uint8(event.frame)
    h = PANEL_HEIGHT
    scale = h / fpv_bgr.shape[0]
    fpv_panel = cv2.resize(fpv_bgr, (int(fpv_bgr.shape[1] * scale), h))
    if card.shape[0] != h:
        card = cv2.resize(card, (card_width, h))
    divider = np.full((h, 4, 3), (60, 60, 60), dtype=np.uint8)
    return np.hstack([fpv_panel, divider, card])


def _compose_multiview_strip(
    views_bgr: list[np.ndarray],
    *,
    highlight_idx: int = 0,
    width: int = CARD_WIDTH,
    height: int = 124,
) -> np.ndarray:
    if not views_bgr:
        return np.zeros((height, width, 3), dtype=np.uint8)
    cell_w = width // max(1, min(len(views_bgr), 4))
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    for i, view in enumerate(views_bgr[:4]):
        resized = cv2.resize(view, (cell_w - 4, height - 8))
        x0 = i * cell_w + 2
        y0 = 4
        strip[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        color = (0, 255, 255) if i == highlight_idx % max(1, len(views_bgr)) else (80, 80, 80)
        cv2.rectangle(strip, (x0 - 2, y0 - 2), (x0 + resized.shape[1] + 2, y0 + resized.shape[0] + 2), color, 2)
    return strip


def _silent_scan_views(controller, start_yaw: float) -> tuple[list[np.ndarray], list[float], list[str]]:
    view_frames: list[np.ndarray] = []
    view_yaws: list[float] = []
    event = controller.last_event
    for i in range(SCAN_VIEWS):
        if i > 0:
            event = controller.step(action="RotateRight", degrees=SCAN_YAW_STEP)
        view_yaws.append(float(event.metadata["agent"]["rotation"]["y"]))
        view_frames.append(rgb_to_bgr_uint8(event.frame))
    rotate_toward(controller, start_yaw, rotate_deg=NAV_APPROACH_ROTATE_DEG)
    return view_frames, view_yaws, []


def _resolve_changepoint(sam, door_id: str, door_center: dict[str, float]) -> str:
    try:
        return _cp_for_door(sam, door_id)
    except ValueError:
        cp_ids = [nid for nid, n in sam.nodes.items() if n.kind == "changepoint"]
        if not cp_ids:
            raise RuntimeError(f"no changepoints for door {door_id}")
        return min(
            cp_ids,
            key=lambda nid: distance_xz(
                sam.nodes[nid].world[0],
                sam.nodes[nid].world[1],
                door_center["x"],
                door_center["z"],
            ),
        )


def _farthest_reachable_in_room(
    reachable: list[dict[str, float]],
    house: dict[str, Any],
    room_id: str,
    target_x: float,
    target_z: float,
) -> tuple[float, float, float]:
    best: tuple[float, float, float] | None = None
    best_d = -1.0
    for p in reachable:
        px, pz = float(p["x"]), float(p["z"])
        room = room_containing(house, px, pz)
        if room is None or str(room.get("id")) != room_id:
            continue
        d = distance_xz(px, pz, target_x, target_z)
        if d > best_d:
            best_d = d
            best = (px, float(p["y"]), pz)
    if best is None:
        raise RuntimeError(f"no reachable points in {room_id}")
    return best


def _parse_start(raw: str) -> tuple[float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--start expects x,z got {raw!r}")
    return float(parts[0]), float(parts[1])


def _right_panel_slice(frame: np.ndarray, card_width: int = CARD_WIDTH) -> np.ndarray:
    return frame[:, -card_width:, :]


def _panel_content_score(panel: np.ndarray) -> float:
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    return float(np.count_nonzero(gray > 40))


def run_changepoint_arrival_demo(
    controller,
    config: HazardConfig,
    *,
    house: dict[str, Any],
    out_video: Path,
    changepoints_json: Path,
    payload_png: Path,
    house_json: str,
    door_id: str,
    start_room_id: str,
    start_override: tuple[float, float] | None,
    grid_res_px: float,
    cp_threshold: float,
    min_sep_m: float,
) -> dict[str, Any]:
    profile = dict(SCENE_CAPTURE_PROFILES["earthquake"])
    fps = int(profile["fps"])

    base_y = _ground_floor_base_y(house)
    _settle_scene_physics(controller)
    add_overhead_camera_at_floor(controller, base_y)
    pass_event = controller.step(action="Pass")
    tpf = getattr(pass_event, "third_party_camera_frames", None)
    if not tpf:
        raise RuntimeError("overhead camera produced no third_party_camera_frames")
    oh_h, oh_w = tpf[0].shape[:2]

    map_props_event = controller.step(action="GetMapViewCameraProperties")
    map_props = map_props_event.metadata.get("actionReturn") or {}
    proj = map_projection_from_props(map_props, oh_w, oh_h)

    reachable = get_reachable_positions(controller)
    if house_schema(house) == "2.0.0":
        reachable = reachable_on_floor(reachable, base_y)

    h = int(proj["image_height"])
    w = int(proj["image_width"])
    free_open = build_reachable_free_mask(reachable, proj, h, w)
    radius_px = agent_radius_px(proj)

    door = _door_by_id(house, door_id)
    door_frame = door_world_frame(door, house)
    door_center = door_world_center(door, house)
    corridor = doorway_corridor_rect(door_frame, depth_m=CORRIDOR_DEPTH_M)
    rooms_by_id = {str(r.get("id")): r for r in (house.get("rooms") or []) if r.get("id")}
    stage_room = rooms_by_id.get("room|3") or rooms_by_id.get(start_room_id)
    if stage_room is None:
        raise RuntimeError(f"staging room missing for door {door_id}")
    staged_ids = stage_doorway_blockers(
        controller,
        door_frame,
        stage_room,
        door_id=door_id,
        count=STAGE_COUNT,
        offset_m=STAGE_OFFSET_M,
        spread_absolute=True,
        skip_preplaced=True,
        mass_tiebreak=True,
    )
    if not staged_ids:
        fallback_room = rooms_by_id.get(start_room_id)
        if fallback_room is not None and fallback_room is not stage_room:
            staged_ids = stage_doorway_blockers(
                controller,
                door_frame,
                fallback_room,
                door_id=door_id,
                count=STAGE_COUNT,
                offset_m=STAGE_OFFSET_M,
                spread_absolute=True,
                skip_preplaced=True,
                mass_tiebreak=True,
            )

    objects = list(controller.last_event.metadata.get("objects") or [])
    agent_y = float(controller.last_event.metadata["agent"]["position"]["y"])
    sam_probe = build_scene_action_map(
        free_open,
        proj,
        grid_res_px=grid_res_px,
        cp_threshold=cp_threshold,
        min_sep_m=min_sep_m,
        house=house,
        destinations=[],
        agent_radius_px=radius_px,
        objects=objects,
        world_y=agent_y,
    )
    cp_id = _resolve_changepoint(sam_probe, door_id, door_center)
    cp_probe = sam_probe.nodes[cp_id]
    cp_x, cp_z = float(cp_probe.world[0]), float(cp_probe.world[1])

    if start_override is not None:
        sx, sz = start_override
        snap = snap_to_reachable(sx, sz, reachable, max_dist=2.0)
        if snap is None:
            raise RuntimeError(f"--start ({sx},{sz}) not reachable")
        start_x, start_y, start_z = snap
    else:
        start_x, start_y, start_z = _farthest_reachable_in_room(
            reachable, house, start_room_id, cp_x, cp_z
        )
    start_dist_m = distance_xz(start_x, start_z, cp_x, cp_z)

    sam = build_scene_action_map(
        free_open,
        proj,
        grid_res_px=grid_res_px,
        cp_threshold=cp_threshold,
        min_sep_m=min_sep_m,
        house=house,
        destinations=[("start", (start_x, start_z))],
        agent_radius_px=radius_px,
        objects=objects,
        world_y=agent_y,
    )
    cp_id = _resolve_changepoint(sam, door_id, door_center)
    cp_node = sam.nodes[cp_id]
    cp_x, cp_z = float(cp_node.world[0]), float(cp_node.world[1])
    start_yaw = yaw_toward(start_x, start_z, cp_x, cp_z)
    teleport(controller, start_x, start_y, start_z, yaw=start_yaw)

    live = get_reachable_positions(controller)
    if house_schema(house) == "2.0.0":
        live = reachable_on_floor(live, base_y)
    goal_snap = snap_to_reachable(cp_x, cp_z, live, max_dist=2.0)
    if goal_snap is None:
        goal_snap = snap_to_reachable(door_center["x"], door_center["z"], live, max_dist=2.0)
    if goal_snap is None:
        raise RuntimeError(f"changepoint {cp_id} not reachable on navmesh")
    goal_x, goal_y, goal_z = goal_snap
    path, _ = get_shortest_path(controller, goal_x, goal_z, y=goal_y)
    if not path or len(path) < 2:
        raise RuntimeError(f"no navmesh path to changepoint {cp_id}")

    blank_card = render_decision_card(None, None, width=CARD_WIDTH, height=PANEL_HEIGHT)
    cp_log = ChangepointLog(changepoints_json, label=config.scene, house_json=house_json)
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    first_frame: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    agent_path_m = 0.0
    prev_ax, prev_az, _ = agent_pose(controller.last_event)
    peak_moving = 0
    passage_blocked = False
    blocked_step: int | None = None
    arrived = False
    arrival_label = ControllerLabel(
        node_id=cp_id,
        behaviour="go-forward",
        status="open",
        decision="proceed",
        decision_frame=decision_frame_text("proceed", cp_node),
        message="changepoint reached",
    )

    def _write_frame(event, card: np.ndarray) -> None:
        nonlocal writer, frames_written, first_frame, last_frame
        frame = _compose_two_panel(event, card, card_width=CARD_WIDTH)
        if frame is None:
            return
        if writer is None:
            fh, fw = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (fw, fh),
            )
        writer.write(frame)
        frames_written += 1
        if first_frame is None:
            first_frame = frame.copy()
        last_frame = frame.copy()

    def _at_or_past_standoff() -> bool:
        ax_now, az_now, _ = agent_pose(controller.last_event)
        return distance_xz(ax_now, az_now, door_center["x"], door_center["z"]) <= PRE_QUAKE_STANDOFF_M

    def _pre_quake_nav(event) -> None:
        nonlocal agent_path_m, prev_ax, prev_az
        ax_now, az_now, _ = agent_pose(controller.last_event)
        agent_path_m += distance_xz(prev_ax, prev_az, ax_now, az_now)
        prev_ax, prev_az = ax_now, az_now
        if _at_or_past_standoff():
            raise _ReachedStandoff()
        _write_frame(event, blank_card)

    pause_physics(controller)
    try:
        try:
            follow_path_discrete(
                controller,
                path,
                on_event=_pre_quake_nav,
                max_failures=12,
                max_consecutive_skips=8,
                max_steps=800,
                step_m=NAV_APPROACH_M,
                rotate_deg=NAV_APPROACH_ROTATE_DEG,
            )
        except _ReachedStandoff:
            pass
    finally:
        unpause_physics(controller)

    baseline = object_state_snapshot(controller.last_event)
    eq_params = earthquake_params_from_latents(config.severity, config.earthquake)
    pause_physics(controller)
    start_earthquake(controller, eq_params)
    detector = _BlockingDetector(staged_ids, corridor)
    nav_hold = False

    for step in range(QUAKE_STEPS):
        event = advance_physics(controller, PHYSICS_TIMESTEP)
        peak_moving = max(peak_moving, _count_moving(event))
        ax_now, az_now, _ = agent_pose(event)
        agent_path_m += distance_xz(prev_ax, prev_az, ax_now, az_now)
        prev_ax, prev_az = ax_now, az_now

        if step % PHYSICS_STEPS_PER_NAV_ACTION == 0 and not nav_hold:
            dist_goal = distance_xz(ax_now, az_now, goal_x, goal_z)
            if dist_goal <= CP_ARRIVE_M:
                arrived = True
                nav_hold = True
            else:
                _nav_step_toward(controller, goal_x, goal_z)

        if detector.observe(step, event):
            passage_blocked = True
            blocked_step = detector.blocked_step

        _write_frame(event, blank_card)

    stop_earthquake(controller)

    for step in range(SETTLE_STEPS):
        phys_step = QUAKE_STEPS + step
        event = advance_physics(controller, PHYSICS_TIMESTEP)
        peak_moving = max(peak_moving, _count_moving(event))
        if not passage_blocked:
            if detector.observe(phys_step, event):
                passage_blocked = True
                blocked_step = detector.blocked_step
        _write_frame(event, blank_card)

    unpause_physics(controller)

    if not arrived:
        path, _ = get_shortest_path(controller, goal_x, goal_z, y=goal_y)
        if path and len(path) >= 2:

            def _post_quake_nav(event) -> None:
                nonlocal agent_path_m, prev_ax, prev_az
                ax_now, az_now, _ = agent_pose(controller.last_event)
                agent_path_m += distance_xz(prev_ax, prev_az, ax_now, az_now)
                prev_ax, prev_az = ax_now, az_now
                _write_frame(event, blank_card)

            pause_physics(controller)
            try:
                follow_path_discrete(
                    controller,
                    path,
                    on_event=_post_quake_nav,
                    max_failures=12,
                    max_consecutive_skips=8,
                    max_steps=400,
                    step_m=NAV_APPROACH_M,
                    rotate_deg=NAV_APPROACH_ROTATE_DEG,
                )
            finally:
                unpause_physics(controller)

    if not arrived:
        ax_now, az_now, _ = agent_pose(controller.last_event)
        arrived = distance_xz(ax_now, az_now, goal_x, goal_z) <= max(CP_ARRIVE_M * 2.0, 1.5)
    if not arrived:
        raise RuntimeError(f"agent did not reach changepoint {cp_id}")

    ax_now, az_now, _ = agent_pose(controller.last_event)
    agent_path_m = max(
        agent_path_m,
        distance_xz(start_x, start_z, ax_now, az_now),
    )

    after = object_state_snapshot(controller.last_event)
    fallen_ids = objects_fallen(baseline, after)
    num_objects_fallen = len(fallen_ids)
    blocking_ids = _doorway_blocking_ids(
        fallen_ids, staged_ids, corridor, controller.last_event
    )
    fallen_door_overlap = len(blocking_ids)

    arrival_event = controller.last_event
    if passage_blocked:
        _sever_target_doorway(
            sam, arrival_event, blocking_ids, proj, door_id, cp_id
        )
        active_mask = paint_debris_on_mask(
            free_open, arrival_event, blocking_ids, proj
        )
        live_objects = list(arrival_event.metadata.get("objects") or [])
        recompute_edges_from_mask(
            sam, active_mask, proj, objects=live_objects, agent_radius_px=radius_px
        )
        cp_node.blocked = True
        arrival_label = ControllerLabel(
            node_id=cp_id,
            behaviour="",
            status="blocked",
            decision="backtrack",
            decision_frame=decision_frame_text("backtrack", cp_node),
            message="passage blocked",
        )
    ax_now, az_now, ayaw_now = agent_pose(arrival_event)
    ay_now = float(arrival_event.metadata["agent"]["position"]["y"])
    arrival_yaw = ayaw_now
    view_frames, view_yaws, _ = _silent_scan_views(controller, arrival_yaw)

    out_dir = payload_png.parent
    label_stem = payload_png.stem.replace("_payload", "")
    view_paths: list[str] = []
    for i, (view, yaw) in enumerate(zip(view_frames, view_yaws, strict=False)):
        view_path = out_dir / f"{label_stem}_view_{int(i * SCAN_YAW_STEP)}.png"
        cv2.imwrite(str(view_path), view)
        view_paths.append(str(view_path))

    clip_path = out_dir / f"{label_stem}_cp_clip.mp4"
    if view_frames:
        vh, vw = view_frames[0].shape[:2]
        clip_writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (vw, vh),
        )
        for vf in view_frames:
            for _ in range(max(1, fps // 3)):
                clip_writer.write(vf)
        clip_writer.release()
        finalize_mp4(clip_path)

    shake_el = QUAKE_STEPS * PHYSICS_TIMESTEP
    cp = changepoint_from_sam_node(
        cp_node,
        sam,
        label=arrival_label,
        phase="arrival",
        agent={"x": ax_now, "y": ay_now, "z": az_now, "yaw": ayaw_now},
        agent_path_m=agent_path_m,
        quake_active=True,
        shake_elapsed_s=shake_el,
        visit_index=0,
    )
    cp.clip = str(clip_path) if clip_path.is_file() else ""
    cp.views = view_paths
    cp_node.clip = cp.clip
    cp_node.views = view_paths

    payload_img = render_node_payload_png(
        cp_node,
        arrival_label,
        view_frames,
        view_yaws=view_yaws,
    )
    cv2.imwrite(str(payload_png), payload_img)
    cp.payload_png = str(payload_png)
    cp_node.payload_png = str(payload_png)
    cp_log.append(cp)

    hold_frames = round(PAUSE_SECONDS * fps)
    cycle_every = max(1, fps // 3)
    hold_cards: list[np.ndarray] = []
    for i in range(hold_frames):
        event = controller.step(action="Pass")
        highlight = (i // cycle_every) % len(view_frames)
        header = _compose_multiview_strip(view_frames, highlight_idx=highlight)
        card = render_decision_card(
            cp_node,
            arrival_label,
            width=CARD_WIDTH,
            height=PANEL_HEIGHT,
            cp=cp,
            header_image=header,
        )
        hold_cards.append(card)
        _write_frame(event, card)

    if writer is not None:
        writer.release()
    finalize_mp4(out_video)

    return {
        "video_path": str(out_video),
        "changepoints_json": str(changepoints_json),
        "payload_png": str(payload_png),
        "sam": sam,
        "objects": objects,
        "cp_id": cp_id,
        "door_id": door_id,
        "staged_ids": staged_ids,
        "start_pos": {"x": start_x, "y": start_y, "z": start_z},
        "start_dist_m": start_dist_m,
        "agent_path_m": agent_path_m,
        "frames_written": frames_written,
        "hold_frames": hold_frames,
        "peak_num_moving": peak_moving,
        "passage_blocked": passage_blocked,
        "blocked_step": blocked_step,
        "num_objects_fallen": num_objects_fallen,
        "fallen_door_overlap": fallen_door_overlap,
        "blocking_ids": blocking_ids,
        "fallen_ids": fallen_ids,
        "changepoint_scan_count": len(cp_log.records),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "hold_cards": hold_cards,
        "cp": cp,
    }


def self_check(result: dict[str, Any]) -> None:
    video_path = Path(result["video_path"])
    probe = probe_video(video_path)
    assert result["frames_written"] > 0, "no frames written"
    assert probe.get("frame_count", 0) > 0, f"unreadable mp4: {video_path}"

    cp_json = Path(result["changepoints_json"])
    assert cp_json.is_file(), f"missing changepoints json: {cp_json}"
    loaded = load_changepoints(cp_json)
    assert len(loaded) == 1, f"expected 1 changepoint, got {len(loaded)}"
    cp = loaded[0]
    assert cp.visit_index == 0, f"visit_index should be 0, got {cp.visit_index}"
    assert cp.door_id == result["door_id"], f"door_id mismatch: {cp.door_id}"
    assert "FloorLamp" in cp.cluster_object_types, (
        f"changepoint should sit on lamp cluster, got {cp.cluster_object_types}"
    )
    staged = result.get("staged_ids") or []
    assert staged == REFERENCE_STAGED_IDS, f"expected {REFERENCE_STAGED_IDS}, got {staged}"
    assert cp.clip and Path(cp.clip).is_file(), f"missing clip: {cp.clip}"
    assert len(cp.views) >= SCAN_VIEWS, f"expected >={SCAN_VIEWS} views, got {len(cp.views)}"
    for vp in cp.views:
        assert Path(vp).is_file(), f"missing view: {vp}"
    dest_edges = [e for e in cp.exits if "dest_start" in (e.src, e.dst)]
    assert dest_edges, "expected dest_start edge in changepoint exits"
    if result["passage_blocked"]:
        assert cp.blocked, f"changepoint should be blocked, got blocked={cp.blocked}"
        assert not cp.traversable_exits(), (
            f"blocked changepoint should have no traversable exits, got {cp.traversable_exits()}"
        )
        assert cp.decision == "backtrack", f"expected backtrack, got {cp.decision}"
        assert "0 traversable exit(s)" in cp.connectivity, (
            f"connectivity should report 0 exits: {cp.connectivity}"
        )
        assert all(not e.traversable for e in dest_edges), (
            "dest_start edges should be severed when passage blocked"
        )
    assert cp.cluster_size >= MIN_CLUSTER_OBJECTS, (
        f"cluster too small: {cp.cluster_size} < {MIN_CLUSTER_OBJECTS}"
    )
    round_trip = Changepoint.from_dict(cp.to_dict())
    assert round_trip.to_dict() == cp.to_dict(), "changepoint round-trip mismatch"
    assert Path(result["payload_png"]).is_file(), "payload png missing"

    assert result["start_dist_m"] > 4.0, (
        f"start should be far from CP, got {result['start_dist_m']:.2f}m"
    )
    assert result["agent_path_m"] > 3.0, (
        f"agent should walk meaningful distance, got {result['agent_path_m']:.2f}m"
    )
    assert result["hold_frames"] >= round(PAUSE_SECONDS * 30) - 1, "hold segment too short"
    assert result["peak_num_moving"] > 0, (
        f"quake must move objects, got peak_num_moving={result['peak_num_moving']}"
    )
    assert result["passage_blocked"], "passage should be blocked by fallen objects"
    assert result["num_objects_fallen"] >= MIN_OBJECTS_FALLEN, (
        f"expected >={MIN_OBJECTS_FALLEN} fallen objects, got {result['num_objects_fallen']}"
    )
    assert result["fallen_door_overlap"] >= 1, (
        f"expected fallen objects in doorway, got {result['fallen_door_overlap']}"
    )

    first = result.get("first_frame")
    last = result.get("last_frame")
    assert first is not None and last is not None, "missing frame snapshots"
    first_score = _panel_content_score(_right_panel_slice(first))
    last_score = _panel_content_score(_right_panel_slice(last))
    assert last_score > first_score * 2.0, (
        f"right panel should fill on arrival (first={first_score:.0f}, last={last_score:.0f})"
    )

    hold_cards = result.get("hold_cards") or []
    assert len(hold_cards) >= 2, "need hold cards for strip cycle check"
    strip0 = hold_cards[0][:124, :CARD_WIDTH, :]
    strip1 = hold_cards[min(len(hold_cards) - 1, max(1, len(hold_cards) // 3))][:124, :CARD_WIDTH, :]
    assert not np.array_equal(strip0, strip1), "localisation strip should cycle during hold"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="scene_0040_cp")
    parser.add_argument("--house", type=Path, default=DEFAULT_HOUSE)
    parser.add_argument("--door", default=DEFAULT_DOOR)
    parser.add_argument("--start-room", default=DEFAULT_START_ROOM)
    parser.add_argument(
        "--start",
        default="",
        help="Override start position as x,z (default: farthest reachable in --start-room)",
    )
    parser.add_argument("--grid-res", type=float, default=0.75)
    parser.add_argument("--cp-threshold", type=float, default=0.55)
    parser.add_argument("--cp-min-sep", type=float, default=1.2)
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--local-executable", type=Path, default=default_local_executable())
    args = parser.parse_args()

    if not args.house.is_file():
        raise RuntimeError(f"house json not found: {args.house}")
    if not args.local_executable.is_file():
        raise RuntimeError(
            f"custom executable not found: {args.local_executable}\n"
            "Run: ./ai2thor_custom/build_local.sh"
        )

    house = load_house_json(args.house)
    out_dir = hazard_output_dir() / "earthquake" / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{args.label}.mp4"
    changepoints_json = out_dir / f"{args.label}.changepoints.json"
    payload_png = out_dir / f"{args.label}_payload.png"
    start_override = _parse_start(args.start) if args.start.strip() else None

    config = HazardConfig(
        hazard_type="earthquake",
        scene=args.label,
        severity=args.severity,
        seed=args.seed,
        native_effects=True,
        earthquake=EarthquakeLatents(),
    )

    controller = make_procedural_controller(
        house,
        headless=args.headless,
        width=args.width,
        height=args.height,
        render_depth=False,
        local_executable_path=str(args.local_executable),
    )

    try:
        map_props_event = controller.step(action="GetMapViewCameraProperties")
        map_props = map_props_event.metadata.get("actionReturn") or {}
        mpp_x = (2.0 * float(map_props.get("orthographicSize", 3.0))) / float(args.width)
        grid_res_px = max(4.0, args.grid_res / mpp_x)

        result = run_changepoint_arrival_demo(
            controller,
            config,
            house=house,
            out_video=out_video,
            changepoints_json=changepoints_json,
            payload_png=payload_png,
            house_json=str(args.house),
            door_id=args.door,
            start_room_id=args.start_room,
            start_override=start_override,
            grid_res_px=grid_res_px,
            cp_threshold=args.cp_threshold,
            min_sep_m=args.cp_min_sep,
        )
        self_check(result)
    finally:
        controller.stop()

    summary = {
        "label": args.label,
        "house_json": str(args.house),
        "door_id": args.door,
        "cp_id": result["cp_id"],
        "start_room_id": args.start_room,
        "start_pos": result["start_pos"],
        "start_dist_m": result["start_dist_m"],
        "agent_path_m": result["agent_path_m"],
        "staged_ids": result["staged_ids"],
        "passage_blocked": result["passage_blocked"],
        "blocked_step": result["blocked_step"],
        "peak_num_moving": result["peak_num_moving"],
        "num_objects_fallen": result["num_objects_fallen"],
        "fallen_door_overlap": result["fallen_door_overlap"],
        "blocking_ids": result["blocking_ids"],
        "cp_blocked": result["cp"].blocked,
        "cp_decision": result["cp"].decision,
        "cp_traversable_exits": len(result["cp"].traversable_exits()),
        "frames_written": result["frames_written"],
        "hold_frames": result["hold_frames"],
        "changepoint_scan_count": result["changepoint_scan_count"],
        "video": str(out_video),
        "changepoints_json": str(changepoints_json),
        "payload_png": str(payload_png),
    }
    summary_path = out_dir / f"{args.label}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_video}")
    print(f"wrote {changepoints_json}")
    print(f"wrote {payload_png}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
