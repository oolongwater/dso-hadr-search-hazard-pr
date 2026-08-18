"""Parameterized hazard families for AI2-THOR (proposal section 3.1.1).

Each hazard exposes latent variables via HazardConfig and a transparent
per-tick propagation rule via tick(). This is the SIM-05 direction built on
the verified SIM-04 PhysX primitives in hazard/utils.py.
"""

from __future__ import annotations

import math
import random
from typing import Any

from hazard.functions import (
    FireParams,
    HazardConfig,
    ObstructionParams,
    SmokeParams,
    ThermalParams,
    advance_heat_field,
    configure_thermal,
    count_hot_objects,
    earthquake_params_from_latents,
    place_obstruction,
    sample_agent_temperature_c,
    set_smoke,
    start_earthquake,
    start_fire,
    stop_earthquake,
    stop_fire,
)

from hazard.utils import (
    diff_states,
    find_object,
    find_objects,
    object_state_snapshot,
    settle_physics,
    visibility_metric_from_density,
)
from core.thor import (
    agent_pose,
    distance_xz,
    get_reachable_positions,
    teleport,
    yaw_toward,
)

BREAKABLE_HINT_TYPES = ("Bottle", "Bowl", "Mug", "Plate", "Cup", "Egg", "Vase", "Pan", "Pot")
BLOCKER_TYPES = ("Stool", "GarbageCan", "Chair", "Box", "Pot", "Pan", "Bowl")
FALLABLE_MIN_Y_M = 0.35
VIEW_MIN_DIST_M = 1.5
VIEW_MAX_DIST_M = 4.0
VIEW_FOV_DEG = 60.0
VIEW_YAW_STEP_DEG = 30
VIEW_GROUND_TRUTH_CANDIDATES = 16


def _dist(a: dict[str, float], b: dict[str, float]) -> float:
    return distance_xz(float(a["x"]), float(a["z"]), float(b["x"]), float(b["z"]))


class SmokeFireHazard:
    """Radial density field grows from a source; degrades visibility and can ignite objects.

    Latents: source position, emission rate, spread rate, density field, heat threshold.
    Propagation: density rises over time and spreads outward; agent-local density drives
    visibility degradation; objects past the heat threshold break (object-state change).
    """

    def __init__(self, config: HazardConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.source: dict[str, float] = {}
        self.baseline: dict[str, dict[str, Any]] = {}
        self.broken_ids: set[str] = set()
        self.ignited: dict[str, int] = {}
        self.burn_delays: dict[str, int] = {}
        self.render_density: float = 0.0
        self.fire_object_id: str | None = None
        self._fire_started: bool = False
        self.heat_field = None
        self.thermal_delta_time: float = 0.2
        self._density_span: tuple[float, float] = (0.0, 0.0)
        self._substep_stats: dict[str, Any] = {}

    def total_steps(self) -> int:
        return self.config.total_ticks

    def setup(self, controller) -> dict[str, Any]:
        event = controller.last_event
        self.baseline = object_state_snapshot(event)

        if self.config.smoke.source_position is not None:
            self.source = dict(self.config.smoke.source_position)
        else:
            candidate = None
            for obj_type in BREAKABLE_HINT_TYPES:
                candidate = find_object(event, object_type=obj_type)
                if candidate is not None:
                    break
            if candidate is None:
                candidate = find_objects(event, predicate=lambda o: o.get("breakable"))
                candidate = candidate[0] if candidate else None
            if candidate is not None:
                self.source = dict(candidate["position"])
                self.fire_object_id = candidate["objectId"]
            else:
                agent = event.metadata["agent"]["position"]
                self.source = dict(agent)
        self.config.smoke.source_position = dict(self.source)

        if self.fire_object_id is not None:
            if self.config.native_effects:
                start_fire(
                    controller,
                    FireParams(object_id=self.fire_object_id, severity=self.config.severity),
                )
                self._fire_started = True
            self.ignited[self.fire_object_id] = max(0, self.config.onset_step - 1)
            self.burn_delays[self.fire_object_id] = self._burn_delay_ticks()

        reachable = get_reachable_positions(controller)
        viewpoint = None
        best = -1.0
        for point in reachable:
            d = _dist(point, self.source)
            if 1.2 <= d <= 2.2 and d > best:
                best = d
                viewpoint = point
        if viewpoint is None and reachable:
            viewpoint = min(reachable, key=lambda p: abs(_dist(p, self.source) - 1.6))
        if viewpoint is not None:
            yaw = yaw_toward(
                float(viewpoint["x"]), float(viewpoint["z"]),
                float(self.source["x"]), float(self.source["z"]),
            )
            teleport(
                controller,
                float(viewpoint["x"]), float(viewpoint["y"]), float(viewpoint["z"]),
                yaw=yaw,
            )
            controller.step(action="LookDown", degrees=15)

        thermal_info: dict[str, Any] = {}
        if self.config.native_effects:
            props_event = controller.step(action="GetMapViewCameraProperties")
            props = props_event.metadata.get("actionReturn")
            if props and props_event.metadata.get("lastActionSuccess", False):
                lat = self.config.thermal
                sev = float(max(0.05, self.config.severity))
                thermal = ThermalParams(
                    ambient_c=lat.ambient_c,
                    flame_core_c=lat.flame_core_c * (0.35 + 0.65 * sev),
                    diffusivity=max(0.02, lat.diffusivity * sev),
                    convection_gain=lat.convection_gain * sev,
                    cooling_rate=lat.cooling_rate,
                    flame_radius_m=lat.flame_radius_m * (0.5 + 0.5 * sev),
                    hot_threshold_c=lat.hot_threshold_c,
                    resolution_x=lat.resolution_x,
                    resolution_z=lat.resolution_z,
                )
                configure_thermal(controller, thermal, props)
                thermal_info = {"thermal_configured": True, "thermal_severity_scale": sev}

        return {
            "source_position": self.source,
            "num_breakable": len(
                find_objects(event, predicate=lambda o: o.get("breakable"))
            ),
            **thermal_info,
        }

    def _density_at(self, point: dict[str, float], step: float) -> tuple[float, float]:
        cfg = self.config
        lat = cfg.smoke
        elapsed = max(0, step - cfg.onset_step)
        source_density = min(cfg.severity, lat.emission_rate * cfg.severity * elapsed * 4.0)
        radius = lat.spread_rate * elapsed
        dist = _dist(point, self.source)
        if radius <= 1e-6:
            return (source_density if dist < 0.3 else 0.0), radius
        # smoke front: once the expanding front passes a point it fills to the
        # source density; a soft edge gives a gradient just ahead of the front.
        reach = max(0.0, min(1.0, (radius - dist) / 0.6 + 1.0)) if dist > radius else 1.0
        return source_density * reach, radius

    def _burn_delay_ticks(self) -> int:
        lat = self.config.smoke
        lo = min(lat.burn_delay_ticks_min, lat.burn_delay_ticks_max)
        hi = max(lat.burn_delay_ticks_min, lat.burn_delay_ticks_max)
        return self.rng.randint(lo, hi)

    def _fire_radius(self, step: float) -> float:
        cfg = self.config
        elapsed = max(0, step - cfg.onset_step)
        return cfg.smoke.fire_spread_rate * elapsed

    def _room_fill(self, step: float) -> float:
        cfg = self.config
        elapsed = max(0, step - cfg.onset_step)
        return min(cfg.severity, cfg.smoke.room_fill_rate * elapsed)

    def _fog_density_at(self, step: float, agent: dict[str, float]) -> tuple[float, float, float, float]:
        agent_density, radius = self._density_at(agent, step)
        agent_density = float(max(0.0, min(1.0, agent_density)))
        room_fill = float(max(0.0, min(1.0, self._room_fill(step))))
        fog_density = max(agent_density, room_fill)
        return fog_density, agent_density, room_fill, radius

    def _nearest_ignited(self, point: dict[str, float], event) -> dict[str, float] | None:
        best_dist = float("inf")
        best_pos: dict[str, float] | None = None
        ignited_ids = set(self.ignited)
        for obj in find_objects(event, predicate=lambda o: o["objectId"] in ignited_ids):
            dist = _dist(point, obj["position"])
            if dist < best_dist:
                best_dist = dist
                best_pos = dict(obj["position"])
        return best_pos

    def _maybe_ignite_one(self, controller, step: int) -> dict[str, Any] | None:
        cfg = self.config
        if step < cfg.onset_step:
            return None
        event = controller.last_event

        if cfg.native_effects and self.heat_field is not None:
            max_ignitions = 1 + int(round(cfg.severity * 10))
            if len(self.ignited) >= max_ignitions:
                return None
            ignition_c = cfg.smoke.ignition_temperature_c
            candidates: list[tuple[float, dict[str, Any]]] = []
            breakable_ids = {
                o["objectId"]
                for o in find_objects(event, predicate=lambda o: o.get("breakable"))
            }
            for sample in self.heat_field.objects:
                if sample.object_id not in breakable_ids:
                    continue
                if sample.object_id in self.ignited or sample.object_id in self.broken_ids:
                    continue
                if sample.temperature_c < ignition_c:
                    continue
                obj = next(
                    (
                        o
                        for o in find_objects(event)
                        if o["objectId"] == sample.object_id
                    ),
                    None,
                )
                if obj is None or obj.get("isBroken"):
                    continue
                candidates.append((sample.temperature_c, obj))
            if not candidates:
                return None
            temp_c, target = max(candidates, key=lambda item: item[0])
            start_fire(
                controller,
                FireParams(
                    object_id=target["objectId"],
                    severity=min(1.0, cfg.severity),
                ),
            )
            self._fire_started = True
            self.ignited[target["objectId"]] = step
            self.burn_delays[target["objectId"]] = self._burn_delay_ticks()
            return {
                "objectId": target["objectId"],
                "objectType": target["objectType"],
                "temperature_c": round(temp_c, 2),
            }

        if cfg.native_effects:
            return None

        fire_radius = self._fire_radius(step)
        if fire_radius <= 1e-6:
            return None

        candidates = []
        for obj in find_objects(event, predicate=lambda o: o.get("breakable")):
            obj_id = obj["objectId"]
            if obj_id in self.ignited or obj_id in self.broken_ids or obj.get("isBroken"):
                continue
            anchor = self._nearest_ignited(obj["position"], event)
            if anchor is None:
                anchor = self.source
            dist_from_front = _dist(obj["position"], anchor)
            if dist_from_front > fire_radius:
                continue
            candidates.append((dist_from_front, obj))

        if not candidates:
            return None

        _, target = min(candidates, key=lambda item: item[0])
        if cfg.native_effects:
            start_fire(
                controller,
                FireParams(
                    object_id=target["objectId"],
                    severity=min(1.0, cfg.severity),
                ),
            )
            self._fire_started = True
        self.ignited[target["objectId"]] = step
        self.burn_delays[target["objectId"]] = self._burn_delay_ticks()
        return {
            "objectId": target["objectId"],
            "objectType": target["objectType"],
            "dist_from_front_m": round(min(candidates, key=lambda item: item[0])[0], 4),
        }

    def _maybe_break_burned(self, controller, step: int) -> list[dict[str, Any]]:
        newly_broken: list[dict[str, Any]] = []
        for obj_id, ignite_step in list(self.ignited.items()):
            if obj_id in self.broken_ids:
                continue
            delay = self.burn_delays.get(obj_id, self.config.smoke.burn_delay_ticks_min)
            if step - ignite_step < delay:
                continue
            res = controller.step(action="BreakObject", objectId=obj_id, forceAction=True)
            if res.metadata.get("lastActionSuccess", False):
                self.broken_ids.add(obj_id)
                newly_broken.append({"objectId": obj_id})
        return newly_broken

    def substep(self, controller, dt: float, frac: float) -> None:
        start, end = self._density_span
        fog_density = start + (end - start) * frac
        if self.config.native_effects:
            set_smoke(controller, SmokeParams(density=fog_density))
            self.render_density = 0.0
            self.heat_field = advance_heat_field(controller, dt)
        else:
            self.render_density = fog_density
            self.heat_field = None

        agent = controller.last_event.metadata["agent"]["position"]
        agent_temp_c = sample_agent_temperature_c(self.heat_field, agent)
        hot_threshold = self.config.thermal.hot_threshold_c
        num_hot = count_hot_objects(self.heat_field, hot_threshold)
        max_temp_c = round(self.heat_field.max_c, 2) if self.heat_field else 0.0
        self._substep_stats = {
            "fog_density": round(fog_density, 4),
            "max_temp_c": max_temp_c,
            "agent_temp_c": round(agent_temp_c, 2),
            "num_hot_objects": num_hot,
        }

    def tick(self, controller, step: int) -> dict[str, Any]:
        event = controller.last_event
        agent = event.metadata["agent"]["position"]
        fog_start, agent_density, room_fill, radius = self._fog_density_at(float(step), agent)
        fog_end, _, _, _ = self._fog_density_at(float(step + 1), agent)
        self._density_span = (fog_start, fog_end)

        newly_ignited = self._maybe_ignite_one(controller, step)
        newly_broken = self._maybe_break_burned(controller, step)

        agent_temp_c = sample_agent_temperature_c(self.heat_field, agent)
        hot_threshold = self.config.thermal.hot_threshold_c
        num_hot = count_hot_objects(self.heat_field, hot_threshold)
        max_temp_c = round(self.heat_field.max_c, 2) if self.heat_field else 0.0

        visibility = visibility_metric_from_density(agent_density)
        access_invalidated = agent_density >= self.config.smoke.access_density_cutoff
        return {
            "step": step,
            "source_density": round(min(self.config.severity, self._density_at(self.source, float(step))[0]), 4),
            "smoke_radius_m": round(radius, 4),
            "fire_radius_m": round(self._fire_radius(float(step)), 4),
            "agent_density": round(agent_density, 4),
            "room_fill": round(room_fill, 4),
            "fog_density": round(fog_end, 4),
            "max_temp_c": max_temp_c,
            "agent_temp_c": round(agent_temp_c, 2),
            "num_hot_objects": num_hot,
            "visibility": visibility,
            "access_invalidated": access_invalidated,
            "newly_ignited": newly_ignited,
            "newly_broken": newly_broken,
            "num_ignited_total": len(self.ignited),
            "num_broken_total": len(self.broken_ids),
        }

    def finalize(self, controller) -> dict[str, Any]:
        if self.config.native_effects:
            set_smoke(controller, SmokeParams(density=0.0))
            if self._fire_started:
                stop_fire(controller)
        after = object_state_snapshot(controller.last_event)
        return {
            "broken_object_ids": sorted(self.broken_ids),
            "state_changes": diff_states(self.baseline, after)[:12],
            "feasibility": {
                "can_render": True,
                "can_alter_simulator_state": len(self.broken_ids) > 0,
                "robot_can_sense_consequence": True,
                "changes_planning_or_task_success": True,
            },
            "unity_side_code_required": not self.config.native_effects,
            "unity_side_note": (
                "Native Unity particles + scene fog drive visibility when native_effects "
                "is enabled; otherwise an image-space overlay is used. Object ignition "
                "uses real PhysX BreakObject in both modes."
            ),
        }


class EarthquakeHazard:
    """Seeded impulses shake movable objects; high cumulative impact breaks them.

    Latents: disturbance magnitude (severity), per-object stability, integrity threshold, seed.
    Propagation: objects move/fall, support relations break, debris appears; traversability changes.
    """

    def __init__(self, config: HazardConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.baseline: dict[str, dict[str, Any]] = {}
        self.cumulative_impulse: dict[str, float] = {}
        self.broken_ids: set[str] = set()
        self.movable_ids: list[str] = []
        self.reachable_before: int = 0
        self.viewpoint: dict[str, float] | None = None
        self.view_yaw: float = 0.0
        self.view_horizon: float = 0.0
        self.visible_fallable_ids: list[str] = []
        self.render_density = 0.0
        self.render_shift: tuple[int, int] = (0, 0)
        self._earthquake_started: bool = False

    def total_steps(self) -> int:
        return self.config.total_ticks

    def _is_fallable(self, obj: dict[str, Any], *, floor_y: float = 0.0) -> bool:
        if not (obj.get("pickupable") or obj.get("moveable")):
            return False
        pos = obj.get("position") or {}
        return float(pos.get("y", 0.0)) >= floor_y + FALLABLE_MIN_Y_M

    def _score_pose(
        self,
        fallable: list[dict[str, Any]],
        px: float,
        pz: float,
        yaw: float,
    ) -> float:
        """Reward poses that frame nearby elevated fallable objects."""
        score = 0.0
        for obj in fallable:
            pos = obj.get("position") or {}
            dx = float(pos.get("x", 0.0)) - px
            dz = float(pos.get("z", 0.0)) - pz
            dist = math.hypot(dx, dz)
            if dist > VIEW_MAX_DIST_M or dist < VIEW_MIN_DIST_M:
                continue
            bearing = math.degrees(math.atan2(dx, dz))
            rel = abs((bearing - yaw + 180.0) % 360.0 - 180.0)
            if rel > VIEW_FOV_DEG / 2.0:
                continue
            score += 1.0 / (0.5 + dist)
        return score

    def _visible_fallable_ids(self, event, fallable: list[dict[str, Any]]) -> list[str]:
        fallable_by_id = {str(o["objectId"]): o for o in fallable if o.get("objectId")}
        visible: list[str] = []
        for obj in event.metadata.get("objects") or []:
            oid = obj.get("objectId")
            if oid in fallable_by_id and obj.get("visible"):
                visible.append(str(oid))
        return visible

    def _cluster_horizon(
        self,
        fallable: list[dict[str, Any]],
        px: float,
        py: float,
        pz: float,
    ) -> float:
        nearby = []
        for obj in fallable:
            pos = obj.get("position") or {}
            dist = distance_xz(float(pos.get("x", 0.0)), float(pos.get("z", 0.0)), px, pz)
            if dist <= VIEW_MAX_DIST_M:
                nearby.append(float(pos.get("y", 0.0)))
        if not nearby:
            return 12.0
        mean_y = sum(nearby) / len(nearby)
        dy = mean_y - py
        if dy <= 0.05:
            return 8.0
        return max(5.0, min(25.0, math.degrees(math.atan2(dy, 2.5))))

    def setup(
        self,
        controller,
        *,
        reachable: list[dict[str, float]] | None = None,
        floor_base_y: float = 0.0,
    ) -> dict[str, Any]:
        event = controller.last_event
        self.baseline = object_state_snapshot(event)
        self.reachable_before = len(get_reachable_positions(controller))

        movable = find_objects(
            event, predicate=lambda o: o.get("pickupable") or o.get("moveable")
        )
        self.movable_ids = [o["objectId"] for o in movable]
        fallable = [o for o in movable if self._is_fallable(o, floor_y=floor_base_y)]

        if reachable is None:
            reachable = get_reachable_positions(controller)

        candidates: list[tuple[float, dict[str, float], float]] = []
        if fallable and reachable:
            sample = reachable[:: max(1, len(reachable) // 60)]
            for p in sample:
                for yaw in range(0, 360, VIEW_YAW_STEP_DEG):
                    s = self._score_pose(fallable, float(p["x"]), float(p["z"]), float(yaw))
                    if s > 0.0:
                        candidates.append((s, p, float(yaw)))
            candidates.sort(key=lambda item: item[0], reverse=True)

        best_score = -1.0
        best_visible = 0
        if candidates:
            for geo_score, p, yaw in candidates[:VIEW_GROUND_TRUTH_CANDIDATES]:
                horizon = self._cluster_horizon(
                    fallable,
                    float(p["x"]),
                    float(p["y"]),
                    float(p["z"]),
                )
                teleport(
                    controller,
                    float(p["x"]),
                    float(p["y"]),
                    float(p["z"]),
                    yaw=yaw,
                    horizon=horizon,
                )
                visible_ids = self._visible_fallable_ids(controller.last_event, fallable)
                vis_count = len(visible_ids)
                if vis_count > best_visible or (
                    vis_count == best_visible and geo_score > best_score
                ):
                    best_visible = vis_count
                    best_score = geo_score
                    self.viewpoint = dict(p)
                    self.view_yaw = yaw
                    self.view_horizon = horizon
                    self.visible_fallable_ids = visible_ids

        if self.viewpoint is not None:
            teleport(
                controller,
                float(self.viewpoint["x"]),
                float(self.viewpoint["y"]),
                float(self.viewpoint["z"]),
                yaw=self.view_yaw,
                horizon=self.view_horizon,
            )
        return {
            "num_movable": len(self.movable_ids),
            "num_fallable": len(fallable),
            "reachable_before": self.reachable_before,
            "viewpoint": self.viewpoint,
            "view_yaw": self.view_yaw,
            "view_horizon": self.view_horizon,
            "viewpoint_score": round(best_score, 4),
            "visible_fallable_at_viewpoint": best_visible,
            "visible_fallable_ids": list(self.visible_fallable_ids),
        }

    def _active(self, step: int) -> bool:
        cfg = self.config
        return cfg.onset_step <= step < cfg.onset_step + cfg.total_ticks

    def _should_break_on_impact(self, obj: dict[str, Any]) -> bool:
        """Break only after a fall or fast motion, not from quake exposure alone."""
        oid = str(obj.get("objectId") or "")
        base = self.baseline.get(oid)
        if base is None:
            return bool(obj.get("isMoving"))
        pos = obj.get("position") or {}
        fallen = float(pos.get("y", 0.0)) <= float(base.get("y", 0.0)) - 0.3
        return fallen or bool(obj.get("isMoving"))

    def tick(self, controller, step: int) -> dict[str, Any]:
        cfg = self.config
        pushed: list[str] = []
        newly_broken: list[dict[str, Any]] = []

        # sinusoidal ground-shake phase; drives image-space camera shake in overlay
        # mode, or Unity-side gravity oscillation when native_effects is enabled.
        lat = cfg.earthquake
        period = max(2, lat.shake_period_ticks)
        phase = math.sin(2.0 * math.pi * step / period)
        if cfg.native_effects:
            self.render_shift = (0, 0)
        else:
            self.render_shift = (int(round(phase * lat.shake_pixels)), 0)
        magnitude = 0.0

        if step == cfg.onset_step and cfg.native_effects and not self._earthquake_started:
            eq_params = earthquake_params_from_latents(cfg.severity, lat)
            res = start_earthquake(controller, eq_params)
            if res.metadata.get("lastActionSuccess", False):
                self._earthquake_started = True

        if step >= cfg.onset_step and not cfg.native_effects:
            # a shaking floor is, in the room frame, one synchronized pseudo-force
            # on every object: same axis, direction flipping with the phase sign.
            magnitude = lat.impulse_base_newtons * cfg.severity * abs(phase) * lat.impulse_scale
            angle = lat.shake_axis_deg if phase >= 0 else (lat.shake_axis_deg + 180.0) % 360.0
            event = controller.last_event
            movable = find_objects(
                event, predicate=lambda o: (o.get("pickupable") or o.get("moveable"))
                and not o.get("isBroken"),
            )
            for obj in movable:
                res = controller.step(
                    action="DirectionalPush",
                    objectId=obj["objectId"],
                    moveMagnitude=magnitude,
                    pushAngle=angle,
                    forceAction=True,
                )
                if res.metadata.get("lastActionSuccess", False):
                    pushed.append(obj["objectId"])
                    mass = float(obj.get("mass", 1.0)) or 1.0
                    self.cumulative_impulse[obj["objectId"]] = (
                        self.cumulative_impulse.get(obj["objectId"], 0.0)
                        + magnitude / (100.0 * mass)
                    )
                    if (
                        obj.get("breakable")
                        and obj["objectId"] not in self.broken_ids
                        and self.cumulative_impulse[obj["objectId"]] >= lat.integrity_threshold
                        and self._should_break_on_impact(obj)
                    ):
                        br = controller.step(
                            action="BreakObject", objectId=obj["objectId"], forceAction=True
                        )
                        if br.metadata.get("lastActionSuccess", False):
                            self.broken_ids.add(obj["objectId"])
                            newly_broken.append(
                                {"objectId": obj["objectId"], "objectType": obj["objectType"]}
                            )
        elif step >= cfg.onset_step and cfg.native_effects:
            magnitude = lat.impulse_base_newtons * cfg.severity * abs(phase) * lat.impulse_scale
            event = controller.last_event
            movable = find_objects(
                event, predicate=lambda o: (o.get("pickupable") or o.get("moveable"))
                and not o.get("isBroken"),
            )
            for obj in movable:
                mass = float(obj.get("mass", 1.0)) or 1.0
                self.cumulative_impulse[obj["objectId"]] = (
                    self.cumulative_impulse.get(obj["objectId"], 0.0)
                    + magnitude / (100.0 * mass)
                )
                if (
                    obj.get("breakable")
                    and obj["objectId"] not in self.broken_ids
                    and self.cumulative_impulse[obj["objectId"]] >= lat.integrity_threshold
                    and self._should_break_on_impact(obj)
                ):
                    br = controller.step(
                        action="BreakObject", objectId=obj["objectId"], forceAction=True
                    )
                    if br.metadata.get("lastActionSuccess", False):
                        self.broken_ids.add(obj["objectId"])
                        newly_broken.append(
                            {"objectId": obj["objectId"], "objectType": obj["objectType"]}
                        )
                if obj.get("isMoving"):
                    pushed.append(obj["objectId"])

        num_moving = len(
            find_objects(controller.last_event, predicate=lambda o: o.get("isMoving"))
        )
        max_rise_m = 0.0
        for obj in controller.last_event.metadata.get("objects") or []:
            oid = str(obj.get("objectId") or "")
            base = self.baseline.get(oid)
            if base is None:
                continue
            pos = obj.get("position") or {}
            rise = float(pos.get("y", 0.0)) - float(base.get("y", 0.0))
            if rise > max_rise_m:
                max_rise_m = rise
        return {
            "step": step,
            "active": step >= cfg.onset_step,
            "impulse_newtons": round(magnitude, 3),
            "num_pushed": len(pushed),
            "num_moving": num_moving,
            "max_rise_m": round(max_rise_m, 4),
            "newly_broken": newly_broken,
            "num_broken_total": len(self.broken_ids),
        }

    def finalize(self, controller) -> dict[str, Any]:
        if self.config.native_effects and self._earthquake_started:
            stop_earthquake(controller)
        # restore a clean view and let objects settle
        if self.viewpoint is not None:
            teleport(
                controller,
                float(self.viewpoint["x"]),
                float(self.viewpoint["y"]),
                float(self.viewpoint["z"]),
                yaw=self.view_yaw,
                horizon=self.view_horizon,
            )
        settle_physics(controller, steps=15)
        after = object_state_snapshot(controller.last_event)
        reachable_after = len(get_reachable_positions(controller))
        changes = diff_states(self.baseline, after)
        return {
            "reachable_before": self.reachable_before,
            "reachable_after": reachable_after,
            "broken_object_ids": sorted(self.broken_ids),
            "num_state_changes": len(changes),
            "state_changes": changes[:12],
            "feasibility": {
                "can_render": True,
                "can_alter_simulator_state": len(changes) > 0,
                "robot_can_sense_consequence": True,
                "changes_planning_or_task_success": True,
            },
            "unity_side_code_required": not self.config.native_effects,
        }


class ProgressiveObstructionHazard:
    """Objects accumulate in a passage until it becomes blocked.

    Latents: spawn location (chokepoint), growth rate, clearance threshold.
    Propagation: free space shrinks, path cost rises, connector state goes
    open -> constrained -> blocked, requiring replanning.
    """

    def __init__(self, config: HazardConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.viewpoint: dict[str, float] = {}
        self.view_yaw: float = 0.0
        self.baseline_cost: float | None = None
        self.blockers: list[dict[str, Any]] = []
        self.slots: list[tuple[float, float]] = []
        self.used_slots: set[int] = set()
        self.placed_poses: list[dict[str, Any]] = []
        self.placed_count: int = 0
        self.edge_state: str = "open"
        self.render_density = 0.0

    def total_steps(self) -> int:
        return self.config.total_ticks

    def _path_cost(self, controller) -> float | None:
        agent = controller.last_event.metadata["agent"]["position"]
        for reach in (2.0, 1.5, 1.0):
            target = {
                "x": agent["x"] + math.sin(math.radians(self.view_yaw)) * reach,
                "y": agent["y"],
                "z": agent["z"] + math.cos(math.radians(self.view_yaw)) * reach,
            }
            ev = controller.step(
                action="GetShortestPathToPoint", position=agent, target=target
            )
            if not ev.metadata.get("lastActionSuccess", False):
                continue
            ret = ev.metadata.get("actionReturn") or {}
            corners = ret.get("corners") or []
            if len(corners) < 2:
                continue
            cost = 0.0
            for i in range(len(corners) - 1):
                cost += _dist(corners[i], corners[i + 1])
            return cost
        return None

    def _remaining_blockers(self, controller) -> list[dict[str, Any]]:
        """Blockers still present and not yet placed (SetObjectPoses-free)."""
        placed_ids = {p["objectId"] for p in self.placed_poses}
        event = controller.last_event
        present = {o["objectId"]: o for o in event.metadata.get("objects") or []}
        out = []
        for b in self.blockers:
            if b["objectId"] in placed_ids:
                continue
            cur = present.get(b["objectId"])
            if cur is not None and (cur.get("moveable") or cur.get("pickupable")):
                out.append(cur)
        return out

    def _moveahead_ok(self, controller) -> bool:
        teleport(
            controller,
            float(self.viewpoint["x"]), float(self.viewpoint["y"]), float(self.viewpoint["z"]),
            yaw=self.view_yaw,
        )
        ok = controller.step(action="MoveAhead").metadata.get("lastActionSuccess", False)
        teleport(
            controller,
            float(self.viewpoint["x"]), float(self.viewpoint["y"]), float(self.viewpoint["z"]),
            yaw=self.view_yaw,
        )
        return bool(ok)

    def _pile_centroid(self) -> dict[str, float]:
        """Where the objects are collating (slot centre ahead until first placed)."""
        if self.placed_poses:
            n = len(self.placed_poses)
            cx = sum(p["position"]["x"] for p in self.placed_poses) / n
            cz = sum(p["position"]["z"] for p in self.placed_poses) / n
            return {"x": cx, "z": cz}
        yaw_rad = math.radians(self.view_yaw)
        return {
            "x": self.viewpoint["x"] + math.sin(yaw_rad) * 0.75,
            "z": self.viewpoint["z"] + math.cos(yaw_rad) * 0.75,
        }

    def aim_at_pile(self, controller, look_down_deg: float = 30.0) -> None:
        """Point the FPV camera at the accumulation spot and tilt down onto it."""
        pile = self._pile_centroid()
        aim_yaw = yaw_toward(
            float(self.viewpoint["x"]), float(self.viewpoint["z"]),
            float(pile["x"]), float(pile["z"]),
        )
        teleport(
            controller,
            float(self.viewpoint["x"]), float(self.viewpoint["y"]), float(self.viewpoint["z"]),
            yaw=aim_yaw,
        )
        controller.step(action="LookDown", degrees=look_down_deg)

    def setup(self, controller) -> dict[str, Any]:
        from hazard.utils import find_moveahead_success_pose

        pose = find_moveahead_success_pose(controller)
        if pose is None:
            raise RuntimeError("No pose where MoveAhead succeeds; cannot site obstruction.")
        point, yaw = pose
        self.viewpoint = dict(point)
        self.view_yaw = float(yaw)
        teleport(
            controller,
            float(point["x"]), float(point["y"]), float(point["z"]), yaw=self.view_yaw,
        )
        self.baseline_cost = self._path_cost(controller)

        event = controller.last_event
        for obj_type in BLOCKER_TYPES:
            for obj in find_objects(event, object_type=obj_type):
                if obj.get("moveable") or obj.get("pickupable"):
                    self.blockers.append(obj)
        # place the largest object dead-centre first so MoveAhead is blocked,
        # then fill widely-spaced side/back slots with smaller objects
        self.blockers.sort(key=lambda o: float(o.get("mass", 1.0) or 1.0), reverse=True)
        self.slots = [
            (0.5, 0.0),
            (0.72, 0.5), (0.72, -0.5),
            (0.98, 0.25), (0.98, -0.25),
            (0.72, 0.82), (0.72, -0.82),
            (1.02, 0.0), (1.2, 0.0),
        ]
        self.aim_at_pile(controller)
        return {
            "viewpoint": self.viewpoint,
            "view_yaw": self.view_yaw,
            "baseline_path_cost": self.baseline_cost,
            "num_candidate_blockers": len(self.blockers),
            "blocker_ids": [b["objectId"] for b in self.blockers],
        }

    def tick(self, controller, step: int) -> dict[str, Any]:
        cfg = self.config
        lat = cfg.obstruction
        place_interval = max(1, round(1.0 / max(lat.growth_rate, 1e-3)))
        placed_now = None

        should_place = (
            step >= cfg.onset_step
            and (step - cfg.onset_step) % place_interval == 0
        )
        if should_place:
            remaining = self._remaining_blockers(controller)
            if remaining:
                blocker = remaining[0]
                yaw_rad = math.radians(self.view_yaw)
                for slot_idx, (fwd, lateral) in enumerate(self.slots):
                    if slot_idx in self.used_slots:
                        continue
                    target = {
                        "x": self.viewpoint["x"] + math.sin(yaw_rad) * fwd
                        + math.cos(yaw_rad) * lateral,
                        "y": float(blocker["position"]["y"]) + 0.05,
                        "z": self.viewpoint["z"] + math.cos(yaw_rad) * fwd
                        - math.sin(yaw_rad) * lateral,
                    }
                    res = place_obstruction(
                        controller,
                        ObstructionParams(object_id=blocker["objectId"], position=target),
                    )
                    if res.metadata.get("lastActionSuccess", False):
                        self.used_slots.add(slot_idx)
                        self.placed_count += 1
                        self.placed_poses.append(
                            {"objectId": blocker["objectId"], "position": target}
                        )
                        placed_now = {"objectType": blocker["objectType"], "position": target}
                        break

        cost = self._path_cost(controller)
        move_ok = self._moveahead_ok(controller)
        # re-aim the FPV at the accumulation spot for the captured frames
        self.aim_at_pile(controller)

        prev_state = self.edge_state
        if not move_ok:
            self.edge_state = "blocked"
        elif (
            self.baseline_cost
            and cost is not None
            and cost >= self.baseline_cost * lat.constrained_cost_ratio
        ):
            self.edge_state = "constrained"
        else:
            self.edge_state = "open"

        return {
            "step": step,
            "placed_count": self.placed_count,
            "placed_now": placed_now,
            "placed_ids": [p["objectId"] for p in self.placed_poses],
            "path_cost": None if cost is None else round(cost, 4),
            "moveahead_ok": move_ok,
            "edge_state": self.edge_state,
            "edge_transition": None if prev_state == self.edge_state else f"{prev_state}->{self.edge_state}",
        }

    def finalize(self, controller) -> dict[str, Any]:
        final_ok = self._moveahead_ok(controller)
        return {
            "baseline_path_cost": self.baseline_cost,
            "final_edge_state": self.edge_state,
            "objects_placed": self.placed_count,
            "final_moveahead_ok": final_ok,
            "passage_blocked": not final_ok,
            "feasibility": {
                "can_render": True,
                "can_alter_simulator_state": True,
                "robot_can_sense_consequence": True,
                "changes_planning_or_task_success": not final_ok,
            },
            "unity_side_code_required": False,
        }


HAZARD_CLASSES = {
    "smoke": SmokeFireHazard,
    "earthquake": EarthquakeHazard,
    "obstruction": ProgressiveObstructionHazard,
}


def build_hazard(name: str, config: HazardConfig):
    if name not in HAZARD_CLASSES:
        raise ValueError(f"Unknown hazard '{name}'. Options: {list(HAZARD_CLASSES)}")
    return HAZARD_CLASSES[name](config)
