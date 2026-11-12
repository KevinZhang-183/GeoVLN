"""Rule policies. VLM selection is a later step; landmarks are chosen geometrically."""

from __future__ import annotations

import numpy as np

from geonav._bootstrap import install

install()

from parse_instructions import _landmark, _place_only  # noqa: E402
from portal_map import NOT_A_DOOR, _agent_axes  # noqa: E402

from geonav.geom import bearing_deg, plan, unknown_beyond  # noqa: E402
from geonav.waypoints import Waypoint, landmark_waypoint, point_waypoint, portal_waypoint, stand_point  # noqa: E402
from geonav.world import _is_door  # noqa: E402

TURNS = {
    "turn_left": ("turn_left", 6),
    "turn_right": ("turn_right", 6),
    "turn_around": ("turn_left", 12),
}
APPROACH_LOOK_M = 2.0
TURN_NAMES = set(TURNS)
# The final landmark ends the episode, so one or two flickering masks are not enough.
FINAL_HITS = 3
FINAL_SCORE = 0.6
# R2R counts a stop within 3 m of the goal, and episodes do not start that close to it.
MIN_STOP_TRAVEL_M = 3.0
FALLBACK_STEPS = 80
# Keeps a frontier at the agent's feet from scoring without bound.
FRONTIER_D0 = 1.0
# A doorway opens a new space however little of it is visible yet, so it earns
# area on top of what lies behind it. Widths near 1.5 m earn the most.
DOOR_BONUS_CONFIRMED = 4.0
DOOR_BONUS_CANDIDATE = 2.0
DOOR_WIDTH_BEST = 1.5
DOOR_WIDTH_SIGMA = 0.5
# After a look, the locked door gives way only to a target this many times better.
SWITCH_RATIO = 1.5
WRONG_SIDE = 0.25
ROOM_DEPTH_M = 1.5
ROOM_RADIUS_M = 4.0
ROOM_ANCHORS = {
    "kitchen": ("refrigerator", "stove", "oven"),
    "bedroom": ("bed",),
    "bathroom": ("toilet", "bathtub", "shower"),
    "living room": ("sofa", "couch", "tv"),
    "dining room": ("dining table",),
    "office": ("desk",),
    "laundry room": ("washing machine",),
}
SYNONYMS = {
    "lamp": ("light", "floor lamp", "table lamp"),
    "couch": ("sofa",),
    "sofa": ("couch",),
    "tv": ("television",),
    "television": ("tv",),
    "fridge": ("refrigerator",),
    "refrigerator": ("fridge",),
    "rug": ("carpet",),
    "carpet": ("rug",),
}


class Decision:
    def __init__(self, kind, action=None, waypoint=None, reason=""):
        self.kind = kind
        self.action = action
        self.waypoint = waypoint
        self.reason = reason


class Policy:
    def __init__(self, mode, parsed, goal_xyz, allow_look=True):
        self.mode = mode
        self.parsed = parsed or {}
        self.steps = list(self.parsed.get("subtasks") or [])
        self.goal_xyz = None if goal_xyz is None else np.asarray(goal_xyz, dtype=np.float64)
        self.allow_look = allow_look
        self.survey = allow_look
        self.index = 0
        self.lock = None
        self.pending = []
        self.pending_reason = "turn"
        self.turn_sets_prior = False
        self.prior = None
        self.arrived_name = None
        self.banned = []
        self.tried_skip = False
        self.did_backtrack = False
        self.crossed = set()
        self.baseline = set()
        self.start_step = 0
        self.traveled = 0.0
        self.last_xz = None
        self.step = 0
        self.ready = False
        self.looks = []
        self.origin_xz = None
        self.rejected = set()
        self.entered = None
        self.fallback = False
        self.room_rejected = False
        self.score_log = []
        self.recheck = False

    def prompts(self):
        """Phrases for SAM3. Goal follows the dataset point and does not build landmarks."""
        if self.mode == "goal":
            return []
        index = self.index if self.mode == "landmark" else 0
        return prompts_for(self.parsed, index)

    def decide(self, world, pose, step):
        self._move(pose, step)
        if self.origin_xz is None:
            self.origin_xz = np.asarray(pose, dtype=np.float64)[[0, 2], 3].copy()
        if not self.ready:
            self._begin(world, pose)
            self.ready = True
        if self.mode == "goal":
            return self._goal(pose)
        self._note_entry(world, pose)
        if self.pending:
            return Decision("action", action=self.pending.pop(0), reason=self.pending_reason)
        if self.turn_sets_prior:
            self.prior = _agent_axes(pose)[0].copy()
            self.turn_sets_prior = False
            self._finish(world, pose)
        if self.mode == "landmark":
            self._skip_if_needed(world, pose)
            self._start_fallback()
            self._complete_ready(world, pose)
            if self._current() is None:
                return Decision("stop", reason="fallback" if self.fallback else "finished")
            current = self._current()
            if current["action"] in TURN_NAMES:
                action, count = TURNS[current["action"]]
                self.pending = [action] * count
                self.pending_reason = current["action"]
                self.turn_sets_prior = True
                return Decision("action", action=self.pending.pop(0), reason=self.pending_reason)
        if self._object_upgrade(world, pose):
            return Decision("waypoint", waypoint=self.lock, reason=self.lock.name)
        if self.lock is not None and self._lock_ok(world):
            if self._approach_due(world, self.lock, pose):
                # The door stays locked through the look, so a door off to the side cannot take over.
                return Decision("look", reason="approach")
            if self.recheck:
                self.recheck = False
                better = self._better_than_lock(world, pose)
                if better is not None:
                    self.lock = better
            return Decision("waypoint", waypoint=self.lock, reason=self.lock.name)
        self.recheck = False
        choice = self._choose(world, pose)
        if choice is not None:
            self.lock = choice
            self.did_backtrack = False
            if self._approach_due(world, choice, pose):
                return Decision("look", reason="approach")
            return Decision("waypoint", waypoint=choice, reason=choice.name)
        if self.allow_look:
            # Walk through the doorway ahead and survey once inside.
            door = self._explored_ahead(world, pose)
            if door is not None:
                self.lock = door
                if self._approach_due(world, door, pose):
                    return Decision("look", reason="approach")
                self.did_backtrack = False
                return Decision("waypoint", waypoint=door, reason=door.name)
            return Decision("look", reason="no candidate")
        if not self.did_backtrack and len(world.nodes) >= 2:
            node = world.nodes[-2]
            self.did_backtrack = True
            self.lock = point_waypoint(f"N{node.id}", node.xz, radius=0.6, snap_m=1.0)
            return Decision("waypoint", waypoint=self.lock, reason="backtrack")
        return Decision("stop", reason="no candidate")

    def note_look(self, xz):
        self.looks.append(np.asarray(xz, dtype=np.float64).copy())
        self.recheck = True

    def _better_than_lock(self, world, pose):
        """A target worth leaving the locked door for, or None. Door steps only weigh other doors."""
        if self.lock is None or not str(self.lock.name).startswith("P"):
            return None
        track = world.portals.track(self.lock.portal_id)
        if track is None:
            return None
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        current = self._portal_utility(world, track, agent, pose, self._step_side())
        held = 0.0 if current is None else current[0]
        others = [item for item in self._rank_portals(world, pose, None, self._step_side()) if item[3].id != track.id]
        if not self._door_step():
            others += self._rank_frontiers(world, pose)
        if not others:
            return None
        best = max(others, key=lambda item: item[0])
        if best[0] >= SWITCH_RATIO * max(held, 1e-6):
            return best[2]
        return None

    def _approach_due(self, world, waypoint, pose):
        """Look once when the agent is already within 2 m of an unsurveyed door.

        A door farther away is walked to first. Spinning in place there would
        finish still outside the 2 m radius and start another look forever.
        """
        if not self.survey or waypoint is None or not str(waypoint.name).startswith("P"):
            return False
        track = world.portals.track(waypoint.portal_id) if waypoint.portal_id is not None else None
        center = track.center if track is not None else waypoint.stages[0].xz
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        if float(np.hypot(agent[0] - center[0], agent[1] - center[1])) > APPROACH_LOOK_M:
            return False
        return not _near_look(self.looks, center, APPROACH_LOOK_M)

    def note_reached(self, waypoint):
        self.arrived_name = None if waypoint is None else waypoint.name
        if waypoint is not None and waypoint.portal_id is not None:
            self.crossed.add(waypoint.portal_id)
        if waypoint is not None and (waypoint.name[:1] in ("F", "N") or self.mode != "landmark"):
            self.banned.append(np.asarray(waypoint.stages[-1].xz, dtype=np.float64))
        self.lock = None

    def abandon(self, waypoint, world):
        if waypoint is None:
            return
        self.banned.append(np.asarray(waypoint.stages[-1].xz, dtype=np.float64))
        if waypoint.portal_id is not None:
            self.crossed.add(waypoint.portal_id)
            world.portals.abandon(waypoint.portal_id)
        self.lock = None
        self.arrived_name = None

    def _goal(self, pose):
        if self.goal_xyz is None:
            return Decision("stop", reason="no goal")
        agent = np.asarray(pose, dtype=np.float64)[:3, 3]
        if float(np.linalg.norm(agent - self.goal_xyz)) <= 1.5:
            return Decision("stop", reason="goal")
        waypoint = point_waypoint("G", self.goal_xyz[[0, 2]], radius=0.4, snap_m=1.5)
        self.lock = waypoint
        return Decision("waypoint", waypoint=waypoint, reason="goal")

    def _choose(self, world, pose):
        if self.mode == "landmark":
            current = self._current()
            if current is not None and self._is_final():
                choice = self._final_choice(world, pose, current)
                if choice is not None:
                    return choice
            elif current is not None and current["action"] in ("go_to", "stop"):
                if not _place_only(current.get("landmark") or "") and not _is_door(current.get("landmark") or ""):
                    found = self._best_instance(world, current, pose)
                    if found is not None:
                        return self._landmark_wp(found, world, pose, current.get("side") or "")
                if _place_only(current.get("landmark") or "") or _is_door(current.get("landmark") or ""):
                    choice = self._door_choice(world, pose, current)
                    if choice is not None:
                        return choice
            elif current is not None and current["action"] == "exit":
                choice = self._door_choice(world, pose, current)
                if choice is not None:
                    return choice
        portals = self._rank_portals(world, pose, None, "")
        frontiers = self._rank_frontiers(world, pose)
        pool = sorted(portals + frontiers, key=lambda item: (round(item[0], 2), item[1]), reverse=True)
        if not pool:
            return None
        self._log(pool[0][2], portals, frontiers)
        return pool[0][2]

    def _door_step(self):
        current = self._current() if self.mode == "landmark" else None
        if current is None:
            return False
        landmark = current.get("landmark") or ""
        return current["action"] == "exit" or _place_only(landmark) or _is_door(landmark)

    def _step_side(self):
        current = self._current() if self.mode == "landmark" else None
        return "" if current is None else current.get("side") or ""

    def _log(self, chosen, portals, frontiers):
        self.score_log.append({
            "step": self.step,
            "chosen": chosen.name,
            "portals": [
                {"id": item[3].id, "area_m2": round(item[4], 2), "path_m": round(item[5], 2), "utility": round(item[0], 3)}
                for item in portals
            ],
            "frontiers": [
                {"id": item[3].id, "area_m2": round(item[4], 2), "path_m": round(item[5], 2), "utility": round(item[0], 3)}
                for item in frontiers
            ],
        })

    def _door_choice(self, world, pose, step):
        linked = []
        for instance in world.semantic.instances:
            if instance.hits < 2 or instance.portal_id is None or instance.label != step.get("landmark"):
                continue
            linked.append(instance.portal_id)
        side = step.get("side") or ""
        if linked:
            choice = self._best_portal(world, pose, set(linked), side)
            if choice is not None:
                return choice
        return self._best_portal(world, pose, None, side)

    def _final_choice(self, world, pose, step):
        """Where the last step walks before it may stop. Every stop has a go part first."""
        action = step["action"]
        landmark = step.get("landmark") or ""
        if action == "exit" or _place_only(landmark):
            anchor = self._room_anchor(world, step, pose) if _place_only(landmark) else None
            if anchor is not None:
                return self._landmark_wp(anchor, world, pose, step.get("side") or "")
            room = self._room_waypoint(world, pose)
            if room is not None:
                return room
            if self.fallback:
                return None
            return self._door_choice(world, pose, step)
        if self.fallback:
            return self._room_waypoint(world, pose)
        if action not in ("go_to", "stop") or not landmark:
            return None
        if _is_door(landmark):
            door = self._door_choice(world, pose, step)
            if door is None:
                return None
            # Stop in front of the door, not on the far side of it.
            return Waypoint(door.name, door.stages[:1], portal_id=door.portal_id)
        found = self._best_instance(world, step, pose, final=True)
        if found is not None:
            return self._landmark_wp(found, world, pose, step.get("side") or "")
        return None

    def _room_anchor(self, world, step, pose):
        labels = _room_anchor_labels(step.get("landmark") or "")
        if not labels:
            return None
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        pool = [
            item for item in world.semantic.instances
            if item.label in labels and item.id not in self.rejected and _strong(item)
        ]
        if not pool:
            return None
        pool.sort(key=lambda item: float(np.hypot(item.x - agent[0], item.z - agent[1])))
        return pool[0]

    def _room_waypoint(self, world, pose):
        """Walk into the space beyond the last doorway crossed during this step or the one before it."""
        if self.room_rejected or self.entered is None or world.grid is None:
            return None
        track_id, sign, index = self.entered
        if not self.fallback and index < self.index - 1:
            return None
        track = world.portals.track(track_id)
        if track is None:
            return None
        point = room_point(world.grid, track.center, sign * np.asarray(track.normal, dtype=np.float64))
        if point is None:
            return None
        return point_waypoint("R", point, radius=0.6, snap_m=1.0)

    def _note_entry(self, world, pose):
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        for track_id in getattr(world, "new_traversals", None) or []:
            track = world.portals.track(track_id)
            if track is None or getattr(track, "reject", "") in NOT_A_DOOR:
                continue
            side = float(np.dot(agent - np.asarray(track.center), np.asarray(track.normal)))
            self.entered = (track_id, 1.0 if side >= 0.0 else -1.0, self.index)
            self.room_rejected = False

    def _start_fallback(self):
        if self.fallback or not self._is_final():
            return
        if self.step - self.start_step < FALLBACK_STEPS:
            return
        self.fallback = True
        self.lock = None

    def _is_final(self):
        return self.mode == "landmark" and bool(self.steps) and self.index == len(self.steps) - 1

    def _may_stop(self, pose):
        if self.origin_xz is None:
            return True
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        return float(np.hypot(agent[0] - self.origin_xz[0], agent[1] - self.origin_xz[1])) >= MIN_STOP_TRAVEL_M

    def _object_upgrade(self, world, pose):
        if self.mode != "landmark":
            return False
        current = self._current()
        if current is None or current["action"] not in ("go_to", "stop"):
            return False
        landmark = current.get("landmark") or ""
        if _place_only(landmark) or _is_door(landmark):
            return False
        if self._is_final() and self.fallback:
            return False
        found = self._best_instance(world, current, pose, final=self._is_final())
        if found is None:
            return False
        if self.lock is not None and self.lock.name == f"L{found.id}":
            return False
        if self.lock is not None and self.lock.name.startswith("L"):
            return False
        self.lock = self._landmark_wp(found, world, pose, current.get("side") or "")
        return True

    def _landmark_wp(self, instance, world, pose, side):
        center = np.array([instance.x, instance.z], dtype=np.float64)
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        radius = max(float(instance.radius_m) + 0.5, 1.0)
        point, heading = stand_point(world.grid, center, agent, radius, side)
        return landmark_waypoint(instance, point, heading, radius)

    def _best_instance(self, world, step, pose, final=False):
        landmark = step.get("landmark") or ""
        if not landmark:
            return None
        labels = _labels(landmark)
        matches = []
        need = world.semantic.config.confirm_hits
        for instance in world.semantic.instances:
            if instance.hits < need or instance.label not in labels or instance.id in self.rejected:
                continue
            if final and not _strong(instance):
                continue
            if not self._near_ok(instance, step, world):
                continue
            matches.append(instance)
        if not matches:
            return None
        side = step.get("side") or ""
        sided = [item for item in matches if _side_ok(item, pose, side)]
        pool = sided or matches
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        pool.sort(key=lambda item: float(np.hypot(item.x - agent[0], item.z - agent[1])))
        return pool[0]

    def _near_ok(self, instance, step, world):
        near = step.get("near") or ""
        if not near:
            return True
        refs = [item for item in world.semantic.instances if item.label == near]
        if not refs:
            return True
        return min(float(np.hypot(instance.x - item.x, instance.z - item.z)) for item in refs) <= 2.0

    def _explored_ahead(self, world, pose, reach=1.2):
        """The explored doorway in front of the agent, when nothing else is selectable."""
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        best, best_dist = None, reach
        for track in world.portals.tracks:
            if track.state != "explored" or track.id in self.crossed:
                continue
            if self._banned(track.center):
                continue
            if abs(bearing_deg(pose, track.center)) > 75.0:
                continue
            dist = float(np.hypot(track.center[0] - agent[0], track.center[1] - agent[1]))
            if dist < best_dist:
                best, best_dist = track, dist
        if best is None:
            return None
        waypoint = portal_waypoint(best, agent)
        if world.grid is not None and plan(world.grid, agent, waypoint.stages[0].xz, snap_m=0.8) is None:
            return None
        return waypoint

    def _best_portal(self, world, pose, restrict, side):
        ranked = self._rank_portals(world, pose, restrict, side)
        if not ranked:
            return None
        self._log(ranked[0][2], ranked, [])
        return ranked[0][2]

    def _rank_portals(self, world, pose, restrict, side):
        """``(utility, align, waypoint, track, area, path)`` for each reachable active door, best first."""
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        ranked = []
        for track in world.active_portals():
            if restrict is not None and track.id not in restrict:
                continue
            if self._banned(track.center):
                continue
            scored = self._portal_utility(world, track, agent, pose, side)
            if scored is not None:
                ranked.append(scored)
        ranked.sort(key=lambda item: (round(item[0], 2), item[1]), reverse=True)
        return ranked

    def _portal_utility(self, world, track, agent, pose, side):
        waypoint = portal_waypoint(track, agent)
        route = None
        if world.grid is not None:
            route = plan(world.grid, agent, waypoint.stages[0].xz, snap_m=0.8)
            if route is None:
                return None
        length = _route_length(agent, route, waypoint.stages[0].xz)
        center = np.asarray(track.center, dtype=np.float64)
        normal = np.asarray(track.normal, dtype=np.float64)
        toward_agent = 1.0 if float((agent - center) @ normal) > 0.0 else -1.0
        width = float(getattr(track, "width", DOOR_WIDTH_BEST))
        area = 0.0
        if world.grid is not None:
            area = unknown_beyond(world.grid, center, -toward_agent * normal, width / 2.0)
        bonus = {"confirmed": DOOR_BONUS_CONFIRMED, "candidate": DOOR_BONUS_CANDIDATE}.get(track.state, 0.0)
        utility = (area + bonus * door_width_weight(width)) / (length + FRONTIER_D0)
        if not _side_ok_xz(center, pose, side):
            utility *= WRONG_SIDE
        return utility, self._align(center, agent, pose), waypoint, track, area, length

    def _rank_frontiers(self, world, pose):
        """Most unknown area per meter of path first. Heading only breaks near ties."""
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        ranked = []
        for frontier in world.frontiers:
            if self._banned(frontier.xz):
                continue
            route = None
            if world.grid is not None:
                route = plan(world.grid, agent, frontier.xz, snap_m=0.8)
                if route is None:
                    continue
            length = _route_length(agent, route, frontier.xz)
            area = float(getattr(frontier, "area", 0.0))
            utility = area / (length + FRONTIER_D0)
            waypoint = point_waypoint(f"F{frontier.id}", frontier.xz, radius=0.6, snap_m=0.8)
            ranked.append((utility, self._align(frontier.xz, agent, pose), waypoint, frontier, area, length))
        ranked.sort(key=lambda item: (round(item[0], 2), item[1]), reverse=True)
        return ranked

    def _align(self, xz, agent, pose):
        forward = self.prior if self.prior is not None else _agent_axes(pose)[0]
        delta = np.asarray(xz, dtype=np.float64) - agent
        dist = float(np.linalg.norm(delta))
        return 0.0 if dist < 1e-6 else float(np.dot(delta / dist, forward))

    def _lock_ok(self, world):
        name = self.lock.name
        if name.startswith("P"):
            track = world.portals.track(self.lock.portal_id)
            # Stay committed after the far side becomes visible, and finish the crossing.
            return track is not None and track.state in ("candidate", "confirmed", "explored")
        if name.startswith("L"):
            wanted = int(name[1:])
            need = world.semantic.config.confirm_hits
            return any(item.id == wanted and item.hits >= need for item in world.semantic.instances)
        if name.startswith("F"):
            wanted = int(name[1:])
            return any(item.id == wanted for item in world.frontiers)
        return True

    def _complete_ready(self, world, pose):
        for _ in range(len(self.steps) + 1):
            if not self._geometry_done(world, pose):
                return
            self._finish(world, pose)

    def _geometry_done(self, world, pose):
        current = self._current()
        if current is None or current["action"] in TURN_NAMES:
            return False
        if self._is_final():
            return self._final_done(world, pose, current)
        action = current["action"]
        landmark = current.get("landmark") or ""
        room = action == "exit" or (action == "go_to" and _place_only(landmark))
        if room or (action == "go_to" and _is_door(landmark)):
            # A span too wide or too diagonal to be a door is not a crossing, even if the path walked over it.
            crossed = [
                track for track in world.portals.tracks
                if track.traversed and track.id not in self.baseline
                and getattr(track, "reject", "") not in NOT_A_DOOR
            ]
            if crossed:
                return True
            return bool(self.arrived_name and self.arrived_name.startswith("P"))
        if action in ("go_to", "stop") and self.arrived_name and self.arrived_name.startswith("L"):
            return True
        return False

    def _final_done(self, world, pose, step):
        """The last step ends only after its go part, and never within 3 m of the start."""
        arrived = self.arrived_name or ""
        action = step["action"]
        landmark = step.get("landmark") or ""
        if arrived == "R" or (self.fallback and self._room_waypoint(world, pose) is None):
            if self._may_stop(pose):
                return True
            if arrived == "R":
                self.room_rejected = True
                self.arrived_name = None
            return False
        if action == "exit" or _place_only(landmark):
            if not arrived.startswith("L"):
                return False
        elif action in ("go_to", "stop") and not landmark:
            return self._may_stop(pose)
        elif _is_door(landmark):
            if not arrived.startswith("P"):
                return False
        elif not arrived.startswith("L"):
            return False
        if self._may_stop(pose):
            return True
        if arrived.startswith("L") and arrived[1:].isdigit():
            self.rejected.add(int(arrived[1:]))
        self.arrived_name = None
        return False

    def _skip_if_needed(self, world, pose):
        if self.tried_skip or self._current() is None:
            return
        used = self.step - self.start_step
        if used < 40 and self.traveled < 8.0:
            return
        self.tried_skip = True
        last = len(self.steps) - 1
        for index in range(self.index + 1, len(self.steps)):
            step = self.steps[index]
            if step["action"] not in ("go_to", "stop"):
                continue
            if _place_only(step.get("landmark") or ""):
                continue
            if self._best_instance(world, step, pose, final=index == last) is None:
                continue
            self.index = index
            self.lock = None
            self.arrived_name = None
            self._begin(world, pose)
            return

    def _finish(self, world, pose):
        self.index += 1
        self.lock = None
        self.arrived_name = None
        self.tried_skip = False
        self.did_backtrack = False
        self._begin(world, pose)

    def _begin(self, world, pose):
        self.baseline = {track.id for track in world.portals.tracks if track.traversed}
        self.start_step = self.step
        self.traveled = 0.0
        self.last_xz = np.asarray(pose, dtype=np.float64)[[0, 2], 3].copy()

    def _move(self, pose, step):
        self.step = int(step)
        xz = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        if self.last_xz is not None:
            self.traveled += float(np.hypot(xz[0] - self.last_xz[0], xz[1] - self.last_xz[1]))
        self.last_xz = xz.copy()

    def _current(self):
        if self.index < 0 or self.index >= len(self.steps):
            return None
        return self.steps[self.index]

    def _banned(self, xz):
        point = np.asarray(xz, dtype=np.float64)
        return any(float(np.hypot(point[0] - item[0], point[1] - item[1])) < 0.75 for item in self.banned)


def prompts_for(parsed, index):
    """Object phrases for the current step, the next step, and the final step.

    A room name cannot be segmented, so it is replaced by objects that mark the room.
    The final landmark also asks for its common other names.
    """
    steps = [] if not parsed else list(parsed.get("subtasks") or [])
    if not steps:
        return []
    found = []
    last = len(steps) - 1
    for cursor in (index, index + 1, last):
        if cursor < 0 or cursor >= len(steps):
            continue
        step = steps[cursor]
        for key in ("landmark", "near"):
            text = _landmark(step.get(key, ""))
            if not text:
                continue
            if _place_only(text):
                names = _room_anchor_labels(text) if key == "landmark" else ()
            elif key == "landmark" and cursor == last:
                names = sorted(_labels(text), key=lambda name: name != text)
            else:
                names = (text,)
            for name in names:
                if name not in found:
                    found.append(name)
    return found


def room_point(grid, door_center, inward, radius=ROOM_RADIUS_M, depth=ROOM_DEPTH_M):
    """A free, explored point inside the space beyond a doorway.

    The centroid of explored floor on the far side within ``radius`` meters is
    used when it sits at least ``depth`` meters past the door; otherwise the
    point ``depth`` meters straight in. Either is snapped to the nearest
    explored free cell on that side.
    """
    center = np.asarray(door_center, dtype=np.float64)
    normal = np.asarray(inward, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    normal = normal / norm
    explored = getattr(grid, "explored", None)
    if explored is None:
        return None
    reach = int(np.ceil(radius / grid.res))
    col0 = int(np.floor((center[0] - grid.origin[0]) / grid.res))
    row0 = int(np.floor((center[1] - grid.origin[1]) / grid.res))
    rows = slice(max(0, row0 - reach), min(explored.shape[0], row0 + reach + 1))
    cols = slice(max(0, col0 - reach), min(explored.shape[1], col0 + reach + 1))
    free = explored[rows, cols] & (grid.logodds[rows, cols] < grid.config.free_logodds)
    rr, cc = np.nonzero(free)
    if rr.size == 0:
        return None
    xs = grid.origin[0] + (cc + cols.start + 0.5) * grid.res
    zs = grid.origin[1] + (rr + rows.start + 0.5) * grid.res
    points = np.column_stack((xs, zs))
    offset = points - center
    ahead = offset @ normal
    keep = (ahead >= 0.5) & (np.hypot(offset[:, 0], offset[:, 1]) <= radius)
    if not np.any(keep):
        return None
    inside = points[keep]
    target = inside.mean(axis=0)
    if float((target - center) @ normal) < depth:
        target = center + normal * depth
    nearest = int(np.argmin(np.hypot(inside[:, 0] - target[0], inside[:, 1] - target[1])))
    return inside[nearest].copy()


def door_width_weight(width):
    """1 at 1.5 m, falling off as a Gaussian toward the 0.8 m and 2.2 m limits."""
    return float(np.exp(-((float(width) - DOOR_WIDTH_BEST) ** 2) / (2.0 * DOOR_WIDTH_SIGMA ** 2)))


def _side_ok_xz(xz, pose, side):
    if side not in ("left", "right"):
        return True
    bearing = bearing_deg(pose, xz)
    return bearing <= 15.0 if side == "left" else bearing >= -15.0


def _route_length(agent, route, goal):
    if not route:
        return float(np.hypot(*(np.asarray(goal, dtype=np.float64) - agent)))
    points = np.vstack([agent, np.asarray(route, dtype=np.float64)])
    return float(np.sum(np.hypot(*np.diff(points, axis=0).T)))


def _labels(landmark):
    text = _landmark(landmark) or str(landmark)
    return {text, str(landmark), *SYNONYMS.get(text, ())}


def _room_anchor_labels(landmark):
    text = _landmark(landmark) or str(landmark)
    for room, anchors in ROOM_ANCHORS.items():
        if room in text:
            return anchors
    return ()


def _strong(instance):
    hits = max(int(instance.hits), 1)
    return hits >= FINAL_HITS and float(instance.score_sum) / hits >= FINAL_SCORE


def _near_look(looks, xz, radius):
    point = np.asarray(xz, dtype=np.float64)
    for look in looks:
        item = np.asarray(look, dtype=np.float64)
        if float(np.hypot(item[0] - point[0], item[1] - point[1])) <= radius:
            return True
    return False


def _side_ok(instance, pose, side):
    if side not in ("left", "right"):
        return True
    bearing = bearing_deg(pose, [instance.x, instance.z])
    if side == "left":
        return bearing < 15.0
    return bearing > -15.0
