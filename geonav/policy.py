"""Rule policies. VLM selection is a later step; landmarks are chosen geometrically."""

from __future__ import annotations

import numpy as np

from geonav._bootstrap import install

install()

from parse_instructions import _landmark, _place_only  # noqa: E402
from portal_map import _agent_axes  # noqa: E402

from geonav.geom import bearing_deg, plan  # noqa: E402
from geonav.waypoints import landmark_waypoint, point_waypoint, portal_waypoint, stand_point  # noqa: E402
from geonav.world import _is_door  # noqa: E402

TURNS = {
    "turn_left": ("turn_left", 6),
    "turn_right": ("turn_right", 6),
    "turn_around": ("turn_left", 12),
}
TURN_NAMES = set(TURNS)


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
        self.baseline = set()
        self.start_step = 0
        self.traveled = 0.0
        self.last_xz = None
        self.step = 0
        self.ready = False

    def prompts(self):
        index = self.index if self.mode == "landmark" else 0
        return prompts_for(self.parsed, index)

    def decide(self, world, pose, step):
        self._move(pose, step)
        if not self.ready:
            self._begin(world, pose)
            self.ready = True
        if self.mode == "goal":
            return self._goal(pose)
        if self.pending:
            return Decision("action", action=self.pending.pop(0), reason=self.pending_reason)
        if self.turn_sets_prior:
            self.prior = _agent_axes(pose)[0].copy()
            self.turn_sets_prior = False
            self._finish(world, pose)
        if self.mode == "landmark":
            self._skip_if_needed(world, pose)
            self._complete_ready(world, pose)
            if self._current() is None:
                return Decision("stop", reason="finished")
            if self.index >= len(self.steps) - 2 and self._near_end(world, pose):
                return Decision("stop", reason="end landmark")
            current = self._current()
            if current["action"] in TURN_NAMES:
                action, count = TURNS[current["action"]]
                self.pending = [action] * count
                self.pending_reason = current["action"]
                self.turn_sets_prior = True
                return Decision("action", action=self.pending.pop(0), reason=self.pending_reason)
            if current["action"] == "stop" and _place_only(current.get("landmark") or ""):
                return Decision("stop", reason="room")
        if self._object_upgrade(world, pose):
            return Decision("waypoint", waypoint=self.lock, reason=self.lock.name)
        if self.lock is not None and self._lock_ok(world):
            return Decision("waypoint", waypoint=self.lock, reason=self.lock.name)
        choice = self._choose(world, pose)
        if choice is not None:
            self.lock = choice
            self.did_backtrack = False
            return Decision("waypoint", waypoint=choice, reason=choice.name)
        if self.allow_look:
            return Decision("look", reason="no candidate")
        if not self.did_backtrack and len(world.nodes) >= 2:
            node = world.nodes[-2]
            self.did_backtrack = True
            self.lock = point_waypoint(f"N{node.id}", node.xz, radius=0.6, snap_m=1.0)
            return Decision("waypoint", waypoint=self.lock, reason="backtrack")
        return Decision("stop", reason="no candidate")

    def note_reached(self, waypoint):
        self.arrived_name = None if waypoint is None else waypoint.name
        if waypoint is not None and (waypoint.name[:1] in ("F", "N") or self.mode != "landmark"):
            self.banned.append(np.asarray(waypoint.stages[-1].xz, dtype=np.float64))
        self.lock = None

    def abandon(self, waypoint, world):
        if waypoint is None:
            return
        self.banned.append(np.asarray(waypoint.stages[-1].xz, dtype=np.float64))
        if waypoint.portal_id is not None:
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
            if current is not None and current["action"] in ("go_to", "stop"):
                if not _place_only(current.get("landmark") or "") and not _is_door(current.get("landmark") or ""):
                    found = self._best_instance(world, current, pose)
                    if found is not None:
                        return self._landmark_wp(found, world, pose, current.get("side") or "")
                if (
                    current["action"] == "exit"
                    or _place_only(current.get("landmark") or "")
                    or _is_door(current.get("landmark") or "")
                ):
                    choice = self._door_choice(world, pose, current)
                    if choice is not None:
                        return choice
        portal = self._best_portal(world, pose, None, "")
        if portal is not None:
            return portal
        return self._best_frontier(world, pose)

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

    def _object_upgrade(self, world, pose):
        if self.mode != "landmark":
            return False
        current = self._current()
        if current is None or current["action"] not in ("go_to", "stop"):
            return False
        landmark = current.get("landmark") or ""
        if _place_only(landmark) or _is_door(landmark):
            return False
        found = self._best_instance(world, current, pose)
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

    def _best_instance(self, world, step, pose):
        landmark = step.get("landmark") or ""
        if not landmark:
            return None
        matches = []
        need = world.semantic.config.confirm_hits
        for instance in world.semantic.instances:
            if instance.hits < need or instance.label != landmark:
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

    def _best_portal(self, world, pose, restrict, side):
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        ranked = []
        for track in world.active_portals():
            if restrict is not None and track.id not in restrict:
                continue
            if self._banned(track.center):
                continue
            waypoint = portal_waypoint(track, agent)
            if world.grid is not None and plan(world.grid, agent, waypoint.stages[0].xz, snap_m=0.8) is None:
                continue
            ranked.append((self._score(track.center, pose, track.state == "confirmed", side), waypoint))
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]

    def _best_frontier(self, world, pose):
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        best, best_score = None, 1e9
        for frontier in world.frontiers:
            if self._banned(frontier.xz):
                continue
            if world.grid is not None and plan(world.grid, agent, frontier.xz, snap_m=0.8) is None:
                continue
            score = self._score(frontier.xz, pose, False, "")
            if score < best_score:
                best, best_score = frontier, score
        if best is None:
            return None
        return point_waypoint(f"F{best.id}", best.xz, radius=0.6, snap_m=0.8)

    def _score(self, xz, pose, confirmed, side):
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        delta = np.asarray(xz, dtype=np.float64) - agent
        dist = float(np.linalg.norm(delta))
        forward = self.prior if self.prior is not None else _agent_axes(pose)[0]
        align = 0.0 if dist < 1e-6 else float(np.dot(delta / dist, forward))
        score = dist - 2.0 * align
        if confirmed:
            score -= 1.0
        bearing = bearing_deg(pose, xz)
        if side == "left" and bearing > 15.0:
            score += 3.0
        if side == "right" and bearing < -15.0:
            score += 3.0
        return score

    def _lock_ok(self, world):
        name = self.lock.name
        if name.startswith("P"):
            track = world.portals.track(self.lock.portal_id)
            return track is not None and track.state in ("candidate", "confirmed")
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
        action = current["action"]
        landmark = current.get("landmark") or ""
        room = action == "exit" or (action == "go_to" and _place_only(landmark))
        if room or (action == "go_to" and _is_door(landmark)):
            if any(track.traversed and track.id not in self.baseline for track in world.portals.tracks):
                return True
            return bool(self.arrived_name and self.arrived_name.startswith("P"))
        if action in ("go_to", "stop") and self.arrived_name and self.arrived_name.startswith("L"):
            return True
        return False

    def _skip_if_needed(self, world, pose):
        if self.tried_skip or self._current() is None:
            return
        used = self.step - self.start_step
        if used < 40 and self.traveled < 8.0:
            return
        self.tried_skip = True
        for index in range(self.index + 1, len(self.steps)):
            step = self.steps[index]
            if step["action"] not in ("go_to", "stop"):
                continue
            if _place_only(step.get("landmark") or ""):
                continue
            if self._best_instance(world, step, pose) is None:
                continue
            self.index = index
            self.lock = None
            self.arrived_name = None
            self._begin(world, pose)
            return

    def _near_end(self, world, pose):
        landmark = self.parsed.get("end_landmark") or ""
        if not landmark or _place_only(landmark):
            return False
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        need = world.semantic.config.confirm_hits
        for instance in world.semantic.instances:
            if instance.label != landmark or instance.hits < need:
                continue
            if float(np.hypot(instance.x - agent[0], instance.z - agent[1])) <= 1.5:
                return True
        return False

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
    """Object phrases for the current step, the next step, and the final step."""
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
            if not text or _place_only(text) or text in found:
                continue
            found.append(text)
    return found


def _side_ok(instance, pose, side):
    if side not in ("left", "right"):
        return True
    bearing = bearing_deg(pose, [instance.x, instance.z])
    if side == "left":
        return bearing < 15.0
    return bearing > -15.0
