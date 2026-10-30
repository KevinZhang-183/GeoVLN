"""Follow one waypoint with A* and the simulator's discrete actions."""

from __future__ import annotations

import numpy as np

from geonav.geom import bearing_deg, block_ahead, heading_error_deg, lookahead, plan

FORWARD = "move_forward"
LEFT = "turn_left"
RIGHT = "turn_right"


class Executor:
    def __init__(self, inflate_m=0.15, lookahead_m=0.6, turn_deg=10.0):
        self.inflate_m = inflate_m
        self.lookahead_m = lookahead_m
        self.turn_deg = turn_deg
        self.waypoint = None
        self.stage = 0
        self.path = []
        self._forwards = []
        self.stuck_count = 0
        self.abandoned = False
        self.reached = False

    def clear(self):
        self.waypoint = None
        self.stage = 0
        self.path = []
        self._forwards = []
        self.stuck_count = 0
        self.abandoned = False
        self.reached = False

    def set_waypoint(self, waypoint):
        if waypoint is None:
            self.clear()
            return
        if self.waypoint is not None and self.waypoint.name == waypoint.name and not self.abandoned:
            self.waypoint = waypoint
            self.stage = min(self.stage, len(waypoint.stages) - 1)
            return
        self.waypoint = waypoint
        self.stage = 0
        self.path = []
        self._forwards = []
        self.stuck_count = 0
        self.abandoned = False
        self.reached = False

    def target(self):
        if self.waypoint is None or self.stage >= len(self.waypoint.stages):
            return None
        return self.waypoint.stages[self.stage]

    def act(self, pose, grid):
        """Return the next discrete action name, or None when the waypoint is done or given up."""
        if self.abandoned:
            return None
        target = self.target()
        if target is None:
            self.reached = self.waypoint is not None
            return None
        agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
        if float(np.hypot(agent[0] - target.xz[0], agent[1] - target.xz[1])) <= target.radius:
            if target.heading is not None and abs(heading_error_deg(pose, target.heading)) > self.turn_deg:
                return _turn(heading_error_deg(pose, target.heading))
            self.stage += 1
            self._forwards = []
            self.stuck_count = 0
            self.path = []
            if self.target() is None:
                self.reached = True
                return None
            return self.act(pose, grid)
        if self._stuck(agent):
            block_ahead(grid, pose)
            self._forwards = []
            self.path = []
            self.stuck_count += 1
            if self.stuck_count >= 2:
                self.abandoned = True
                return None
        route = self.path if _path_open(self.path, grid, agent, target) else None
        if not route:
            route = plan(
                grid,
                agent,
                target.xz,
                inflate_m=self.inflate_m,
                snap_m=target.snap_m,
            )
            self.path = route or []
        if not route:
            self.stuck_count += 1
            if self.stuck_count >= 2:
                self.abandoned = True
                return None
            return LEFT
        aim = lookahead(route, agent, self.lookahead_m)
        error = bearing_deg(pose, aim)
        if abs(error) > self.turn_deg:
            return _turn(error)
        self._forwards.append(agent.copy())
        return FORWARD

    def _stuck(self, agent):
        if len(self._forwards) < 5:
            return False
        start = self._forwards[-5]
        return float(np.hypot(agent[0] - start[0], agent[1] - start[1])) < 0.1


def _path_open(path, grid, agent, target):
    """Keep a plan while it still reaches this target and does not cross a wall.

    Replanning through unknown space every step flips between equal paths, and
    the agent turns in place instead of walking.
    """
    if not path or grid is None:
        return False
    points = np.asarray(path, dtype=np.float64)
    end = points[-1]
    if float(np.hypot(end[0] - target.xz[0], end[1] - target.xz[1])) > max(target.snap_m, 1.0):
        return False
    nearest = np.min(np.hypot(points[:, 0] - agent[0], points[:, 1] - agent[1]))
    if float(nearest) > 1.25:
        return False
    return not np.any(grid.codes(points) == 2)


def _turn(error_deg):
    return RIGHT if error_deg > 0.0 else LEFT
