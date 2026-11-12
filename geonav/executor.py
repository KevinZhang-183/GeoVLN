"""Follow one waypoint with A* and the simulator's discrete actions."""

from __future__ import annotations

import numpy as np

from geonav.geom import INFLATE_M, bearing_deg, block_ahead, heading_error_deg, lookahead, plan

FORWARD = "move_forward"
LEFT = "turn_left"
RIGHT = "turn_right"
# One simulator turn is 15 degrees, one step is 0.25 m. Arrival accepts 30
# degrees. A full circle of turns with no step means the aim is not usable.
ALIGN_DEG = 30.0
STEP_M = 0.25
TURN_LIMIT = 24


class Executor:
    def __init__(self, inflate_m=INFLATE_M, lookahead_m=0.6, turn_deg=10.0, align_deg=ALIGN_DEG):
        self.inflate_m = inflate_m
        self.lookahead_m = lookahead_m
        self.turn_deg = turn_deg
        self.align_deg = align_deg
        self.waypoint = None
        self.stage = 0
        self.path = []
        self._forwards = []
        self.stuck_count = 0
        self.abandoned = False
        self.reached = False
        self._align_heading = None
        self._align_stage = None
        self._progress_aim = None
        self._progress_stage = None
        self._turns = 0

    def clear(self):
        self.waypoint = None
        self.stage = 0
        self.path = []
        self._forwards = []
        self.stuck_count = 0
        self.abandoned = False
        self.reached = False
        self._align_heading = None
        self._align_stage = None
        self._progress_aim = None
        self._progress_stage = None
        self._turns = 0

    def set_waypoint(self, waypoint):
        if waypoint is None:
            self.clear()
            return
        if self.waypoint is not None and self.waypoint.name == waypoint.name and not self.abandoned:
            self.waypoint = waypoint
            stage = min(self.stage, len(waypoint.stages) - 1)
            if stage != self.stage:
                self._forget_aim()
                self.stage = stage
            return
        self.waypoint = waypoint
        self.stage = 0
        self.path = []
        self._forwards = []
        self.stuck_count = 0
        self.abandoned = False
        self.reached = False
        self._forget_aim()
        self._turns = 0

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
            if target.heading is not None and not self._heading_ok(pose, target):
                return self._give_turn(heading_error_deg(pose, self._align_heading))
            self._forget_aim()
            self._turns = 0
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
            return self._give_turn(-1.0)
        aim = self._follow_aim(route, agent, target)
        error = bearing_deg(pose, aim)
        if abs(error) > self.turn_deg:
            return self._give_turn(error)
        self._turns = 0
        self._forwards.append(agent.copy())
        return FORWARD

    def _follow_aim(self, route, agent, target):
        """Steer along the path, or to the stage point once the path has run out.

        A lookahead under the agent's feet has no stable bearing. The door point
        is also refit every frame, so latch the stage point the moment the path
        stops leading anywhere.
        """
        aim = lookahead(route, agent, self.lookahead_m)
        if aim is not None and float(np.hypot(aim[0] - agent[0], aim[1] - agent[1])) >= STEP_M:
            self._progress_aim = None
            self._progress_stage = None
            return aim
        if self._progress_stage != self.stage or self._progress_aim is None:
            self._progress_aim = np.array(target.xz, dtype=np.float64, copy=True)
            self._progress_stage = self.stage
        return self._progress_aim

    def _give_turn(self, error):
        self._turns += 1
        if self._turns > TURN_LIMIT:
            self.abandoned = True
            return None
        return _turn(error)

    def _forget_aim(self):
        self._align_heading = None
        self._align_stage = None
        self._progress_aim = None
        self._progress_stage = None

    def _heading_ok(self, pose, target):
        """Face the heading this stage had when the agent arrived.

        The portal normal is refit every frame. Chasing that moving direction
        turns in place and never steps through the door.
        """
        if self._align_stage != self.stage or self._align_heading is None:
            self._align_heading = np.array(target.heading, dtype=np.float64, copy=True)
            self._align_stage = self.stage
        return abs(heading_error_deg(pose, self._align_heading)) <= self.align_deg

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
