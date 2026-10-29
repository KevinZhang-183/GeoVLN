"""Waypoint geometry. A waypoint is one or two stages the executor walks in order."""

from __future__ import annotations

import numpy as np

from geonav._bootstrap import install

install()

from portal_map import _unit  # noqa: E402


class Stage:
    def __init__(self, xz, radius=0.3, heading=None, snap_m=1.0):
        self.xz = np.asarray(xz, dtype=np.float64)
        self.radius = float(radius)
        self.heading = None if heading is None else _unit(heading)
        self.snap_m = float(snap_m)


class Waypoint:
    def __init__(self, name, stages, portal_id=None):
        self.name = name
        self.stages = list(stages)
        self.portal_id = portal_id

    @property
    def xz(self):
        return self.stages[0].xz if self.stages else np.zeros(2)


def point_waypoint(name, xz, radius, heading=None, snap_m=1.0, portal_id=None):
    return Waypoint(name, [Stage(xz, radius, heading, snap_m)], portal_id)


def portal_waypoint(track, agent_xz):
    """Approach from the agent's side, then step through along the portal normal."""
    center = np.asarray(track.center, dtype=np.float64)
    normal = _unit(track.normal)
    toward_agent = 1.0 if float((np.asarray(agent_xz, dtype=np.float64) - center) @ normal) > 0.0 else -1.0
    front = center + normal * toward_agent * 0.5
    back = center - normal * toward_agent * 0.8
    heading = _unit(back - front)
    stages = [
        Stage(front, radius=0.3, heading=heading, snap_m=0.8),
        Stage(back, radius=0.3, heading=heading, snap_m=1.2),
    ]
    return Waypoint(f"P{track.id}", stages, portal_id=track.id)


def landmark_waypoint(instance, point, heading, snap_m):
    """Stand at ``point``, facing the instance."""
    return Waypoint(
        f"L{instance.id}",
        [Stage(point, radius=1.0, heading=heading, snap_m=snap_m)],
        portal_id=getattr(instance, "portal_id", None),
    )


def stand_point(grid, center, agent_xz, radius, side=""):
    """Pick a free cell on the circle around ``center``. ``side`` is left or right of the approach."""
    center = np.asarray(center, dtype=np.float64)
    agent = np.asarray(agent_xz, dtype=np.float64)
    if grid is None:
        return center, None
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    circle = np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], axis=1)
    codes = grid.codes(circle)
    keep = codes == 1
    if side in ("left", "right"):
        approach = center - agent
        norm = float(np.linalg.norm(approach))
        if norm > 1e-6:
            forward = approach / norm
            right = np.array([-forward[1], forward[0]])
            lateral = (circle - center) @ right
            if side == "left":
                keep = keep & (lateral < 0.0)
            else:
                keep = keep & (lateral > 0.0)
    if not np.any(keep):
        return center, _unit(center - agent) if float(np.linalg.norm(center - agent)) > 1e-6 else None
    choices = circle[keep]
    nearest = int(np.argmin(np.linalg.norm(choices - agent, axis=1)))
    point = choices[nearest]
    heading = _unit(center - point)
    return point, heading
