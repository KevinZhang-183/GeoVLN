"""Checks that do not need Habitat or SAM3. Run from GeoVLN: python -m geonav.self_check"""

from __future__ import annotations

import numpy as np

from geonav._bootstrap import install

install()

from portal_map import MapConfig, OccupancyGrid  # noqa: E402
from semantic_map import Instance, SemanticConfig, SemanticMap  # noqa: E402

from geonav.executor import Executor  # noqa: E402
from geonav.geom import bearing_deg, frontiers, plan  # noqa: E402
from geonav.policy import Policy  # noqa: E402
from geonav.waypoints import portal_waypoint  # noqa: E402


def pose_at(x, z, forward):
    forward = np.asarray(forward, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    right = np.array([-forward[1], forward[0]])
    pose = np.eye(4)
    pose[0, 0], pose[2, 0] = right
    pose[0, 2], pose[2, 2] = -forward
    pose[0, 3], pose[2, 3] = x, z
    return pose


def make_grid(free=True):
    config = MapConfig()
    config.resolution = 0.2
    config.size_m = 10.0
    grid = OccupancyGrid(np.array([0.0, 0.0]), config)
    if free:
        grid.logodds[:] = config.free_logodds - 0.5
    return grid


def check_bearing():
    pose = pose_at(0.0, 0.0, (0.0, 1.0))
    assert abs(bearing_deg(pose, [0.0, 3.0])) < 1.0
    assert bearing_deg(pose, [-3.0, 0.0]) > 45.0
    assert bearing_deg(pose, [3.0, 0.0]) < -45.0


def check_plan():
    grid = make_grid()
    wall = 25
    grid.logodds[:, wall] = grid.config.occupied_logodds + 2.0
    grid.logodds[20:30, wall] = grid.config.free_logodds - 0.5
    route = plan(grid, [-2.0, 0.0], [2.0, 0.0], inflate_m=0.15, coarse_m=0.2)
    assert route, "path should cross the gap"
    grid.logodds[:, wall] = grid.config.occupied_logodds + 2.0
    assert plan(grid, [-2.0, 0.0], [2.0, 0.0], inflate_m=0.15, coarse_m=0.2) is None


def check_frontiers():
    grid = make_grid(free=False)
    grid.logodds[:] = 0.0
    grid.logodds[:, :25] = grid.config.free_logodds - 0.5
    points = frontiers(grid, np.array([-2.0, 0.0]), min_m=0.5, pull_m=0.5)
    assert points, "open space should produce a frontier"
    assert max(point[0] for point in points) < 0.5


def check_stuck():
    grid = make_grid()
    pose = pose_at(0.0, 0.0, (0.0, 1.0))
    executor = Executor()
    executor.set_waypoint(_goal_waypoint())
    for _ in range(40):
        action = executor.act(pose, grid)
        if action in ("turn_left", "turn_right"):
            pose = _turned(pose, action)
        if executor.abandoned:
            break
    assert executor.abandoned, "standing still in front of a waypoint should be abandoned"


def _turned(pose, action):
    forward = -pose[[0, 2], 2]
    right = pose[[0, 2], 0]
    left = -right
    angle = np.deg2rad(15.0)
    side = left if action == "turn_left" else right
    new_forward = forward * np.cos(angle) + side * np.sin(angle)
    return pose_at(float(pose[0, 3]), float(pose[2, 3]), new_forward)


def check_portal_side():
    class Track:
        id = 3
        center = np.array([2.0, 0.0])
        normal = np.array([1.0, 0.0])

    waypoint = portal_waypoint(Track(), np.array([0.0, 0.0]))
    front, back = waypoint.stages
    assert abs(front.xz[0] - 1.5) < 1e-6
    assert abs(back.xz[0] - 2.8) < 1e-6
    assert waypoint.name == "P3"


def check_merge():
    semantic = SemanticMap(SemanticConfig(assoc_dist_m=0.8))
    semantic.instances = [
        Instance(2, "chair", [0.0, 0.0], 0.2, 0.8, "a"),
        Instance(5, "chair", [0.4, 0.0], 0.2, 0.9, "b"),
        Instance(7, "table", [3.0, 0.0], 0.2, 0.9, "c"),
    ]
    assert semantic.merge_overlapping() == 1
    labels = sorted(item.label for item in semantic.instances)
    assert labels == ["chair", "table"]
    chair = next(item for item in semantic.instances if item.label == "chair")
    assert chair.id == 2 and chair.hits == 2


def check_shared_grid():
    grid = make_grid()
    before = float(grid.logodds[0, 0])
    semantic = SemanticMap()
    semantic.attach_grid(grid)
    semantic.note_frame(pose_at(0.0, 0.0, (0.0, 1.0)), depth=np.ones((8, 8), dtype=np.float32))
    assert float(grid.logodds[0, 0]) == before
    assert semantic.path


def check_landmark_policy():
    grid = make_grid()
    world = _FakeWorld(grid)
    chair = Instance(4, "chair", [0.0, 3.0], 0.2, 0.95, "0004")
    chair.hits = 2
    world.semantic.instances.append(chair)
    parsed = {
        "end_landmark": "chair",
        "subtasks": [
            {"id": 1, "action": "turn_left", "landmark": "", "near": "", "side": ""},
            {"id": 2, "action": "go_to", "landmark": "chair", "near": "", "side": ""},
            {"id": 3, "action": "stop", "landmark": "chair", "near": "", "side": ""},
        ],
    }
    policy = Policy("landmark", parsed, None, allow_look=False)
    pose = pose_at(0.0, 0.0, (0.0, 1.0))
    reasons = [policy.decide(world, pose, step).reason for step in range(6)]
    assert reasons == ["turn_left"] * 6, reasons
    decision = policy.decide(world, pose, 6)
    assert decision.kind == "waypoint" and decision.waypoint.name == "L4", decision.reason


def check_goal_stop():
    grid = make_grid()
    world = _FakeWorld(grid)
    policy = Policy("goal", None, np.array([0.0, 0.0, 0.2]), allow_look=False)
    decision = policy.decide(world, pose_at(0.0, 0.0, (0.0, 1.0)), 0)
    assert decision.kind == "stop"


class _Semantic:
    def __init__(self):
        self.instances = []
        self.config = SemanticConfig()


class _Portals:
    def __init__(self, grid):
        self.grid = grid
        self.tracks = []

    def track(self, track_id):
        return None

    def abandon(self, track_id):
        return None


class _FakeWorld:
    def __init__(self, grid):
        self.grid = grid
        self.semantic = _Semantic()
        self.portals = _Portals(grid)
        self.frontiers = []
        self.nodes = []
        self.new_traversals = []

    def active_portals(self):
        return []


def _goal_waypoint():
    from geonav.waypoints import point_waypoint

    return point_waypoint("G", [0.0, 4.0], radius=0.3, snap_m=1.0)


def main():
    check_bearing()
    check_plan()
    check_frontiers()
    check_portal_side()
    check_merge()
    check_shared_grid()
    check_stuck()
    check_landmark_policy()
    check_goal_stop()
    print("geonav self_check ok")


if __name__ == "__main__":
    main()
