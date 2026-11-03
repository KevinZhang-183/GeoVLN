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
    parsed = {
        "end_landmark": "chair",
        "subtasks": [{"id": 1, "action": "go_to", "landmark": "chair", "near": "", "side": ""}],
    }
    policy = Policy("goal", parsed, np.array([0.0, 0.0, 0.2]), allow_look=False)
    assert policy.prompts() == []
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
        return next((item for item in self.tracks if item.id == track_id), None)

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


def check_look_budget():
    from geonav.loop import ENTRY_LOOK_M, LOOK_TURNS, budget_step, entry_look_due, look_covers_entry

    step, charged = 0, False
    for _ in range(LOOK_TURNS):
        step, charged = budget_step(step, True, charged)
    assert step == 1 and charged
    step, charged = budget_step(step, False, charged)
    assert step == 2 and not charged
    anchor = np.array([0.0, 0.0])
    forward = np.array([1.0, 0.0])
    assert not entry_look_due(anchor, forward, np.array([ENTRY_LOOK_M - 0.01, 0.0]))
    assert entry_look_due(anchor, forward, np.array([ENTRY_LOOK_M, 0.0]))
    assert not entry_look_due(anchor, forward, np.array([-ENTRY_LOOK_M - 0.1, 0.0]))
    assert not entry_look_due(None, forward, np.array([1.0, 0.0]))
    assert look_covers_entry(np.array([0.2, 0.0]), anchor)
    assert not look_covers_entry(np.array([3.0, 0.0]), anchor)
    assert not look_covers_entry(None, anchor)


def check_explored_door():
    """A doorway that has become explored is still the way through."""
    grid = make_grid()
    world = _FakeWorld(grid)
    door = _Track(4, [0.0, 1.0], [0.0, 1.0], "explored")
    world.portals.tracks.append(door)
    parsed = {
        "end_landmark": "chair",
        "subtasks": [{"id": 1, "action": "go_to", "landmark": "chair", "near": "", "side": ""}],
    }
    pose = pose_at(0.0, 0.0, (0.0, 1.0))
    policy = Policy("landmark", parsed, None, allow_look=True)
    decision = policy.decide(world, pose, 0)
    assert decision.kind == "waypoint" and decision.waypoint.name == "P4", decision.reason

    held = Policy("landmark", parsed, None, allow_look=True)
    held.lock = decision.waypoint
    held.ready = True
    again = held.decide(world, pose, 1)
    assert again.kind == "waypoint" and again.waypoint.name == "P4"

    behind = _Track(5, [0.0, -1.0], [0.0, 1.0], "explored")
    world.portals.tracks[:] = [behind]
    looking = Policy("landmark", parsed, None, allow_look=True)
    assert looking.decide(world, pose, 0).kind == "look"


class _Track:
    def __init__(self, track_id, center, normal, state):
        self.id = track_id
        self.center = np.asarray(center, dtype=np.float64)
        self.normal = np.asarray(normal, dtype=np.float64)
        self.state = state
        self.traversed = False


def check_landmark_placement():
    from semantic_map import mask_points

    config = SemanticConfig(erode_px=0, border_px=0, min_pixels=10, depth_band_m=0.5, pixel_stride=1, min_score=0.0)
    depth = np.full((120, 160), 3.0, dtype=np.float32)
    wall = np.zeros_like(depth, dtype=bool)
    wall[10:110, 30:120] = True
    assert len(mask_points(wall, depth, config)) == 0
    chair = np.zeros_like(depth, dtype=bool)
    chair[55:75, 70:100] = True
    assert len(mask_points(chair, depth, config)) >= 10

    grid = make_grid()
    # Column covering x in [2.0, 2.2).
    grid.logodds[:, 35] = grid.config.occupied_logodds + 2.0
    semantic = SemanticMap(config)
    semantic.attach_grid(grid)
    pose = pose_at(0.0, 0.0, (1.0, 0.0))
    cloud = np.array([[2.05, -0.1], [2.05, 0.0], [2.05, 0.1], [2.1, 0.0]], dtype=np.float64)
    placed = semantic._off_obstacle(cloud, pose)
    assert placed is not None
    center = placed.mean(axis=0)
    assert center[0] < 1.8
    assert int(grid.codes(center.reshape(1, 2))[0]) != 2

    semantic.instances = [
        Instance(1, "chair", [1.7, 0.0], 0.2, 0.8, "a"),
        Instance(2, "chair", [1.7, 1.2], 0.2, 0.8, "b"),
        Instance(3, "table", [0.0, 3.0], 0.2, 0.8, "c"),
    ]
    assert semantic.merge_overlapping() == 1
    assert sorted(item.label for item in semantic.instances) == ["chair", "table"]


def check_vis_video():
    import tempfile
    from pathlib import Path

    from PIL import Image

    from geonav.video import encode_saved_vis

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "scene_1"
        vis = root / "vis"
        vis.mkdir(parents=True)
        Image.new("RGB", (64, 48), (255, 0, 0)).save(vis / "0000.jpg")
        Image.new("RGB", (64, 48), (0, 255, 0)).save(vis / "0010.jpg")
        written = encode_saved_vis(root, fps=5)
        if not written:
            print("vis video encoder unavailable, skipped")
            return
        assert written[0].is_file() and written[0].stat().st_size > 0


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
    check_look_budget()
    check_explored_door()
    check_landmark_placement()
    check_vis_video()
    print("geonav self_check ok")


if __name__ == "__main__":
    main()
