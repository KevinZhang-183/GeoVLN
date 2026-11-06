"""Checks that do not need Habitat or SAM3. Run from GeoVLN: python -m geonav.self_check"""

from __future__ import annotations

import numpy as np

from geonav._bootstrap import install

install()

from portal_map import MapConfig, OccupancyGrid, PortalConfig, PortalMap, PortalTrack  # noqa: E402
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


def _slab_hit(origin, direction, lo, hi):
    """Entry and exit parameters of a ray against one axis-aligned slab."""
    parallel = np.abs(direction) <= 1e-8
    inv = np.divide(1.0, direction, out=np.full_like(direction, np.inf), where=~parallel)
    t0 = (lo - origin) * inv
    t1 = (hi - origin) * inv
    t_near = np.minimum(t0, t1)
    t_far = np.maximum(t0, t1)
    inside = (origin >= lo) & (origin <= hi)
    t_near = np.where(parallel, np.where(inside, -np.inf, np.inf), t_near)
    t_far = np.where(parallel, np.where(inside, np.inf, -np.inf), t_far)
    return t_near, t_far


def _render_depth(width, height, hfov, camera_height, boxes, depth_max=10.0):
    """Depth in meters for a level camera. ``boxes`` are (x0, x1, z0, z1, h0, h1)."""
    from extract_portals import focal_px

    fx = focal_px(width, hfov)
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    dir_x = (uu - (width - 1) / 2.0) / fx
    dir_h = -((vv - (height - 1) / 2.0) / fx)
    dir_z = np.ones_like(dir_x)
    t_hit = np.full(dir_x.shape, np.inf)
    down = dir_h < -1e-8
    t_floor = np.full(dir_x.shape, np.inf)
    t_floor[down] = -camera_height / dir_h[down]
    t_hit = np.where(t_floor > 0.05, t_floor, t_hit)
    for x0, x1, z0, z1, h0, h1 in boxes:
        tnx, tfx = _slab_hit(0.0, dir_x, x0, x1)
        tnh, tfh = _slab_hit(camera_height, dir_h, h0, h1)
        tnz, tfz = _slab_hit(0.0, dir_z, z0, z1)
        enter = np.maximum(np.maximum(tnx, tnh), tnz)
        exit_ = np.minimum(np.minimum(tfx, tfh), tfz)
        take = (enter <= exit_) & (exit_ > 0.05)
        enter = np.where(enter > 0.05, enter, 0.05)
        closer = take & (enter < t_hit)
        t_hit = np.where(closer, enter, t_hit)
    depth = np.where(np.isfinite(t_hit), t_hit, depth_max).astype(np.float32)
    return np.clip(depth, 0.0, depth_max)


def _cell(grid, x, z):
    rows, cols, ok = grid.cells([[x, z]])
    assert bool(ok[0]), (x, z)
    return int(rows[0]), int(cols[0])


def _count_occupied(grid, x0, x1, z0, z1):
    xs = np.arange(x0, x1, grid.res)
    zs = np.arange(z0, z1, grid.res)
    xx, zz = np.meshgrid(xs, zs)
    pts = np.stack([xx.ravel(), zz.ravel()], axis=1)
    rows, cols, ok = grid.cells(pts)
    vals = grid.logodds[rows[ok], cols[ok]]
    return int((vals > grid.config.occupied_logodds).sum())


def check_occupancy_keeps_furniture():
    """Looking at a room marks that wedge explored, and a table stays occupied."""
    config = MapConfig()
    config.size_m = 12.0
    grid = OccupancyGrid(np.array([0.0, 2.0]), config)
    portal = PortalConfig()
    width, height = 160, 120
    table = (-0.4, 0.4, 1.5, 2.0, 0.0, 0.45)
    wall = (-2.5, 2.5, 3.9, 4.05, 0.0, 2.2)

    def depth_at(camera_z):
        boxes = [
            (table[0], table[1], table[2] - camera_z, table[3] - camera_z, table[4], table[5]),
            (wall[0], wall[1], wall[2] - camera_z, wall[3] - camera_z, wall[4], wall[5]),
        ]
        return _render_depth(width, height, portal.hfov, portal.camera_height, boxes, portal.depth_max)

    far = depth_at(0.0)
    for _ in range(4):
        grid.integrate(far, pose_at(0.0, 0.0, (0.0, 1.0)), portal)
    occupied = _count_occupied(grid, -0.35, 0.36, 1.55, 1.96)
    assert occupied > 40, occupied
    assert int((grid.logodds < config.free_logodds).sum()) > 400

    def value(x, z):
        row, col = _cell(grid, x, z)
        return float(grid.logodds[row, col])

    assert value(0.0, 0.4) < config.free_logodds
    assert value(0.8, 2.5) < config.free_logodds
    assert value(0.0, -1.5) == 0.0
    assert value(0.0, 2.8) >= config.free_logodds

    close = depth_at(0.9)
    assert float(close.min()) > 1.5
    for _ in range(8):
        grid.integrate(close, pose_at(0.0, 0.9, (0.0, 1.0)), portal)
    assert _count_occupied(grid, -0.35, 0.36, 1.55, 1.96) == occupied
    assert value(0.0, 2.8) >= config.free_logodds


def check_cleanup_keeps_legs():
    config = MapConfig()
    config.resolution = 0.05
    config.size_m = 4.0
    grid = OccupancyGrid(np.zeros(2), config)
    grid.logodds[:] = config.free_logodds - 1.0
    grid.logodds[40, 40] = 2.0
    grid.logodds[40, 41] = 2.0
    grid._hit_frames[40, 40] = 3
    grid._hit_frames[40, 41] = 3
    grid.logodds[40, 50] = 2.0
    grid._hit_frames[40, 50] = 4
    grid.logodds[40, 60] = 2.0
    grid._hit_frames[40, 60] = 1
    grid.logodds[50:52, 50:52] = 2.0
    grid._hit_frames[50:52, 50:52] = 3
    # Unknown pocket. The confirmed block is larger than 2x2 and does not
    # touch free space; the single-frame block should still be removed.
    grid.logodds[55:80, 55:80] = 0.0
    grid.logodds[72:76, 72:76] = 2.0
    grid._hit_frames[72:76, 72:76] = 4
    grid.logodds[60:64, 60:68] = 2.0
    grid._hit_frames[60:64, 60:68] = 1
    grid._cleanup()
    assert grid.logodds[40, 40] > config.occupied_logodds
    assert grid.logodds[40, 41] > config.occupied_logodds
    assert grid.logodds[40, 50] > config.occupied_logodds
    assert grid.logodds[40, 60] <= 0.0
    assert np.all(grid.logodds[50:52, 50:52] > config.occupied_logodds)
    assert np.all(grid.logodds[72:76, 72:76] > config.occupied_logodds)
    assert np.all(grid.logodds[60:64, 60:68] == 0.0)


def _opening(track_id, a, b, normal=(0.0, 1.0)):
    return PortalTrack(
        id=track_id,
        a=np.asarray(a, dtype=np.float64),
        b=np.asarray(b, dtype=np.float64),
        normal=np.asarray(normal, dtype=np.float64),
        full=True,
        weight=1.0,
        first_step=0,
        last_step=0,
    )


def _fill(grid, x0, z0, x1, z1, kind):
    xs = np.arange(x0, x1, grid.res)
    zs = np.arange(z0, z1, grid.res)
    xx, zz = np.meshgrid(xs, zs)
    pts = np.stack([xx.ravel(), zz.ravel()], axis=1)
    rows, cols, ok = grid.cells(pts)
    value = grid.config.occupied_logodds + 2.0 if kind == "occ" else grid.config.free_logodds - 1.0
    grid.logodds[rows[ok], cols[ok]] = value


def _door_map():
    """A doorway at z=0. The near room (negative z) is wider; the far side stays unknown."""
    config = MapConfig()
    config.resolution = 0.05
    config.size_m = 12.0
    grid = OccupancyGrid(np.zeros(2), config)
    _fill(grid, -1.7, -2.2, 1.7, -0.25, "free")
    _fill(grid, -0.48, -0.25, 0.48, 0.25, "free")
    _fill(grid, -2.05, -2.2, -1.7, -0.25, "occ")
    _fill(grid, 1.7, -2.2, 2.05, -0.25, "occ")
    _fill(grid, -0.85, -0.20, -0.50, 0.20, "occ")
    _fill(grid, 0.50, -0.20, 0.85, 0.20, "occ")
    portal_map = PortalMap(config)
    portal_map.grid = grid
    portal_map.pose = pose_at(0.0, -1.5, (0.0, 1.0))
    return portal_map


def check_portal_geometry():
    assert PortalConfig().max_width_m == 2.5
    assert PortalConfig().edge_max_width_m == 2.5
    portal_map = _door_map()
    door = _opening(1, [-0.45, 0.0], [0.45, 0.0])
    portal_map.tracks = [door]
    portal_map._refresh_states()
    assert door.reject == "", door.reject
    assert door.state == "candidate"

    portal_map._check_traversal(np.array([0.0, -1.0]), np.array([0.0, 1.0]))
    portal_map._refresh_states()
    assert door.traversed and door.state == "traversed"

    floating = _opening(2, [-0.4, -1.2], [0.4, -1.2])
    portal_map.tracks.append(floating)
    portal_map._refresh_states()
    assert floating.reject == "jamb"
    assert floating.state == "invalid"

    blocked = _opening(3, [-0.45, 0.0], [0.45, 0.0])
    _fill(portal_map.grid, -0.3, 0.25, 0.3, 0.7, "occ")
    portal_map.tracks.append(blocked)
    portal_map._refresh_states()
    assert blocked.reject == "side", blocked.reject

    wide = _opening(4, [-1.4, -1.2], [1.4, -1.2])
    portal_map.tracks.append(wide)
    portal_map._check_traversal(np.array([0.0, -1.5]), np.array([0.0, -0.8]))
    portal_map._refresh_states()
    assert wide.reject == "wide"
    assert not wide.traversed and wide.state == "invalid"

    corridor = PortalMap(portal_map.config)
    corridor.grid = OccupancyGrid(np.zeros(2), portal_map.config)
    _fill(corridor.grid, -0.45, -2.0, 0.45, 2.0, "free")
    _fill(corridor.grid, -0.75, -2.0, -0.45, 2.0, "occ")
    _fill(corridor.grid, 0.45, -2.0, 0.75, 2.0, "occ")
    corridor.pose = pose_at(0.0, -1.2, (0.0, 1.0))
    hall = _opening(5, [-0.40, 0.0], [0.40, 0.0])
    corridor.tracks = [hall]
    corridor._refresh_states()
    assert hall.reject == "narrow", hall.reject
    assert hall.state == "invalid"


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
        assert written[0].name == "vis_1.mp4"
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
    check_occupancy_keeps_furniture()
    check_cleanup_keeps_legs()
    check_portal_geometry()
    check_vis_video()
    print("geonav self_check ok")


if __name__ == "__main__":
    main()
