"""Fuse per-frame openings into a world map of persistent, stateful portals.

Two layers share one world ground frame, Habitat (x, z) in meters:

- ``OccupancyGrid``: log-odds free / occupied / unknown from depth ray casting.
- ``PortalTrack``: one doorway seen over many frames, with an id that stays
  fixed for the whole episode and a state the decision layer can read.

States, highest priority first: ``abandoned`` (set by the caller through
``PortalMap.abandon``), ``traversed``, ``invalid``, ``explored``,
``confirmed``, ``candidate``. Only ``confirmed`` and ``candidate`` portals
still lead somewhere unseen.
"""

import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_portals import PortalConfig, camera_xz_to_world, detect, focal_px  # noqa: E402

FREE = (255, 255, 255)
OCCUPIED = (40, 40, 40)
UNKNOWN = (160, 160, 160)
TRAJECTORY = (0, 150, 255)
AGENT = (230, 0, 120)
STATE_COLORS = {
    "candidate": (255, 140, 0),
    "confirmed": (0, 190, 0),
    "explored": (60, 120, 255),
    "traversed": (170, 80, 220),
    "invalid": (220, 40, 40),
    "abandoned": (90, 90, 90),
}
# Invalid openings are rejected; explored space is already on the occupancy grid.
UNDRAWN = ("invalid", "explored")
ACTIVE = ("candidate", "confirmed")
# Rejections that mean the span is not a door at all, so crossing it is not a traversal.
NOT_A_DOOR = ("wide", "oblique")


@dataclass
class MapConfig:
    resolution: float = 0.05
    size_m: float = 40.0
    pixel_stride: int = 4
    obstacle_min_h: float = 0.15
    obstacle_max_h: float = 1.6
    free_cap_m: float = 5.0
    agent_radius_m: float = 0.2
    hit_logodds: float = 0.85
    miss_logodds: float = 0.55
    clamp_logodds: float = 4.0
    occupied_logodds: float = 0.6
    free_logodds: float = -0.4
    hole_cells: int = 3
    wall_gap_cells: int = 3
    hit_match_cells: int = 2
    assoc_dist_m: float = 0.5
    assoc_angle_deg: float = 35.0
    close_dist_m: float = 0.45
    close_angle_deg: float = 70.0
    merge_dist_m: float = 0.70
    merge_endpoint_m: float = 0.35
    merge_overlap_m: float = 0.15
    partial_max_m: float = 2.5
    max_portal_m: float = 2.5
    jamb_reach_m: float = 0.30
    jamb_align_deg: float = 40.0
    # Cone around the segment when deciding whether an end actually runs into a wall.
    # Wide enough for an oblique jamb, narrow enough that a wall beside the opening does not count.
    jamb_cone_deg: float = 25.0
    side_offset_m: float = 0.40
    narrow_reach_m: float = 1.0
    narrow_slack_m: float = 0.15
    gap_search_m: float = 2.5
    width_ratio: float = 1.6
    min_segment_m: float = 0.1
    confirm_hits: int = 2
    confirm_score: float = 1.5
    invalid_score: float = -0.5
    miss_penalty: float = 0.5
    miss_min_m: float = 1.5
    miss_max_m: float = 5.0
    miss_fov_fraction: float = 0.8
    occlusion_margin_m: float = 0.4
    behind_radius_m: float = 3.0
    behind_min_m: float = 0.3
    explored_fraction: float = 0.6
    blocked_fraction: float = 0.5
    traverse_pad_m: float = 0.1
    view_m: float = 12.0
    render_px: int = 480


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _perp(v):
    return np.array([-v[1], v[0]], dtype=np.float64)


def _point_segment_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_cross(p1, p2, q1, q2):
    return (
        _cross(q1, q2, p1) * _cross(q1, q2, p2) < 0
        and _cross(p1, p2, q1) * _cross(p1, p2, q2) < 0
    )


def _segment_sep(a0, a1, b0, b1):
    """Minimum distance between two segments. Zero when they cross."""
    if _segments_cross(a0, a1, b0, b1):
        return 0.0
    return min(
        _point_segment_dist(a0, b0, b1),
        _point_segment_dist(a1, b0, b1),
        _point_segment_dist(b0, a0, a1),
        _point_segment_dist(b1, a0, a1),
    )


def _same_opening(a0, a1, b0, b1, sep_m, end_m, min_overlap):
    """True when two doorway segments are the same opening, including oblique repeats.

    Nearly parallel segments that sit side by side without overlapping stay apart,
    so neighboring doors on one wall are not merged.
    """
    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    if _segments_cross(a0, a1, b0, b1):
        return True
    sep = _segment_sep(a0, a1, b0, b1)
    ends = min(
        float(np.linalg.norm(a0 - b0)),
        float(np.linalg.norm(a0 - b1)),
        float(np.linalg.norm(a1 - b0)),
        float(np.linalg.norm(a1 - b1)),
    )
    if sep > sep_m and ends > end_m:
        return False
    da, db = a1 - a0, b1 - b0
    direction = da if float(np.dot(da, da)) >= float(np.dot(db, db)) else db
    direction = _unit(direction)
    pa = sorted((float(np.dot(a0, direction)), float(np.dot(a1, direction))))
    pb = sorted((float(np.dot(b0, direction)), float(np.dot(b1, direction))))
    overlap = min(pa[1], pb[1]) - max(pa[0], pb[0])
    cos = 1.0
    if float(np.dot(da, da)) > 1e-8 and float(np.dot(db, db)) > 1e-8:
        cos = abs(float(np.dot(_unit(da), _unit(db))))
    if overlap < min_overlap and cos > math.cos(math.radians(25)):
        return False
    # A corner's two doors can sit within sep_m without covering the same span.
    if overlap < min_overlap and ends > end_m:
        return False
    return True


def _rotate(v, deg):
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float64)


def _share(codes, value):
    codes = np.asarray(codes)
    if codes.size == 0:
        return 0.0
    return float((codes == value).mean())


def _dilate(mask, radius=1):
    cur = np.asarray(mask, dtype=bool)
    for _ in range(max(1, radius)):
        nxt = cur.copy()
        nxt[1:] |= cur[:-1]
        nxt[:-1] |= cur[1:]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        nxt[1:, 1:] |= cur[:-1, :-1]
        nxt[:-1, :-1] |= cur[1:, 1:]
        nxt[1:, :-1] |= cur[:-1, 1:]
        nxt[:-1, 1:] |= cur[1:, :-1]
        cur = nxt
    return cur


def _drop_islands(occupied, anchor, span):
    """Occupied cells whose component misses ``anchor`` and is larger than ``span``.

    Components that fit in a ``span`` by ``span`` block are kept. A chair leg is
    that size; a floating patch that was never confirmed is not.
    """
    if not occupied.any():
        return np.zeros(occupied.shape, dtype=bool)
    labels, count = ndimage.label(occupied, structure=np.ones((3, 3), dtype=np.int32))
    if count == 0:
        return np.zeros(occupied.shape, dtype=bool)
    keep = np.zeros(count + 1, dtype=bool)
    if anchor.any():
        keep[np.unique(labels[anchor])] = True
    keep[0] = False
    for index, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None or keep[index]:
            continue
        rows, cols = slc
        if rows.stop - rows.start <= span and cols.stop - cols.start <= span:
            keep[index] = True
    return ~keep[labels] & occupied


def _neighbor_count(mask):
    m = np.asarray(mask, dtype=np.uint8)
    count = np.zeros(m.shape, dtype=np.uint8)
    count[1:] += m[:-1]
    count[:-1] += m[1:]
    count[:, 1:] += m[:, :-1]
    count[:, :-1] += m[:, 1:]
    count[1:, 1:] += m[:-1, :-1]
    count[:-1, :-1] += m[1:, 1:]
    count[1:, :-1] += m[:-1, 1:]
    count[:-1, 1:] += m[1:, :-1]
    return count


def _view_free_indices(grid, pose, slope, end, obstacle_idx):
    """Cells on the ground the camera can see, as flat log-odds indices.

    The wedge starts at the camera and ends at the first surface in each
    column, so a room that is only looked at becomes explored. An occupied
    cell, including an obstacle found in this frame, cuts the wedge off.
    """
    slope = np.asarray(slope, dtype=np.float64)
    end = np.clip(np.asarray(end, dtype=np.float64), 0.0, None)
    empty = np.zeros(0, dtype=np.int64)
    if slope.size == 0 or not np.any(end > grid.res):
        return empty
    step = max(grid.res * 0.5, 1e-3)
    t = np.arange(step, float(end.max()) + step, step)
    forward = t[:, None]
    visible = forward < end[None, :]
    ray_x = forward * slope[None, :]
    pts = np.stack([ray_x.ravel(), np.broadcast_to(forward, ray_x.shape).ravel()], axis=1)
    world = camera_xz_to_world(pts, pose)
    rows, cols, ok = grid.cells(world)
    n = grid.logodds.shape[0]
    blocked_flat = (grid.logodds > grid.config.occupied_logodds).reshape(-1).copy()
    if len(obstacle_idx):
        blocked_flat[np.asarray(obstacle_idx, dtype=np.int64)] = True
    blocked = np.zeros(pts.shape[0], dtype=bool)
    on_grid = np.flatnonzero(ok)
    blocked[on_grid] = blocked_flat[rows[ok] * n + cols[ok]]
    blocked = blocked.reshape(ray_x.shape) & visible
    clear = visible & ~np.maximum.accumulate(blocked, axis=0)
    has = clear.any(axis=0)
    last = np.argmax(np.where(clear, forward, -1.0), axis=0)
    reach = np.where(has, t[last], 0.0)
    agent = np.asarray(pose, dtype=np.float64)[[0, 2], 3]
    tips = camera_xz_to_world(np.stack([slope * reach, reach], axis=1), pose)
    poly = np.vstack([agent, tips])
    px = (poly[:, 0] - grid.origin[0]) / grid.res
    py = (poly[:, 1] - grid.origin[1]) / grid.res
    canvas = Image.new("L", (n, n), 0)
    ImageDraw.Draw(canvas).polygon(list(zip(px.tolist(), py.tolist())), fill=1)
    ys, xs = np.nonzero(np.asarray(canvas, dtype=bool))
    if ys.size == 0:
        return empty
    flat = ys * n + xs
    keep = ~blocked_flat[flat]
    return np.unique(flat[keep])


def _agent_axes(pose):
    """World (x, z) of the agent's forward and right directions."""
    pose = np.asarray(pose, dtype=np.float64)
    return _unit(-pose[[0, 2], 2]), _unit(pose[[0, 2], 0])


def _load_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class OccupancyGrid:
    """Square log-odds grid. ``logodds[row, col]``, col along world x, row along world z."""

    def __init__(self, center_xz, config):
        self.config = config
        self.res = config.resolution
        n = int(round(config.size_m / config.resolution))
        self.logodds = np.zeros((n, n), dtype=np.float32)
        self.origin = np.asarray(center_xz, dtype=np.float64) - config.size_m / 2.0
        self._prev_hit = np.zeros((n, n), dtype=bool)
        self._hit_frames = np.zeros((n, n), dtype=np.uint16)

    def cells(self, points_xz):
        pts = np.atleast_2d(np.asarray(points_xz, dtype=np.float64))
        cols = np.floor((pts[:, 0] - self.origin[0]) / self.res).astype(np.int64)
        rows = np.floor((pts[:, 1] - self.origin[1]) / self.res).astype(np.int64)
        n = self.logodds.shape[0]
        ok = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
        return rows, cols, ok

    def _flat(self, points_xz):
        if len(points_xz) == 0:
            return np.zeros(0, dtype=np.int64)
        rows, cols, ok = self.cells(points_xz)
        return np.unique(rows[ok] * self.logodds.shape[1] + cols[ok])

    def codes(self, points_xz):
        """0 unknown, 1 free, 2 occupied for each point. Points off the grid are unknown."""
        rows, cols, ok = self.cells(points_xz)
        out = np.zeros(len(rows), dtype=np.uint8)
        vals = self.logodds[rows[ok], cols[ok]]
        sub = np.zeros(len(vals), dtype=np.uint8)
        sub[vals < self.config.free_logodds] = 1
        sub[vals > self.config.occupied_logodds] = 2
        out[ok] = sub
        return out

    def integrate(self, depth, pose, portal_config):
        """Ray-cast one depth frame.

        Explored ground is the camera wedge: from the robot out to the first
        surface in each column, not the strip of cells the robot has walked.
        The wedge stops at an occupied cell, so a table that has left the image
        stays occupied and the space behind it stays unseen. Obstacle hits are
        every body point in the height band, and the log-odds rise only when
        that cell repeats across frames.
        """
        c, pc = self.config, portal_config
        height, width = depth.shape
        col_stride = max(1, int(c.pixel_stride))
        fx = focal_px(width, pc.hfov)
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        vs, us = np.mgrid[0:height, 0:width:col_stride]
        z = depth[:, ::col_stride]
        valid = np.isfinite(z) & (z > pc.depth_min)
        hit = valid & (z < pc.depth_max - 0.05)
        x = (us - cx) * z / fx
        h = pc.camera_height - (vs - cy) * z / fx
        # Drop the top band (ceiling, pitched roof). Keep the lower image so nearby obstacles stay.
        body_rows = vs >= int(height * pc.body_top)
        obstacle = hit & body_rows & (h > c.obstacle_min_h) & (h < c.obstacle_max_h)

        usable = obstacle & (z <= c.free_cap_m + self.res)
        near_pts = np.stack([x[usable], z[usable]], axis=1) if usable.any() else np.zeros((0, 2), dtype=np.float64)

        obs_near = np.min(np.where(obstacle, z, np.inf), axis=0)
        far = np.max(np.where(hit, z, 0.0), axis=0)
        has_hit = hit.any(axis=0)
        open_ray = (valid & ~hit).any(axis=0)
        far = np.where(has_hit, far, np.where(open_ray, c.free_cap_m, 0.0))
        has_obs = np.isfinite(obs_near)
        end = np.where(has_obs, obs_near - self.res, far)
        end = np.where(has_hit | open_ray | has_obs, np.clip(end, 0.0, c.free_cap_m), 0.0)
        slope = (us[0] - cx) / fx

        occ = self._flat(camera_xz_to_world(near_pts, pose)) if len(near_pts) else np.zeros(0, dtype=np.int64)
        free = _view_free_indices(self, pose, slope, end, occ)
        free = np.setdiff1d(free, occ, assume_unique=True)

        r = c.agent_radius_m
        g = np.arange(-r, r + 1e-6, self.res)
        gx, gz = np.meshgrid(g, g)
        disk = np.stack([gx.ravel(), gz.ravel()], axis=1)
        disk = disk[np.linalg.norm(disk, axis=1) <= r]
        foot = self._flat(disk + np.asarray(pose)[[0, 2], 3])

        lo = self.logodds.reshape(-1)
        lo[free] -= c.miss_logodds
        if self._prev_hit.shape != self.logodds.shape:
            self._prev_hit = np.zeros(self.logodds.shape, dtype=bool)
            self._hit_frames = np.zeros(self.logodds.shape, dtype=np.uint16)
        hit_now = np.zeros(self.logodds.shape, dtype=bool)
        hit_now.reshape(-1)[occ] = True
        frames = self._hit_frames.reshape(-1)
        frames[occ] = np.minimum(frames[occ] + 1, np.iinfo(frames.dtype).max)
        repeated = _dilate(self._prev_hit, c.hit_match_cells) if c.hit_match_cells else self._prev_hit
        lo[hit_now.reshape(-1) & repeated.reshape(-1)] += c.hit_logodds
        self._prev_hit = hit_now
        lo[foot] = np.minimum(lo[foot], c.free_logodds - c.miss_logodds)
        np.clip(lo, -c.clamp_logodds, c.clamp_logodds, out=lo)
        self._cleanup()

    def _cleanup(self):
        """Drop occupied islands with no free corridor, then fill speckled unknown cells.

        A component that fits in a 2x2 block stays, so chair and table legs are
        not wiped. So does any obstacle confirmed over more than one frame:
        a table in the camera's near blind zone does not touch free space yet.
        A cell with one occupied neighbor, or none, is removed only when a
        single frame produced it.
        """
        c = self.config
        lo = self.logodds
        free = lo < c.free_logodds
        occ = lo > c.occupied_logodds
        if occ.any() and free.any():
            near_free = occ & _dilate(free, c.wall_gap_cells)
            anchored = near_free | (occ & (self._hit_frames >= 2))
            drop = _drop_islands(occ, anchored, span=2)
            lo[drop] = 0.0
            self._hit_frames[drop] = 0
            occ = lo > c.occupied_logodds
            confirmed = self._hit_frames >= 2
            speck = occ & (_neighbor_count(occ) <= 1) & ~confirmed & _dilate(free, 1)
            lo[speck] = 0.0
            self._hit_frames[speck] = 0
        for _ in range(c.hole_cells):
            free = lo < c.free_logodds
            occ = lo > c.occupied_logodds
            known = free | occ
            holes = (~known) & (_neighbor_count(known) >= 6) & (_neighbor_count(free) >= 4)
            if not holes.any():
                break
            lo[holes] = c.free_logodds - 0.05

    def known_bounds(self):
        """World (x, z) min and max of all known cells, or ``None``."""
        known = np.abs(self.logodds) > min(abs(self.config.free_logodds), self.config.occupied_logodds)
        rows, cols = np.nonzero(known)
        if rows.size == 0:
            return None
        lo = self.origin + np.array([cols.min(), rows.min()]) * self.res
        hi = self.origin + np.array([cols.max() + 1, rows.max() + 1]) * self.res
        return lo, hi


@dataclass
class PortalTrack:
    id: int
    a: np.ndarray
    b: np.ndarray
    normal: np.ndarray
    full: bool
    weight: float
    first_step: int
    last_step: int
    hits: int = 1
    misses: int = 0
    traversed: bool = False
    explored: bool = False
    blocked: bool = False
    abandoned: bool = False
    state: str = "candidate"
    best: dict = None
    reject: str = ""

    @property
    def center(self):
        return (self.a + self.b) / 2.0

    @property
    def width(self):
        return float(np.linalg.norm(self.b - self.a))

    def to_dict(self):
        return {
            "id": self.id,
            "state": self.state,
            "a": np.round(self.a, 3).tolist(),
            "b": np.round(self.b, 3).tolist(),
            "center": np.round(self.center, 3).tolist(),
            "normal": np.round(self.normal, 3).tolist(),
            "width_m": round(self.width, 2),
            "full": self.full,
            "hits": self.hits,
            "misses": self.misses,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "traversed": self.traversed,
            "explored": self.explored,
            "blocked": self.blocked,
            "abandoned": self.abandoned,
            "reject": self.reject,
            "best": self.best,
        }


class PortalMap:
    """Online map. Call ``update`` once per frame with depth in meters and the Habitat agent pose."""

    def __init__(self, config=None, portal_config=None):
        self.config = config or MapConfig()
        self.portal_config = portal_config or PortalConfig()
        self.grid = None
        self.tracks = []
        self.trajectory = []
        self.pose = None
        self.step = -1
        self._next_id = 1
        self._alias = {}
        c = self.config
        g = np.arange(-c.behind_radius_m, c.behind_radius_m + 1e-6, c.resolution)
        gx, gz = np.meshgrid(g, g)
        disk = np.stack([gx.ravel(), gz.ravel()], axis=1)
        self._disk = disk[np.linalg.norm(disk, axis=1) <= c.behind_radius_m]

    def track(self, track_id):
        return next((t for t in self.tracks if t.id == track_id), None)

    def abandon(self, track_id):
        tr = self.track(track_id)
        if tr is not None:
            tr.abandoned = True
            tr.state = "abandoned"

    def update(self, depth, pose, step, frame=None):
        """Integrate one frame. Returns ``[(portal, track_id), ...]`` for the detections in this frame."""
        pose = np.asarray(pose, dtype=np.float64)
        depth = np.squeeze(depth).astype(np.float32)
        agent_xz = pose[[0, 2], 3].copy()
        if self.grid is None:
            self.grid = OccupancyGrid(agent_xz, self.config)
        self.grid.integrate(depth, pose, self.portal_config)

        openings, edges, dropped = detect(depth, self.portal_config)
        obs, rejected = [], []
        for portal in openings + edges:
            o = self._observe(portal, depth.shape, pose)
            if o is None:
                continue
            # A border opening wider than a door has one unreliable jamb. Keep it only as a sighting.
            if (not o["full"]) and o["width"] > self.config.partial_max_m:
                rejected.append(o)
            else:
                obs.append(o)
        assigned = self._associate(obs)
        labels, seen = [], set()
        for i, o in enumerate(obs):
            tr = assigned.get(i)
            if tr is None:
                tr = self._spawn(o, step)
            else:
                self._merge(tr, o, step)
            self._update_best(tr, o, step, frame)
            labels.append((o["portal"], tr.id))
            seen.add(tr.id)
        # A gated-out or oversized detection still shows the doorway is there; it must not count as a miss.
        for o in rejected:
            seen.update(tr.id for tr in self.tracks if self._compatible(o, tr) is not None)
        for portal in dropped:
            o = self._observe(portal, depth.shape, pose)
            if o is None:
                continue
            seen.update(tr.id for tr in self.tracks if self._compatible(o, tr) is not None)

        self._alias = {}
        self._merge_tracks()
        labels = [(portal, self._resolve(track_id)) for portal, track_id in labels]
        seen = {self._resolve(track_id) for track_id in seen}
        self._count_misses(depth, pose, seen)
        self.pose = pose
        self.step = step
        self._mark_geometry()
        if self.trajectory:
            self._check_traversal(self.trajectory[-1], agent_xz)
        self.trajectory.append(agent_xz)
        self._refresh_states()
        return labels

    def _observe(self, portal, depth_shape, pose):
        pc = self.portal_config
        width = depth_shape[1]
        seg = portal.segment_xz(depth_shape, pc)
        a, b = camera_xz_to_world(seg, pose)
        if np.linalg.norm(b - a) < self.config.min_segment_m:
            return None
        real = (portal.span[0] > 1, portal.span[1] < width - 2)
        full = all(real)
        center = (a + b) / 2.0
        normal = _unit(_perp(b - a))
        if np.dot(normal, center - pose[[0, 2], 3]) < 0:
            normal = -normal
        jamb = max(float(portal.jamb_m), 1.0)
        cx = (width - 1) / 2.0
        centered = 1.0 - abs((portal.span[0] + portal.span[1]) / 2.0 - cx) / cx
        return {
            "portal": portal,
            "a": a,
            "b": b,
            "center": center,
            "normal": normal,
            "real": real,
            "full": full,
            "width": float(np.linalg.norm(b - a)),
            "weight": (1.0 if full else 0.5) / jamb ** 2,
            "view_score": (1.0 if full else 0.3) * centered / jamb,
        }

    def _compatible(self, o, tr):
        """Association cost in meters, or ``None`` if ``o`` cannot be ``tr``.

        A close or crossing pair may differ by ``close_angle_deg``. Farther pairs
        keep the tighter ``assoc_angle_deg``, so a distant oblique false opening
        does not attach to a real door.
        """
        c = self.config
        if not _same_opening(
            o["a"], o["b"], tr.a, tr.b, c.assoc_dist_m, c.merge_endpoint_m, c.merge_overlap_m
        ):
            return None
        sep = _segment_sep(o["a"], o["b"], tr.a, tr.b)
        cos = abs(float(np.dot(_unit(o["b"] - o["a"]), _unit(tr.b - tr.a))))
        limit = c.close_angle_deg if sep <= c.close_dist_m else c.assoc_angle_deg
        if cos < math.cos(math.radians(limit)):
            return None
        if o["full"] and tr.full and sep > c.close_dist_m:
            w0 = float(np.linalg.norm(o["b"] - o["a"]))
            ratio = max(w0, tr.width) / max(min(w0, tr.width), 1e-3)
            if ratio > c.width_ratio:
                return None
        return sep

    def _associate(self, obs):
        pairs = []
        for i, o in enumerate(obs):
            for tr in self.tracks:
                cost = self._compatible(o, tr)
                if cost is not None:
                    pairs.append((cost, i, tr))
        pairs.sort(key=lambda item: item[0])
        assigned, used = {}, set()
        for _, i, tr in pairs:
            if i in assigned or tr.id in used:
                continue
            assigned[i] = tr
            used.add(tr.id)
        return assigned

    def _spawn(self, o, step):
        tr = PortalTrack(
            id=self._next_id,
            a=o["a"].copy(),
            b=o["b"].copy(),
            normal=o["normal"].copy(),
            full=o["full"],
            weight=o["weight"],
            first_step=step,
            last_step=step,
        )
        self._next_id += 1
        self.tracks.append(tr)
        return tr

    def _merge(self, tr, o, step):
        """Fold one detection into a track. Geometry follows the clearest full view.

        Averaging oblique views rotates the segment and splits one door into
        crossing tracks. A better centered full view replaces the endpoints.
        A border view never moves a full track.
        """
        previous = tr.best["score"] if tr.best else -1.0
        replace = o["full"] and (not tr.full or o["view_score"] > previous)
        if replace:
            tr.a = o["a"].copy()
            tr.b = o["b"].copy()
            tr.normal = o["normal"].copy()
            tr.full = True
            tr.weight = o["weight"]
        elif not tr.full:
            a, b, w = o["a"], o["b"], o["weight"]
            if np.linalg.norm(a - tr.a) + np.linalg.norm(b - tr.b) > np.linalg.norm(a - tr.b) + np.linalg.norm(b - tr.a):
                a, b = b, a
            total = tr.weight + w
            tr.a = (tr.a * tr.weight + a * w) / total
            tr.b = (tr.b * tr.weight + b * w) / total
            tr.weight = total
            normal = _unit(_perp(tr.b - tr.a))
            tr.normal = normal if np.dot(normal, tr.normal) >= 0 else -normal
        tr.hits += 1
        tr.last_step = step

    def _resolve(self, track_id):
        seen = set()
        while track_id in self._alias and track_id not in seen:
            seen.add(track_id)
            track_id = self._alias[track_id]
        return track_id

    def _same_track_opening(self, left, right):
        c = self.config
        return _same_opening(
            left.a, left.b, right.a, right.b, c.merge_dist_m, c.merge_endpoint_m, c.merge_overlap_m
        )

    def _merge_tracks(self):
        """Collapse doorway tracks that already overlap. Ids of the kept track stay put."""
        merged = True
        while merged:
            merged = False
            live = [tr for tr in self.tracks if not tr.abandoned]
            live.sort(key=lambda tr: (tr.full, tr.best["score"] if tr.best else 0.0, tr.hits, -tr.id), reverse=True)
            for i, keep in enumerate(live):
                for drop in live[i + 1:]:
                    if not self._same_track_opening(keep, drop):
                        continue
                    self._absorb(keep, drop)
                    merged = True
                    break
                if merged:
                    break

    def _absorb(self, keep, drop):
        drop_score = drop.best["score"] if drop.best else -1.0
        keep_score = keep.best["score"] if keep.best else -1.0
        if drop.full and (not keep.full or drop_score > keep_score + 0.05):
            keep.a = drop.a.copy()
            keep.b = drop.b.copy()
            keep.normal = drop.normal.copy()
            keep.full = True
            keep.weight = drop.weight
        keep.hits += drop.hits
        keep.misses = min(keep.misses, drop.misses)
        keep.first_step = min(keep.first_step, drop.first_step)
        keep.last_step = max(keep.last_step, drop.last_step)
        keep.traversed = keep.traversed or drop.traversed
        keep.weight = max(keep.weight, drop.weight)
        if drop.best and (keep.best is None or drop.best["score"] > keep.best["score"]):
            keep.best = drop.best
        self._alias[drop.id] = keep.id
        self.tracks = [tr for tr in self.tracks if tr.id != drop.id]

    def _update_best(self, tr, o, step, frame):
        if tr.best is None or o["view_score"] > tr.best["score"]:
            tr.best = {
                "frame": frame,
                "step": step,
                "box": [int(v) for v in o["portal"].box],
                "score": round(float(o["view_score"]), 4),
            }

    def _count_misses(self, depth, pose, seen):
        """A track in clear view at a usable range that was not detected loses confidence."""
        c, pc = self.config, self.portal_config
        height, width = depth.shape
        fx = focal_px(width, pc.hfov)
        cx = (width - 1) / 2.0
        rot, pos = pose[:3, :3], pose[:3, 3]
        margin = (1.0 - c.miss_fov_fraction) * cx
        y0, y1 = int(height * pc.body_top), int(height * pc.body_bottom)
        for tr in self.tracks:
            if tr.id in seen or tr.abandoned or tr.traversed:
                continue
            world = np.array([tr.center[0], pos[1], tr.center[1]])
            local = rot.T @ (world - pos)
            x, z = float(local[0]), float(-local[2])
            if not (c.miss_min_m < z < c.miss_max_m):
                continue
            u = cx + fx * x / z
            if u < margin or u > width - 1 - margin:
                continue
            col = depth[y0:y1, max(0, int(u) - 2):int(u) + 3]
            col = col[np.isfinite(col) & (col > pc.depth_min)]
            if col.size and float(np.median(col)) < z - c.occlusion_margin_m:
                continue
            tr.misses += 1

    def _check_traversal(self, prev_xz, cur_xz):
        """Crossing counts unless the span is too wide or diagonal to be a door.

        ``jamb``, ``side`` and ``narrow`` can be wrong while the map is still
        filling in, so walking through such an opening still counts.
        """
        pad = self.config.traverse_pad_m
        for tr in self.tracks:
            if tr.traversed or tr.reject in NOT_A_DOOR:
                continue
            d = _unit(tr.b - tr.a) * pad
            if _segments_cross(prev_xz, cur_xz, tr.a - d, tr.b + d):
                tr.traversed = True

    def _mark_geometry(self):
        """Width cap and the occupancy checks. Recomputed every frame, not latched.

        A check only rejects on evidence already in the map. While the walls or
        the floor around an opening are still unknown, it stays a portal.
        ``wide`` and ``oblique`` are not doors: not drawn, not a waypoint, and
        not a traversal.
        """
        for tr in self.tracks:
            tr.reject = self._reject_reason(tr)
            if tr.reject == "wide":
                tr.traversed = False

    def _reject_reason(self, tr):
        """Empty when the opening may stay a portal.

        Otherwise ``wide``, ``oblique``, ``jamb``, ``side``, or ``narrow``.
        A side view is straightened onto the jamb-to-jamb line before the other
        checks, so the normal points through the doorway.
        """
        if tr.abandoned:
            return ""
        if tr.width > self.config.max_portal_m:
            return "wide"
        if self.grid is None or self.pose is None:
            return ""
        aligned = self._align_to_jambs(tr)
        if aligned in ("wide", "oblique", "jamb"):
            return aligned
        if self._side_blocked(tr):
            return "side"
        if self._not_narrowest(tr):
            return "narrow"
        return ""

    def _align_to_jambs(self, tr):
        """Turn the segment so it meets both obstacles nearly head-on.

        A side view drops the endpoints on the jamb corners, so the drawn line
        is diagonal and its normal misses the room. Search from the center for
        the direction that strikes both obstacles most squarely, and keep that
        span. Returns ``ok``, ``pending`` while the jambs are still unmapped,
        or a rejection: ``jamb`` when an end sits in mapped open floor or only
        one end runs into a wall, ``oblique`` when the map shows no head-on
        span, ``wide``.
        """
        if self._end_in_open_floor(tr.a) or self._end_in_open_floor(tr.b):
            return "jamb"
        # One end meets a wall and the other already runs out into mapped floor.
        # An unmapped end stays: there is no evidence yet that it lacks a jamb.
        if self._one_sided(tr):
            return "jamb"
        if not self._jambs_touch(tr):
            return "pending"
        span, measured = self._best_head_on_span(tr)
        if span is None:
            return "oblique" if measured else "pending"
        new_a, new_b = span
        if float(np.linalg.norm(new_b - new_a)) > self.config.max_portal_m:
            return "wide"
        normal = _unit(_perp(new_b - new_a))
        if float(np.dot(normal, tr.normal)) < 0.0:
            normal = -normal
        tr.a = new_a
        tr.b = new_b
        tr.normal = normal
        return "ok"

    def _end_in_open_floor(self, endpoint):
        """True when the floor around ``endpoint`` is mapped free with no solid jamb in reach.

        An isolated occupied cell is not a jamb. Unknown cells are not free, so an
        end on the frontier of the map is kept.
        """
        pts = self._disk_points(endpoint, self.config.jamb_reach_m)
        if len(pts) == 0:
            return False
        codes = self.grid.codes(pts)
        if _share(codes, 1) < 0.6:
            return False
        occupied = pts[codes == 2]
        return not any(self._solid_obstacle(point) for point in occupied)

    def _one_sided(self, tr):
        """True when one end runs into a wall and the other into mapped open floor.

        Support is measured along the segment, not anywhere in the disk. A wall
        beside the opening does not hold up an end that continues through free
        cells. Unknown beyond an end is not open floor.
        """
        direction = _unit(tr.b - tr.a)
        if float(np.linalg.norm(direction)) < 1e-9:
            return False
        support_a = self._jamb_support(tr.a, -direction)
        support_b = self._jamb_support(tr.b, direction)
        return (support_a == "wall" and support_b == "open") or (support_b == "wall" and support_a == "open")

    def _jamb_support(self, endpoint, outward):
        """``wall``, ``open``, or ``unknown`` just outside ``endpoint`` along ``outward``."""
        if self._solid_obstacle(endpoint):
            return "wall"
        outward = _unit(outward)
        if float(np.linalg.norm(outward)) < 1e-9:
            return "unknown"
        reach = self.config.jamb_reach_m
        cone = self.config.jamb_cone_deg
        step = 5 if cone >= 5 else 1
        for deg in range(-int(cone), int(cone) + 1, step):
            if self._ray_kind(endpoint, _rotate(outward, deg), reach) == "wall":
                return "wall"
        return self._ray_kind(endpoint, outward, reach)

    def _ray_kind(self, origin, direction, reach):
        """``wall`` when a solid obstacle is within ``reach``, else ``open`` or ``unknown``."""
        direction = _unit(direction)
        dist = self._gap_ray(origin, direction, reach)
        if dist is None:
            return "unknown"
        if self._solid_obstacle(origin + direction * dist):
            return "wall"
        return "open"

    def _best_head_on_span(self, tr):
        """Shortest gap through the center that hits both walls head-on.

        Returns ``(span, measured)``. ``measured`` is true when at least one
        direction was mapped on both sides, so a missing span is evidence and
        not just unknown floor.
        """
        center = tr.center
        if self._code_at(center) != 1:
            return None, False
        base = _unit(tr.b - tr.a)
        if float(np.linalg.norm(base)) < 1e-9:
            return None, False
        limit = math.cos(math.radians(self.config.jamb_align_deg))
        inset = max(self.grid.res, 1e-3)
        best = None
        measured = False
        for deg in range(-80, 81, 5):
            direction = _rotate(base, deg)
            pos = self._wall_hit(center, direction)
            neg = self._wall_hit(center, -direction)
            if pos is None or neg is None:
                continue
            measured = True
            if pos[1] is None or neg[1] is None:
                continue
            dist_pos, out_pos = pos
            dist_neg, out_neg = neg
            width = dist_pos + dist_neg
            if width > self.config.max_portal_m or width < self.config.min_segment_m:
                continue
            if dist_pos <= inset or dist_neg <= inset:
                continue
            onto_pos = float(np.dot(out_pos, -direction))
            onto_neg = float(np.dot(out_neg, direction))
            if onto_pos < limit or onto_neg < limit:
                continue
            score = min(onto_pos, onto_neg)
            if best is not None and (score < best[0] - 1e-6 or (abs(score - best[0]) <= 1e-6 and width >= best[1])):
                continue
            new_a = center - direction * (dist_neg - inset)
            new_b = center + direction * (dist_pos - inset)
            if self._code_at((new_a + new_b) / 2.0) == 2:
                continue
            best = (score, width, new_a, new_b)
        if best is None:
            return None, measured
        return (best[2], best[3]), measured

    def _wall_hit(self, origin, direction):
        """``(distance, outward)`` at a solid obstacle, ``(distance, None)`` when the
        ray ran through mapped open floor, or ``None`` when it reached unknown."""
        dist = self._gap_ray(origin, direction)
        if dist is None:
            return None
        point = origin + direction * dist
        if not self._solid_obstacle(point):
            return dist, None
        outward, _contact = self._jamb_outward(point)
        return dist, outward

    def _jamb_outward(self, endpoint):
        """Unit direction from the nearby obstacle mass toward ``endpoint``, and the nearest occupied point."""
        pts = self._occupied_points(endpoint, self.config.jamb_reach_m)
        if len(pts) == 0:
            return None, None
        endpoint = np.asarray(endpoint, dtype=np.float64)
        nearest = pts[np.argmin(np.linalg.norm(pts - endpoint, axis=1))]
        outward = endpoint - pts.mean(axis=0)
        if float(np.linalg.norm(outward)) < 1e-6:
            return None, None
        return _unit(outward), nearest

    def _disk_points(self, center, radius):
        center = np.asarray(center, dtype=np.float64)
        res = self.grid.res
        span = int(math.ceil(radius / max(res, 1e-6)))
        offsets = np.arange(-span, span + 1) * res
        gx, gz = np.meshgrid(offsets, offsets)
        delta = np.stack([gx.ravel(), gz.ravel()], axis=1)
        delta = delta[np.linalg.norm(delta, axis=1) <= radius + 1e-6]
        return delta + center

    def _occupied_points(self, center, radius):
        pts = self._disk_points(center, radius)
        if len(pts) == 0:
            return np.zeros((0, 2))
        return pts[self.grid.codes(pts) == 2]

    def _jambs_touch(self, tr):
        reach = self.config.jamb_reach_m
        return self._occupied_near(tr.a, reach) and self._occupied_near(tr.b, reach)

    def _occupied_near(self, center, radius):
        pts = self._disk_points(center, radius)
        if len(pts) == 0:
            return False
        return bool((self.grid.codes(pts) == 2).any())

    def _side_blocked(self, tr):
        """True when either side of the opening is already mostly obstacle. Unknown floor does not count."""
        agent = self.pose[[0, 2], 3]
        normal = _unit(tr.normal)
        near = normal if float(np.dot(normal, agent - tr.center)) >= 0.0 else -normal
        near_codes = self.grid.codes(self._side_points(tr, near))
        far_codes = self.grid.codes(self._side_points(tr, -near))
        return _share(near_codes, 2) >= 0.4 or _share(far_codes, 2) >= 0.4

    def _side_points(self, tr, direction):
        ts = np.linspace(0.3, 0.7, 5)
        base = tr.a[None, :] + ts[:, None] * (tr.b - tr.a)[None, :]
        return base + np.asarray(direction, dtype=np.float64)[None, :] * self.config.side_offset_m

    def _not_narrowest(self, tr):
        """True when the map shows the opening is not a pinch point.

        Two kinds of evidence: the passage one meter away is clearly narrower
        than here, or both sides are measured and neither opens up, as in a line
        across the middle of a corridor. Unmeasured sides never reject. Only the
        one-meter sample can show a narrower neighbor, so a chair just inside the
        doorway does not erase the door.
        """
        direction = _unit(tr.b - tr.a)
        normal = _unit(tr.normal)
        here = self._passage_gap(tr.center, direction)
        if here is None:
            return False
        slack = self.config.narrow_slack_m
        wider = False
        measured_sides = 0
        for sign in (1.0, -1.0):
            mid = self._passage_gap(tr.center + normal * sign * 0.5, direction)
            far = self._passage_gap(tr.center + normal * sign * min(1.0, self.config.narrow_reach_m), direction)
            if mid is not None or far is not None:
                measured_sides += 1
            if (mid is not None and mid >= here + slack) or (far is not None and far >= here + slack):
                wider = True
            if far is not None and far < here - slack:
                return True
        return measured_sides == 2 and not wider

    def _passage_gap(self, origin, direction):
        """Free width through ``origin`` along ``direction``, or ``None`` when a side is unknown.

        A ray that stays free out to ``gap_search_m`` counts as that wide. Isolated
        unknown or occupied cells do not close it; a run of either does.
        """
        origin = np.asarray(origin, dtype=np.float64)
        if self._code_at(origin) != 1:
            return None
        left = self._gap_ray(origin, direction)
        right = self._gap_ray(origin, -direction)
        if left is None or right is None:
            return None
        return left + right

    def _gap_ray(self, origin, direction, limit=None):
        step = self.grid.res
        limit = self.config.gap_search_m if limit is None else limit
        dist = step
        while dist <= limit + 1e-6:
            point = origin + direction * dist
            code = self._code_at(point)
            if code == 0:
                # One unknown cell is a speck. A run of them is still unmapped.
                if self._code_at(origin + direction * (dist + step)) == 0:
                    return None
            elif code == 2 and self._solid_obstacle(point):
                return dist
            dist += step
        return limit

    def _code_at(self, point):
        return int(self.grid.codes([point])[0])

    def _solid_obstacle(self, point):
        """True when the hit is more than an isolated cell."""
        rows, cols, ok = self.grid.cells([point])
        if not ok[0]:
            return False
        row, col = int(rows[0]), int(cols[0])
        lo = self.grid.logodds
        patch = lo[max(0, row - 1):row + 2, max(0, col - 1):col + 2]
        return int((patch > self.grid.config.occupied_logodds).sum()) >= 3

    def _refresh_states(self):
        c = self.config
        self._mark_geometry()
        for tr in self.tracks:
            if not tr.traversed and not tr.abandoned:
                behind = self._disk[self._disk @ tr.normal >= c.behind_min_m] + tr.center
                codes = self.grid.codes(behind)
                tr.explored = bool(codes.size) and float((codes > 0).mean()) >= c.explored_fraction
                mid = tr.a[None, :] + np.linspace(0.3, 0.7, 5)[:, None] * (tr.b - tr.a)[None, :]
                tr.blocked = float((self.grid.codes(mid) == 2).mean()) > c.blocked_fraction
            tr.state = self._state(tr)

    def _state(self, tr):
        c = self.config
        score = tr.hits - c.miss_penalty * tr.misses
        if tr.abandoned:
            return "abandoned"
        if tr.traversed:
            return "traversed"
        if tr.reject or tr.blocked or score <= c.invalid_score:
            return "invalid"
        # One jamb is not enough to offer the doorway as a goal.
        if not tr.full:
            return "candidate"
        if tr.explored:
            return "explored"
        if tr.hits >= c.confirm_hits and score >= c.confirm_score:
            return "confirmed"
        return "candidate"

    def _hidden_ids(self):
        """Lower-quality tracks that still overlap a better one. Merge should have removed these."""
        ranked = sorted(
            (tr for tr in self.tracks if tr.state not in ("invalid", "abandoned")),
            key=lambda tr: (tr.full, tr.best["score"] if tr.best else 0.0, tr.hits, -tr.id),
            reverse=True,
        )
        kept, hidden = [], set()
        for tr in ranked:
            if any(self._same_track_opening(tr, other) for other in kept):
                hidden.add(tr.id)
            else:
                kept.append(tr)
        return hidden

    def describe(self, states=ACTIVE):
        """One line per portal, nearest first, relative to the current agent heading."""
        if self.pose is None:
            return []
        agent = self.pose[[0, 2], 3]
        fwd, right = _agent_axes(self.pose)
        hidden = self._hidden_ids()
        rows = []
        for tr in self.tracks:
            if tr.state not in states or tr.id in hidden:
                continue
            d = tr.center - agent
            bearing = math.degrees(math.atan2(float(d @ right), float(d @ fwd)))
            if abs(bearing) < 15:
                side = "ahead"
            elif abs(bearing) > 165:
                side = "behind"
            else:
                side = f"{abs(bearing):.0f} deg {'right' if bearing > 0 else 'left'}"
            rows.append((
                float(np.linalg.norm(d)),
                f"{tr.id}: {side}, {np.linalg.norm(d):.1f} m, width {tr.width:.1f} m, "
                f"{tr.state}, seen {tr.hits}x",
            ))
        return [line for _, line in sorted(rows, key=lambda item: item[0])]

    def render(self, egocentric=True, view_m=None, px=None, overlay=None, codes_of=None):
        """Top-down RGB view. Egocentric keeps the agent at the center facing up.

        ``overlay(draw, to_px)`` can add marks in the same pixel frame. ``to_px``
        maps a world ``(x, z)`` point to ``(column, row)``.
        """
        c = self.config
        px = px or c.render_px
        agent = self.pose[[0, 2], 3]
        if egocentric:
            fwd, right = _agent_axes(self.pose)
            center = agent
            view_m = view_m or c.view_m
        else:
            fwd, right = np.array([0.0, -1.0]), np.array([1.0, 0.0])
            bounds = self.grid.known_bounds()
            if bounds is None:
                center, extent = agent, c.view_m
            else:
                lo, hi = bounds
                center = (lo + hi) / 2.0
                extent = float(max(hi - lo)) + 2.0
            view_m = view_m or max(extent, 4.0)
        scale = view_m / px

        grid = (np.arange(px) + 0.5 - px / 2.0) * scale
        right_off, up_off = np.meshgrid(grid, -grid)
        world = center + right_off.reshape(-1, 1) * right + up_off.reshape(-1, 1) * fwd
        classify = self.grid.codes if codes_of is None else codes_of
        codes = classify(world).reshape(px, px)
        canvas = np.empty((px, px, 3), dtype=np.uint8)
        canvas[:] = UNKNOWN
        canvas[codes == 1] = FREE
        canvas[codes == 2] = OCCUPIED
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)

        def to_px(p):
            d = np.asarray(p) - center
            return (float(d @ right) / scale + px / 2.0, px / 2.0 - float(d @ fwd) / scale)

        if len(self.trajectory) > 1:
            draw.line([to_px(p) for p in self.trajectory], fill=TRAJECTORY, width=2)
        font = _load_font(16)
        hidden = self._hidden_ids()
        for tr in self.tracks:
            if tr.id in hidden or tr.state in UNDRAWN:
                continue
            color = STATE_COLORS[tr.state]
            draw.line([to_px(tr.a), to_px(tr.b)], fill=color, width=5)
            tip = to_px(tr.center + tr.normal * 0.4)
            draw.line([to_px(tr.center), tip], fill=color, width=2)
            lx, ly = to_px(tr.center - tr.normal * 0.35)
            draw.text((lx - 6, ly - 8), str(tr.id), fill=color, font=font)

        ax, ay = to_px(agent)
        heading = _agent_axes(self.pose)[0]
        hx, hy = float(heading @ right), -float(heading @ fwd)
        size = max(6.0, 0.35 / scale)
        tip = (ax + hx * size, ay + hy * size)
        left = (ax - hx * size * 0.5 + hy * size * 0.5, ay - hy * size * 0.5 - hx * size * 0.5)
        right_pt = (ax - hx * size * 0.5 - hy * size * 0.5, ay - hy * size * 0.5 + hx * size * 0.5)
        draw.polygon([tip, left, right_pt], fill=AGENT)
        if overlay is not None:
            overlay(draw, to_px)
        return image

    def to_dict(self):
        return {
            "config": asdict(self.config),
            "portal_config": asdict(self.portal_config),
            "step": self.step,
            "trajectory": np.round(np.array(self.trajectory), 3).tolist(),
            "tracks": [tr.to_dict() for tr in self.tracks],
        }


def draw_labels(image, labels, portal_map):
    """Draw this frame's detections on the RGB, numbered with their track id and colored by state."""
    vis = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(vis)
    font = _load_font(20)
    for portal, track_id in labels:
        tr = portal_map.track(track_id)
        color = STATE_COLORS[tr.state] if tr else (255, 255, 0)
        draw.rectangle(portal.box, outline=color, width=3)
        draw.text((portal.box[0] + 5, portal.box[1] + 4), str(track_id), fill=color, font=font)
    return vis
