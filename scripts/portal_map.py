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
    partial_max_m: float = 2.0
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

        Each column contributes at most one obstacle: the nearest point in the
        body of the image, and only when it lies inside the free-ray cap. A hit
        counts when this frame lands within ``hit_match_cells`` of the previous
        frame, so a few centimeters of jitter still builds the wall. One free
        ray is enough to mark a cell walkable.
        """
        c, pc = self.config, portal_config
        height, width = depth.shape
        s = max(1, c.pixel_stride)
        fx = focal_px(width, pc.hfov)
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        vs, us = np.mgrid[0:height:s, 0:width:s]
        z = depth[::s, ::s]
        valid = np.isfinite(z) & (z > pc.depth_min)
        hit = valid & (z < pc.depth_max - 0.05)
        x = (us - cx) * z / fx
        h = pc.camera_height - (vs - cy) * z / fx
        # Drop the top band (ceiling, pitched roof). Keep the lower image so nearby obstacles stay.
        body_rows = vs >= int(height * pc.body_top)
        obstacle = hit & body_rows & (h > c.obstacle_min_h) & (h < c.obstacle_max_h)
        floor = hit & (h <= c.obstacle_min_h)

        z_obs = np.where(obstacle, z, np.inf)
        cols = np.arange(z_obs.shape[1])
        nearest_row = np.argmin(z_obs, axis=0)
        nearest = z_obs[nearest_row, cols]
        keep = np.isfinite(nearest) & (nearest <= c.free_cap_m + self.res)
        near_pts = np.stack(
            [x[nearest_row[keep], cols[keep]], nearest[keep]], axis=1
        ) if keep.any() else np.zeros((0, 2), dtype=np.float64)

        floor_far = np.where(floor, z, 0.0).max(axis=0)
        open_ray = (valid & ~hit).any(axis=0)
        reach = np.where(
            np.isfinite(nearest),
            nearest - self.res,
            np.where(open_ray, c.free_cap_m, floor_far),
        )
        reach = np.clip(reach, 0.0, c.free_cap_m)

        t = np.arange(0.0, c.free_cap_m, self.res)
        slope = (us[0] - cx) / fx
        inside = t[:, None] < reach[None, :]
        ray_z = np.broadcast_to(t[:, None], inside.shape)[inside]
        ray_x = (t[:, None] * slope[None, :])[inside]

        occ = self._flat(camera_xz_to_world(near_pts, pose)) if len(near_pts) else np.zeros(0, dtype=np.int64)
        free = self._flat(camera_xz_to_world(np.stack([ray_x, ray_z], axis=1), pose)) \
            if ray_z.size else np.zeros(0, dtype=np.int64)
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
        hit_now = np.zeros(self.logodds.shape, dtype=bool)
        hit_now.reshape(-1)[occ] = True
        repeated = _dilate(self._prev_hit, c.hit_match_cells) if c.hit_match_cells else self._prev_hit
        lo[hit_now.reshape(-1) & repeated.reshape(-1)] += c.hit_logodds
        self._prev_hit = hit_now
        lo[foot] = np.minimum(lo[foot], c.free_logodds - c.miss_logodds)
        np.clip(lo, -c.clamp_logodds, c.clamp_logodds, out=lo)
        self._cleanup()

    def _cleanup(self):
        """Drop obstacle islands with no free corridor, then fill speckled unknown cells."""
        c = self.config
        lo = self.logodds
        free = lo < c.free_logodds
        occ = lo > c.occupied_logodds
        if occ.any() and free.any():
            seed = occ & _dilate(free, c.wall_gap_cells)
            keep = seed.copy()
            allowed = occ
            for _ in range(6):
                grown = _dilate(keep, 1) & allowed
                if np.array_equal(grown, keep):
                    break
                keep = grown
            lo[occ & ~keep] = 0.0
            occ = lo > c.occupied_logodds
            speck = occ & (_neighbor_count(occ) <= 1) & _dilate(free, 1)
            lo[speck] = 0.0
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
        if self.trajectory:
            self._check_traversal(self.trajectory[-1], agent_xz)
        self.trajectory.append(agent_xz)
        self.pose = pose
        self.step = step
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
        pad = self.config.traverse_pad_m
        for tr in self.tracks:
            if tr.traversed:
                continue
            d = _unit(tr.b - tr.a) * pad
            if _segments_cross(prev_xz, cur_xz, tr.a - d, tr.b + d):
                tr.traversed = True

    def _refresh_states(self):
        c = self.config
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
        if tr.blocked or score <= c.invalid_score:
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

    def render(self, egocentric=True, view_m=None, px=None):
        """Top-down RGB view. Egocentric keeps the agent at the center facing up."""
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
        codes = self.grid.codes(world).reshape(px, px)
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
