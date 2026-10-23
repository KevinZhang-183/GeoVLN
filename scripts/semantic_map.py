"""Project SAM3 masks onto the ground and fuse them into object instances.

This map is separate from the portal map. Both use Habitat ground coordinates
(x, z) in meters. Fusing the two is a later step that reads both JSON files.

Camera ground frame matches extract_portals: x right, z forward. Habitat's
agent frame points backward along z, and camera_xz_to_world flips that.
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageDraw

from extract_portals import PortalConfig, camera_xz_to_world, focal_px
from parse_instructions import ARTICLES, ROOM_WORDS, _landmark
from portal_map import (  # noqa: E402
    AGENT,
    FREE,
    OCCUPIED,
    TRAJECTORY,
    UNKNOWN,
    MapConfig,
    OccupancyGrid,
    _agent_axes,
    _load_font,
)

# Prompt order, shared with the RGB overlays in build_semantic_map.py.
LABEL_COLORS = (
    (0, 180, 70),
    (220, 120, 0),
    (150, 40, 180),
    (0, 110, 210),
    (190, 40, 40),
    (0, 150, 150),
)


class SemanticConfig:
    def __init__(
        self,
        hfov=79.0,
        depth_min=0.4,
        depth_max=6.0,
        depth_band_m=0.35,
        border_px=8,
        erode_px=2,
        min_pixels=40,
        pixel_stride=2,
        min_score=0.45,
        assoc_dist_m=0.8,
        confirm_hits=2,
    ):
        self.hfov = hfov
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_band_m = depth_band_m
        self.border_px = border_px
        self.erode_px = erode_px
        self.min_pixels = min_pixels
        self.pixel_stride = pixel_stride
        self.min_score = min_score
        self.assoc_dist_m = assoc_dist_m
        self.confirm_hits = confirm_hits


class Instance:
    def __init__(self, index, label, center, radius, score, frame):
        self.id = index
        self.label = label
        self.x = float(center[0])
        self.z = float(center[1])
        self.radius_m = float(radius)
        self.hits = 1
        self.score_sum = float(score)
        self.best_score = float(score)
        self.best_frame = frame

    def add(self, center, radius, score, frame):
        n = self.hits
        self.x = (self.x * n + float(center[0])) / (n + 1)
        self.z = (self.z * n + float(center[1])) / (n + 1)
        self.radius_m = (self.radius_m * n + float(radius)) / (n + 1)
        self.hits = n + 1
        self.score_sum += float(score)
        if score >= self.best_score:
            self.best_score = float(score)
            self.best_frame = frame

    def to_dict(self, confirm_hits):
        return {
            "id": self.id,
            "label": self.label,
            "state": "confirmed" if self.hits >= confirm_hits else "candidate",
            "xz": [round(self.x, 3), round(self.z, 3)],
            "radius_m": round(self.radius_m, 3),
            "hits": self.hits,
            "mean_score": round(self.score_sum / self.hits, 3),
            "best_score": round(self.best_score, 3),
            "best_frame": self.best_frame,
        }


def _erode(mask, radius):
    if radius <= 0:
        return mask
    blocked = ~mask
    grown = blocked.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx > radius * radius:
                continue
            grown |= np.roll(np.roll(blocked, dy, axis=0), dx, axis=1)
    grown[:radius, :] = True
    grown[-radius:, :] = True
    grown[:, :radius] = True
    grown[:, -radius:] = True
    return ~grown


def mask_points(mask, depth, config):
    """Camera-ground points inside a mask, shape (N, 2), or an empty array.

    The mask is eroded and only pixels near the median depth are kept, so a
    chair does not drag a tail onto the wall behind it.
    """
    mask = np.asarray(mask, dtype=bool)
    depth = np.squeeze(depth).astype(np.float32)
    if mask.shape != depth.shape or mask.sum() < config.min_pixels:
        return np.zeros((0, 2), dtype=np.float32)
    margin = config.border_px
    if margin:
        mask = mask.copy()
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False
    mask = _erode(mask, config.erode_px)
    ys, xs = np.nonzero(mask)
    if ys.size < config.min_pixels:
        return np.zeros((0, 2), dtype=np.float32)
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > config.depth_min) & (z <= config.depth_max)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    if z.size < config.min_pixels:
        return np.zeros((0, 2), dtype=np.float32)
    band = np.abs(z - np.median(z)) <= config.depth_band_m
    ys, xs, z = ys[band], xs[band], z[band]
    if z.size < config.min_pixels:
        return np.zeros((0, 2), dtype=np.float32)
    stride = max(1, config.pixel_stride)
    ys, xs, z = ys[::stride], xs[::stride], z[::stride]
    width = depth.shape[1]
    fx = focal_px(width, config.hfov)
    cx = (width - 1) / 2.0
    x = (xs - cx) * z / fx
    return np.stack([x, z], axis=1).astype(np.float32)


def _room_only(text):
    words = [word for word in text.split() if word not in ARTICLES | {"of"}]
    return bool(words) and all(word in ROOM_WORDS for word in words)


def prompts_from_parsed(parsed):
    """Object phrases from one episode, in instruction order. Room names are left out."""
    found = []
    for step in parsed.get("subtasks") or []:
        for key in ("landmark", "near"):
            text = _landmark(step.get(key, ""))
            if not text or _room_only(text) or text in found:
                continue
            found.append(text)
    return found


def pose_separated(previous, current, move_m, turn_deg):
    """True when the agent has walked ``move_m`` or turned ``turn_deg`` since ``previous``."""
    moved = float(np.linalg.norm(np.asarray(current)[[0, 2], 3] - np.asarray(previous)[[0, 2], 3]))
    if moved >= move_m:
        return True
    earlier, later = _agent_axes(previous)[0], _agent_axes(current)[0]
    turn = float(np.degrees(np.arccos(np.clip(float(np.dot(earlier, later)), -1.0, 1.0))))
    return turn >= turn_deg


def select_keyframes(poses, stride=None, move_m=0.5, turn_deg=30.0):
    """Frame indexes to segment. ``stride`` forces a fixed step; otherwise sample by pose.

    The first posed frame and the last posed frame are always included.
    """
    count = len(poses)
    if count == 0:
        return []
    if stride is not None:
        chosen = list(range(0, count, max(1, int(stride))))
    else:
        chosen = []
        previous = None
        for index, pose in enumerate(poses):
            if pose is None:
                continue
            if previous is None or pose_separated(previous, pose, move_m, turn_deg):
                chosen.append(index)
                previous = pose
    last = next((index for index in range(count - 1, -1, -1) if poses[index] is not None), None)
    if last is not None and last not in chosen:
        chosen.append(last)
    return chosen


def _smooth_path(path, window=5):
    """Light moving average for drawing. Endpoints stay on the real poses."""
    points = np.asarray(path, dtype=np.float64)
    if len(points) < 3 or window < 3:
        return points
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in (0, 1)], axis=1)
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def _text_box(text, font):
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    if hasattr(draw, "textbbox"):
        return draw.textbbox((0, 0), text, font=font)
    width, height = draw.textsize(text, font=font)
    return (0, 0, width, height)


def _draw_centered(draw, center, text, font, fill):
    left, top, right, bottom = _text_box(text, font)
    width, height = right - left, bottom - top
    draw.text((center[0] - width / 2 - left, center[1] - height / 2 - top), text, fill=fill, font=font)


class SemanticMap:
    def __init__(self, config=None, portal_config=None, map_config=None):
        self.config = config or SemanticConfig()
        self.portal_config = portal_config or PortalConfig()
        self.map_config = map_config or MapConfig()
        self.instances = []
        self.path = []
        self.heading = None
        self.grid = None
        self._next_id = 1

    def note_frame(self, pose, depth=None):
        """Record one pose. Depth, when present, updates the occupancy grid."""
        pose = np.asarray(pose, dtype=np.float64)
        self.path.append([float(pose[0, 3]), float(pose[2, 3])])
        self.heading = np.asarray(_agent_axes(pose)[0], dtype=np.float64)
        if depth is None:
            return
        agent_xz = pose[[0, 2], 3].copy()
        if self.grid is None:
            self.grid = OccupancyGrid(agent_xz, self.map_config)
        self.grid.integrate(np.squeeze(depth).astype(np.float32), pose, self.portal_config)

    def note_agent(self, pose):
        self.note_frame(pose)

    def observe(self, label, points_xz, score, frame):
        """Fuse one mask. ``points_xz`` is world (x, z). Returns 'new', 'updated', or 'skip'."""
        points = np.atleast_2d(np.asarray(points_xz, dtype=np.float64))
        if points.shape[0] < 3 or score < self.config.min_score:
            return "skip"
        center = points.mean(axis=0)
        radius = float(np.median(np.linalg.norm(points - center, axis=1)))
        best = None
        best_dist = self.config.assoc_dist_m
        for inst in self.instances:
            if inst.label != label:
                continue
            dist = float(np.hypot(inst.x - center[0], inst.z - center[1]))
            if dist < best_dist:
                best, best_dist = inst, dist
        if best is None:
            self.instances.append(Instance(self._next_id, label, center, radius, score, frame))
            self._next_id += 1
            return "new"
        best.add(center, radius, score, frame)
        return "updated"

    def observe_camera(self, label, points_cam_xz, pose, score, frame):
        if len(points_cam_xz) == 0:
            return "skip"
        return self.observe(label, camera_xz_to_world(points_cam_xz, pose), score, frame)

    def label_indexes(self):
        """Per-label number in first-seen order. Two doorways are 1 and 2, not a global id."""
        counts = {}
        indexes = {}
        for inst in self.instances:
            counts[inst.label] = counts.get(inst.label, 0) + 1
            indexes[inst.id] = counts[inst.label]
        return indexes

    def describe(self):
        indexes = self.label_indexes()
        lines = []
        for inst in self.instances:
            state = "confirmed" if inst.hits >= self.config.confirm_hits else "candidate"
            lines.append(
                f"{indexes[inst.id]}: {inst.label} {state} ({inst.x:.2f}, {inst.z:.2f}) "
                f"hits={inst.hits} mean_score={inst.score_sum / inst.hits:.2f} best={inst.best_frame}"
            )
        return lines

    def to_dict(self, prompts):
        confirm = self.config.confirm_hits
        indexes = self.label_indexes()
        instances = []
        for inst in self.instances:
            row = inst.to_dict(confirm)
            row["index"] = indexes[inst.id]
            instances.append(row)
        counts = {"confirmed": 0, "candidate": 0}
        for inst in instances:
            counts[inst["state"]] += 1
        return {
            "prompts": list(prompts),
            "assoc_dist_m": self.config.assoc_dist_m,
            "confirm_hits": confirm,
            "counts": counts,
            "path_xz": [[round(x, 3), round(z, 3)] for x, z in self.path],
            "instances": instances,
        }

    def _color(self, label, prompts):
        if label in prompts:
            return LABEL_COLORS[list(prompts).index(label) % len(LABEL_COLORS)]
        return (90, 90, 90)

    def _view(self, size):
        """Overview frame matching portal_map: +x right, -z up. Extent comes from known cells."""
        bounds = self.grid.known_bounds() if self.grid is not None else None
        if bounds is None:
            points = [[x, z] for x, z in self.path]
            points += [
                [inst.x, inst.z]
                for inst in self.instances
                if inst.hits >= self.config.confirm_hits
            ]
            if not points:
                return None
            pts = np.asarray(points, dtype=np.float64)
            lo, hi = pts.min(axis=0) - 1.0, pts.max(axis=0) + 1.0
            from_grid = False
        else:
            lo, hi = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
            from_grid = True
        center = (lo + hi) / 2.0
        extent = float(max(hi - lo)) + 2.0
        view_m = max(extent, 4.0)
        return center, view_m / size, from_grid

    def render(self, prompts=None, size=800):
        """Occupancy underneath, confirmed instances as numbered dots, legend at the top left."""
        if prompts is None:
            prompts = []
            for inst in self.instances:
                if inst.label not in prompts:
                    prompts.append(inst.label)
        prompts = list(prompts)
        view = self._view(size)
        image = Image.new("RGB", (size, size), UNKNOWN)
        if view is None:
            return image
        center, meters_per_px, from_grid = view
        right = np.array([1.0, 0.0])
        forward = np.array([0.0, -1.0])
        if from_grid:
            axis = (np.arange(size) + 0.5 - size / 2.0) * meters_per_px
            right_off, up_off = np.meshgrid(axis, -axis)
            world = center + right_off.reshape(-1, 1) * right + up_off.reshape(-1, 1) * forward
            codes = self.grid.codes(world).reshape(size, size)
            canvas = np.empty((size, size, 3), dtype=np.uint8)
            canvas[:] = UNKNOWN
            canvas[codes == 1] = FREE
            canvas[codes == 2] = OCCUPIED
            image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)

        def to_px(point):
            delta = np.asarray(point, dtype=np.float64) - center
            return (
                float(delta @ right) / meters_per_px + size / 2.0,
                size / 2.0 - float(delta @ forward) / meters_per_px,
            )

        smoothed = _smooth_path(self.path)
        if len(smoothed) >= 2:
            draw.line([to_px(point) for point in smoothed], fill=TRAJECTORY, width=2)
        if self.path:
            sx, sy = to_px(self.path[0])
            draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), outline=(0, 0, 0), width=2)
        indexes = self.label_indexes()
        number_font = _load_font(14)
        dot = 11
        for inst in self.instances:
            if inst.hits < self.config.confirm_hits:
                continue
            color = self._color(inst.label, prompts)
            cx, cy = to_px((inst.x, inst.z))
            draw.ellipse((cx - dot, cy - dot, cx + dot, cy + dot), fill=color)
            _draw_centered(draw, (cx, cy), str(indexes[inst.id]), number_font, (255, 255, 255))
        if self.heading is not None and self.path:
            origin = to_px(self.path[-1])
            tip = to_px(np.asarray(self.path[-1], dtype=np.float64) + self.heading * 0.45)
            _draw_arrow(draw, origin, tip, AGENT)

        legend = _legend_rows(self, prompts)
        return _decorate(image, legend, meters_per_px)


def _draw_arrow(draw, origin, tip, fill):
    ax, ay = origin
    tx, ty = tip
    dx, dy = tx - ax, ty - ay
    length = float(np.hypot(dx, dy))
    if length < 1:
        return
    if length < 18:
        dx, dy = dx / length * 18, dy / length * 18
        tx, ty = ax + dx, ay + dy
        length = 18.0
    side_x, side_y = -dy / length, dx / length
    wing = 7
    back = 0.25
    left = (ax + dx * back + side_x * wing, ay + dy * back + side_y * wing)
    right = (ax + dx * back - side_x * wing, ay + dy * back - side_y * wing)
    draw.polygon([(tx, ty), left, right], fill=fill)


def _legend_rows(smap, prompts):
    confirm = smap.config.confirm_hits
    rows = []
    for label in prompts:
        count = sum(1 for inst in smap.instances if inst.label == label and inst.hits >= confirm)
        rows.append((smap._color(label, prompts), f"{label}  {count}"))
    rows.append(((0, 0, 0), "start"))
    rows.append((AGENT, "end"))
    return rows


def _decorate(image, rows, meters_per_px):
    """Legend at the top left and a 1 m scale bar. Both sit on a light plate."""
    font = _load_font(18)
    swatch = 14
    gap = 8
    line = 24
    pad = 10
    widths = [_text_box(text, font)[2] - _text_box(text, font)[0] for _, text in rows]
    box_w = pad * 2 + swatch + gap + (max(widths) if widths else 0)
    box_h = pad * 2 + line * len(rows)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    plate = (255, 255, 255, 220)
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle((8, 8, 8 + box_w, 8 + box_h), radius=6, fill=plate)
    else:
        draw.rectangle((8, 8, 8 + box_w, 8 + box_h), fill=plate)
    for index, (color, text) in enumerate(rows):
        y = 8 + pad + index * line
        draw.rectangle((8 + pad, y + 3, 8 + pad + swatch, y + 3 + swatch), fill=color + (255,))
        draw.text((8 + pad + swatch + gap, y), text, fill=(20, 20, 20, 255), font=font)
    bar = 1.0 / meters_per_px
    x0, y0 = 16, image.size[1] - 28
    label = _text_box("1 m", font)
    label_w = label[2] - label[0]
    draw.rectangle((x0 - 6, y0 - 22, x0 + max(bar, label_w) + 8, y0 + 10), fill=plate)
    draw.line((x0, y0, x0 + bar, y0), fill=(0, 0, 0, 255), width=4)
    draw.text((x0, y0 - 20), "1 m", fill=(0, 0, 0, 255), font=font)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def _has_token(text, token):
    import re

    return re.search(rf"(^|[_\-]){re.escape(token)}($|[_\-])", text) is not None


def _match_episode(episodes, episode_dir, scene_id=None, episode_id=None):
    import os

    from parse_instructions import scene_key

    if scene_id and episode_id is not None:
        hits = [
            row for row in episodes
            if scene_key(row.get("scene_id", "")) == scene_key(scene_id)
            and str(row.get("episode_id")) == str(episode_id)
        ]
        if len(hits) == 1:
            return hits[0]
        raise SystemExit(f"parsed file has {len(hits)} rows for {scene_id} / {episode_id}")
    folder = os.path.basename(os.path.normpath(episode_dir))
    hits = []
    for row in episodes:
        key = scene_key(row.get("scene_id", ""))
        eid = str(row.get("episode_id"))
        if key and _has_token(folder, key) and _has_token(folder, eid):
            hits.append(row)
    if len(hits) == 1:
        return hits[0]
    raise SystemExit(
        f"cannot match {folder} to parsed_instructions.json ({len(hits)} hits). "
        "Pass --scene-id and --episode-id."
    )


def self_check():
    config = SemanticConfig(erode_px=0, border_px=0, min_pixels=10, depth_band_m=0.5, pixel_stride=1, min_score=0.0)
    depth = np.ones((120, 160), dtype=np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    col = int(round((160 - 1) / 2.0))
    mask[40:60, col - 5:col + 6] = True
    depth[mask] = 4.0
    points = mask_points(mask, depth, config)
    assert len(points) >= 10
    assert abs(float(points[:, 0].mean())) < 0.15
    assert abs(float(points[:, 1].mean()) - 4.0) < 0.05

    pose = np.eye(4)
    world = camera_xz_to_world(points, pose)
    assert abs(float(world[:, 1].mean()) + 4.0) < 0.05

    smap = SemanticMap(SemanticConfig(min_score=0.0, assoc_dist_m=0.8, confirm_hits=2))
    assert smap.observe("doorway", world, 0.9, "000.jpg") == "new"
    shifted = world + np.array([0.3, 0.0])
    assert smap.observe("doorway", shifted, 0.8, "004.jpg") == "updated"
    assert smap.observe("doorway", world + np.array([3.0, 0.0]), 0.7, "008.jpg") == "new"
    assert len(smap.instances) == 2
    assert smap.instances[0].hits == 2
    assert smap.instances[0].to_dict(2)["state"] == "confirmed"
    assert smap.instances[1].to_dict(2)["state"] == "candidate"
    payload = smap.to_dict(["doorway"])
    assert [row["index"] for row in payload["instances"]] == [1, 2]
    assert payload["path_xz"] == []

    def pose_at(x, z, yaw):
        cosine, sine = np.cos(yaw), np.sin(yaw)
        pose = np.eye(4)
        pose[0, 0], pose[0, 2] = cosine, sine
        pose[2, 0], pose[2, 2] = -sine, cosine
        pose[0, 3], pose[2, 3] = x, z
        return pose

    straight = [pose_at(i * 0.2, 0.0, 0.0) for i in range(6)]
    assert select_keyframes(straight) == [0, 3, 5]
    assert select_keyframes(straight, stride=2) == [0, 2, 4, 5]
    turns = [pose_at(0.0, 0.0, np.deg2rad(deg)) for deg in (0, 20, 40, 60)]
    assert select_keyframes(turns) == [0, 2, 3]
    assert select_keyframes([None, straight[0], None]) == [1]
    smoothed = _smooth_path([[0, 0], [1, 1], [2, 0], [3, 1]])
    assert np.allclose(smoothed[0], [0, 0]) and np.allclose(smoothed[-1], [3, 1])

    drawn = smap.render(["doorway"])
    assert drawn.size == (800, 800)
    assert (np.asarray(drawn) == np.array(LABEL_COLORS[0])).all(axis=-1).any()

    walked = SemanticMap()
    walked.note_frame(pose_at(0.0, 0.0, 0.0), np.full((48, 64), 2.0, np.float32))
    walked.note_frame(pose_at(0.3, 0.0, 0.0))
    assert len(walked.path) == 2 and walked.grid is not None and walked.heading is not None
    assert walked.render(["doorway"]).size == (800, 800)

    parsed = {
        "subtasks": [
            {"landmark": "bathroom", "near": ""},
            {"landmark": "second doorway", "near": "flower pot"},
            {"landmark": "potted plant", "near": "flower pot"},
        ]
    }
    assert prompts_from_parsed(parsed) == ["second doorway", "flower pot", "potted plant"]

    episodes = [
        {"scene_id": "mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb", "episode_id": "140"},
        {"scene_id": "mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb", "episode_id": "79"},
        {"scene_id": "mp3d/2azQ1b91cZZ/2azQ1b91cZZ.glb", "episode_id": "14"},
    ]
    matched = _match_episode(episodes, "/data/2azQ1b91cZZ_140")
    assert matched["episode_id"] == "140"
    print("self-check ok")


def main():
    parser = argparse.ArgumentParser(description="语义地图的几何和融合，不含 SAM3")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    parser.error("pass --self-check, or run scripts/build_semantic_map.py")


if __name__ == "__main__":
    main()
