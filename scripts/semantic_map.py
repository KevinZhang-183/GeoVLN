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

from extract_portals import camera_xz_to_world, focal_px
from parse_instructions import ARTICLES, ROOM_WORDS, _landmark


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


class SemanticMap:
    def __init__(self, config=None):
        self.config = config or SemanticConfig()
        self.instances = []
        self.path = []
        self._next_id = 1

    def note_agent(self, pose):
        pose = np.asarray(pose, dtype=np.float64)
        self.path.append([float(pose[0, 3]), float(pose[2, 3])])

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

    def describe(self):
        lines = []
        for inst in self.instances:
            state = "confirmed" if inst.hits >= self.config.confirm_hits else "candidate"
            lines.append(
                f"{inst.id}: {inst.label} {state} ({inst.x:.2f}, {inst.z:.2f}) "
                f"hits={inst.hits} mean_score={inst.score_sum / inst.hits:.2f} best={inst.best_frame}"
            )
        return lines

    def to_dict(self, prompts):
        confirm = self.config.confirm_hits
        instances = [inst.to_dict(confirm) for inst in self.instances]
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

    def render(self, size=800):
        points = [tuple(p) for p in self.path]
        points += [(inst.x, inst.z) for inst in self.instances]
        image = Image.new("RGB", (size, size), (245, 245, 245))
        if not points:
            return image
        xs = [p[0] for p in points]
        zs = [p[1] for p in points]
        pad = 1.0
        min_x, max_x = min(xs) - pad, max(xs) + pad
        min_z, max_z = min(zs) - pad, max(zs) + pad
        span = max(max_x - min_x, max_z - min_z, 1.0)
        scale = (size - 40) / span

        def px(x, z):
            return (20 + (x - min_x) * scale, size - 20 - (z - min_z) * scale)

        draw = ImageDraw.Draw(image)
        if len(self.path) >= 2:
            draw.line([px(x, z) for x, z in self.path], fill=(0, 150, 255), width=2)
        colors = ((0, 140, 70), (180, 90, 0), (140, 40, 160), (0, 90, 180), (160, 40, 40))
        labels = []
        for inst in self.instances:
            if inst.label not in labels:
                labels.append(inst.label)
            color = colors[labels.index(inst.label) % len(colors)]
            cx, cy = px(inst.x, inst.z)
            radius = max(4, inst.radius_m * scale)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline=color, width=3)
            draw.text((cx + 4, cy - 6), f"{inst.id}:{inst.label}", fill=color)
        return image


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
