"""Detect walkable openings in a depth image and project them to the ground.

An opening is a vertical gap: nearer surfaces on both sides, and a farther
region between them that reaches the floor. Yellow openings have a jamb on
both sides. Orange openings touch the image border, so only one jamb is
visible. Openings whose jamb is closer than ``near_jamb_m`` are dropped.

Camera ground frame used by the projection helpers: x is right, z is forward,
both in meters. Habitat's agent frame points backward along z, so
``camera_xz_to_world`` flips z before applying the agent pose.
"""

import argparse
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageDraw

YELLOW = (255, 255, 0)
ORANGE = (255, 140, 0)


@dataclass
class PortalConfig:
    hfov: float = 79.0
    camera_height: float = 1.25
    depth_min: float = 0.2
    depth_max: float = 10.0
    near_jamb_m: float = 1.2
    min_width_m: float = 0.40
    max_width_m: float = 3.15
    edge_max_width_m: float = 2.45
    squat_min_width_m: float = 1.5
    min_aspect: float = 1.25
    rise_m: float = 1.35
    edge_rise_m: float = 1.4
    inside_gap_m: float = 1.0
    deep_gap_m: float = 0.8
    deep_fraction: float = 0.55
    row_gap_m: float = 0.7
    floor_side_gap_m: float = 0.55
    floor_margin: float = 0.07
    far_jamb_m: float = 6.0
    shallow_jamb_m: float = 4.0
    shallow_delta_m: float = 1.9
    shallow_max_width_m: float = 1.15
    skew_min_width_m: float = 2.4
    skew_delta_m: float = 1.5
    pillar_min_width_m: float = 1.15
    pillar_near_m: float = 0.65
    pillar_jump_m: float = 1.35
    open_side_min_width_m: float = 2.2
    overlap: float = 0.55
    body_top: float = 0.18
    body_bottom: float = 0.62
    body_quantile: float = 0.55
    min_row_fraction: float = 0.22
    min_bottom_fraction: float = 0.38
    far_row_fraction: float = 0.12
    extend_fraction: float = 0.22
    smooth_k: int = 7
    side_px: int = 28
    gap_px: int = 10
    event_px: int = 36
    search_px: int = 70
    max_span_px: int = 320
    min_span_px: int = 16
    row_join_px: int = 40
    box_pad_x: int = 6
    box_pad_y: int = 6
    scale_normalized: bool = True


@dataclass
class Portal:
    kind: str
    box: tuple
    span: tuple
    width_m: float
    inside_m: float
    jamb_m: float
    potential: float
    reason: str = ""
    left_jamb_m: float = None
    right_jamb_m: float = None

    def segment_xz(self, depth_shape, config):
        return doorway_segment(
            self.span, self.jamb_m, depth_shape, config, self.left_jamb_m, self.right_jamb_m
        )


def focal_px(width, hfov):
    return (width / 2.0) / np.tan(np.deg2rad(hfov) / 2.0)


def load_depth(path, config=None):
    config = config or PortalConfig()
    depth = np.squeeze(np.load(path)).astype(np.float32)
    if config.scale_normalized and np.nanmax(depth) <= 1.5:
        depth = depth * config.depth_max
    return depth


def _column_quantile(depth, valid, y0, y1, q):
    width = depth.shape[1]
    out = np.full(width, np.nan, dtype=np.float32)
    band, ok = depth[y0:y1], valid[y0:y1]
    for x in range(width):
        vals = band[:, x][ok[:, x]]
        if vals.size >= 8:
            out[x] = np.quantile(vals, q)
    return out


def _smooth(values, k):
    out = values.copy()
    half = k // 2
    for i in range(len(values)):
        window = values[max(0, i - half):min(len(values), i + half + 1)]
        window = window[np.isfinite(window)]
        if window.size:
            out[i] = float(np.median(window))
    return out


def _side_median(body, a, b):
    if b <= a:
        return None
    vals = body[a:b]
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return None
    return float(np.median(vals))


def _thin_events(events, gap):
    kept = []
    for event in events:
        if kept and event[0] - kept[-1][0] < gap:
            if event[3] > kept[-1][3]:
                kept[-1] = event
            continue
        kept.append(event)
    return kept


def _flank_kind(body, edge, direction, z_ref, fx, config):
    if z_ref is None or not np.isfinite(z_ref):
        return "border"
    near_px = 0
    seen_near = False
    for step in range(1, 110):
        x = int(edge) + direction * step
        if x < 0 or x >= len(body):
            return "border" if seen_near else "none"
        z = body[x]
        if not np.isfinite(z):
            continue
        if z <= z_ref + 1.0:
            seen_near = True
            near_px += 1
            continue
        if not seen_near:
            if step < 16:
                continue
            return "none"
        near_m = near_px / fx * max(float(z_ref), 0.4)
        if z >= z_ref + config.pillar_jump_m and near_m < config.pillar_near_m:
            return "pillar"
        return "wall"
    return "wall" if seen_near else "none"


def _gate(config, depth, body, box, span, width_m, inside_z, jamb_z, fx):
    height, width = depth.shape
    left, right = span
    x0, y0, x1, y1 = box
    if jamb_z < config.near_jamb_m:
        return "near"
    if width_m > config.max_width_m:
        return "too_wide"
    aspect = (y1 - y0) / max(x1 - x0, 1)
    if width_m > config.squat_min_width_m and aspect < config.min_aspect:
        return "squat"
    edge = left <= 1 or right >= width - 2
    if edge and width_m > config.edge_max_width_m:
        return "edge_wide"
    floor_row = (height - 1) / 2.0 + fx * config.camera_height / max(jamb_z, 0.4)
    if y1 < min(height - 1, floor_row) - config.floor_margin * height:
        return "above_floor"
    if (
        jamb_z > config.shallow_jamb_m
        and (inside_z - jamb_z) < config.shallow_delta_m
        and width_m < config.shallow_max_width_m
    ):
        return "shallow"
    left_z = None if left <= 1 else _side_median(body, max(0, left - 46), max(0, left - 10))
    right_z = None if right >= width - 2 else _side_median(body, min(width, right + 10), min(width, right + 46))
    if (
        width_m > config.skew_min_width_m
        and left_z is not None
        and right_z is not None
        and abs(left_z - right_z) > config.skew_delta_m
    ):
        return "skew"
    left_kind = "border" if left <= 1 else _flank_kind(body, left, -1, left_z if left_z is not None else jamb_z, fx, config)
    right_kind = "border" if right >= width - 2 else _flank_kind(body, right, 1, right_z if right_z is not None else jamb_z, fx, config)
    if width_m > config.pillar_min_width_m and (left_kind == "pillar" or right_kind == "pillar"):
        return "pillar"
    if width_m > config.open_side_min_width_m and (left_kind in ("pillar", "none") or right_kind in ("pillar", "none")):
        return "open_side"
    return ""


def _jamb_depths(body, span, jamb_z, config):
    """Depth of the wall just outside each side of the span. ``None`` on the image border."""
    width = len(body)
    left, right = span
    out = []
    for at_border, a, b in (
        (left <= 1, max(0, left - config.side_px), max(0, left - 2)),
        (right >= width - 2, min(width, right + 3), min(width, right + config.side_px + 1)),
    ):
        if at_border:
            out.append(None)
            continue
        z = _side_median(body, a, b)
        if z is None or z > jamb_z + config.rise_m:
            z = float(jamb_z)
        out.append(round(z, 2))
    return out


def _dedupe(portals, overlap):
    portals = sorted(portals, key=lambda item: item.potential, reverse=True)
    kept = []
    for item in portals:
        x0, y0, x1, y1 = item.box
        area = max(1, (x1 - x0) * (y1 - y0))
        if any(
            max(0, min(x1, b) - max(x0, a)) * max(0, min(y1, d) - max(y0, c)) / area > overlap
            for a, c, b, d in (other.box for other in kept)
        ):
            continue
        kept.append(item)
    return kept


def detect(depth, config=None):
    """Return ``(openings, edges, dropped)`` for one depth image in meters."""
    config = config or PortalConfig()
    depth = np.squeeze(depth).astype(np.float32)
    height, width = depth.shape
    fx = focal_px(width, config.hfov)
    valid = np.isfinite(depth) & (depth > config.depth_min)
    body = _smooth(
        _column_quantile(depth, valid, int(height * config.body_top), int(height * config.body_bottom), config.body_quantile),
        config.smooth_k,
    )
    side = config.side_px
    gap = config.gap_px
    rises = []
    for x in range(30, width - 30):
        before = _side_median(body, x - side, x - gap)
        after = _side_median(body, x + gap, x + side)
        if before is None or after is None or after <= before + config.rise_m:
            continue
        rises.append((x, before, after, after - before))
    rises = _thin_events(rises, config.event_px)
    spans = []
    head = _side_median(body, 0, 40)
    for x in range(40, width // 2):
        after = _side_median(body, x, min(width, x + side))
        if head is None or after is None or head < after + config.edge_rise_m:
            continue
        if np.nanmedian(body[x:min(width, x + 40)]) < head - 1.2:
            spans.append((0, x, after))
            break
    tail = _side_median(body, width - 40, width)
    for x in range(width - 40, width // 2, -1):
        before = _side_median(body, max(0, x - side), x)
        if tail is None or before is None or tail < before + config.edge_rise_m:
            continue
        spans.append((x, width - 1, before))
        break
    for rx, before, after, _ in rises:
        window = body[rx:min(width, rx + config.search_px)]
        if not np.isfinite(window).any():
            continue
        peak_x = rx + int(np.nanargmax(window))
        limit = before + 1.15
        left = right = peak_x
        while left > 0 and np.isfinite(body[left - 1]) and body[left - 1] > limit:
            left -= 1
        while right < width - 1 and np.isfinite(body[right + 1]) and body[right + 1] > limit:
            right += 1
        if any(left + 8 < item[0] < right - 8 and item[0] != rx for item in rises):
            continue
        if right - left > config.max_span_px or right - left < config.min_span_px:
            continue
        spans.append((left, right, before))

    openings, edges, dropped = [], [], []
    for left, right, jamb_z in spans:
        hole = body[left:right + 1]
        hole = hole[np.isfinite(hole)]
        if hole.size < 8:
            continue
        inside_z = float(np.median(hole))
        if inside_z < jamb_z + config.inside_gap_m:
            continue
        deep = np.isfinite(body[left:right + 1]) & (body[left:right + 1] > jamb_z + config.deep_gap_m)
        if float(deep.mean()) < config.deep_fraction:
            continue
        mid0 = left + max(1, (right - left) // 3)
        mid1 = right - max(1, (right - left) // 3)
        if mid1 <= mid0:
            mid0, mid1 = left, right
        row_med = np.full(height, np.nan)
        for y in range(height):
            vals = depth[y, mid0:mid1 + 1]
            vals = vals[np.isfinite(vals) & (vals > config.depth_min)]
            if vals.size >= 3:
                row_med[y] = float(np.median(vals))
        far = np.flatnonzero(np.isfinite(row_med) & (row_med > jamb_z + config.row_gap_m))
        if far.size < int(height * config.far_row_fraction):
            continue
        runs, start, prev = [], int(far[0]), int(far[0])
        for y in list(far[1:]) + [10 ** 9]:
            if y <= prev + config.row_join_px:
                prev = min(int(y), height - 1)
                continue
            runs.append((start, prev))
            start = prev = int(y) if y < height else prev
        y0, y1 = max(runs, key=lambda item: item[1] - item[0])
        if y1 - y0 < int(height * config.min_row_fraction) or y1 < int(height * config.min_bottom_fraction):
            continue
        extra = 0
        while y1 + 1 < height and extra < int(height * config.extend_fraction):
            center = row_med[y1 + 1]
            side_chunks = []
            if left > 8:
                side_chunks.append(depth[y1 + 1, max(0, left - side):max(0, left - 6)])
            if right < width - 8:
                side_chunks.append(depth[y1 + 1, min(width, right + 6):min(width, right + side)])
            side_vals = np.concatenate(side_chunks) if side_chunks else np.array([])
            side_vals = side_vals[np.isfinite(side_vals) & (side_vals > config.depth_min)]
            side_z = float(np.median(side_vals)) if side_vals.size >= 3 else None
            if not np.isfinite(center) or side_z is None or center < side_z + config.floor_side_gap_m:
                break
            y1 += 1
            extra += 1
        if jamb_z > config.far_jamb_m and y1 < int(height * 0.72):
            continue
        width_m = (right - left) / fx * max(float(jamb_z), 0.4)
        if width_m < config.min_width_m:
            continue
        potential = width_m * (inside_z - float(jamb_z))
        box = (
            max(0, int(left) - config.box_pad_x),
            max(0, int(y0) - config.box_pad_y),
            min(width - 1, int(right) + config.box_pad_x),
            min(height - 1, int(y1) + config.box_pad_y),
        )
        span = (int(left), int(right))
        reason = _gate(config, depth, body, box, span, width_m, inside_z, jamb_z, fx)
        left_jamb, right_jamb = _jamb_depths(body, span, jamb_z, config)
        portal = Portal(
            "dropped" if reason else "opening",
            box,
            span,
            round(width_m, 2),
            round(inside_z, 2),
            round(float(jamb_z), 2),
            potential,
            reason or "",
            left_jamb,
            right_jamb,
        )
        if reason:
            dropped.append(portal)
        elif left <= 1 or right >= width - 2:
            portal.kind = "edge"
            edges.append(portal)
        else:
            openings.append(portal)
    return (
        _dedupe(openings, config.overlap),
        _dedupe(edges, config.overlap),
        dropped,
    )


def opening_crop(rgb, portal):
    """RGB crop of one opening. ``rgb`` is a PIL image or an HxWx3 array."""
    image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
    x0, y0, x1, y1 = portal.box
    return image.crop((x0, y0, x1 + 1, y1 + 1))


def doorway_segment(span, jamb_m, depth_shape, config=None, left_m=None, right_m=None):
    """Jamb line on the ground, shape (2, 2), columns are x-right and z-forward.

    Each end uses its own jamb depth when given, so an oblique doorway keeps
    its true orientation. A side on the image border falls back to ``jamb_m``.
    """
    config = config or PortalConfig()
    height, width = depth_shape[:2]
    fx = focal_px(width, config.hfov)
    cx = (width - 1) / 2.0
    left, right = span
    zl = float(jamb_m if left_m is None else left_m)
    zr = float(jamb_m if right_m is None else right_m)
    return np.array(
        [[(left - cx) * zl / fx, zl], [(right - cx) * zr / fx, zr]],
        dtype=np.float32,
    )


def opening_points(depth, portal, config=None, stride=4):
    """Valid pixels inside the box, projected to camera ground x-right, z-forward."""
    config = config or PortalConfig()
    depth = np.squeeze(depth).astype(np.float32)
    height, width = depth.shape
    fx = focal_px(width, config.hfov)
    cx = (width - 1) / 2.0
    x0, y0, x1, y1 = portal.box
    crop = depth[y0:y1 + 1, x0:x1 + 1]
    valid = np.isfinite(crop) & (crop > config.depth_min) & (crop <= config.depth_max)
    vs, us = np.nonzero(valid)
    if stride > 1:
        vs, us = vs[::stride], us[::stride]
    z = crop[vs, us].astype(np.float32)
    x = (us + x0 - cx) * z / fx
    return np.stack([x, z], axis=1)


def habitat_agent_matrix(position, rotation_wxyz):
    """4x4 agent pose. Habitat local axes are x right, y up, z backward."""
    w, x, y, z = [float(v) for v in rotation_wxyz]
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rot
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    return pose


def camera_xz_to_world(points_xz, agent_pose):
    """Map camera-ground points to world (x, z). ``agent_pose`` is Habitat's 4x4."""
    pts = np.atleast_2d(np.asarray(points_xz, dtype=np.float64))
    local = np.stack([pts[:, 0], np.zeros(len(pts)), -pts[:, 1], np.ones(len(pts))], axis=1)
    world = (np.asarray(agent_pose, dtype=np.float64) @ local.T).T
    return world[:, [0, 2]]


def stamp_occupancy(grid, origin_xz, resolution, points_xz, value):
    """Write points into ``grid[row, col]``.

    ``origin_xz`` is the world (x, z) of ``grid[0, 0]``. Column increases with
    world x, row increases with world z. ``resolution`` is meters per cell.
    """
    pts = np.atleast_2d(np.asarray(points_xz, dtype=np.float64))
    if pts.size == 0:
        return 0
    cols = np.floor((pts[:, 0] - origin_xz[0]) / resolution).astype(int)
    rows = np.floor((pts[:, 1] - origin_xz[1]) / resolution).astype(int)
    ok = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    grid[rows[ok], cols[ok]] = value
    return int(ok.sum())


def _portal_dict(portal, depth_shape, config):
    segment = portal.segment_xz(depth_shape, config)
    return {
        "kind": portal.kind,
        "reason": portal.reason,
        "box": list(portal.box),
        "span": list(portal.span),
        "width_m": portal.width_m,
        "inside_m": portal.inside_m,
        "jamb_m": portal.jamb_m,
        "left_jamb_m": portal.left_jamb_m,
        "right_jamb_m": portal.right_jamb_m,
        "potential": portal.potential,
        "segment_xz": segment.round(3).tolist(),
    }


def draw_portals(image, openings, edges):
    vis = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(vis)
    for rank, portal in enumerate(openings, 1):
        draw.rectangle(portal.box, outline=YELLOW, width=3)
        draw.text((portal.box[0] + 4, portal.box[1] + 4), str(rank), fill=YELLOW)
    for portal in edges:
        draw.rectangle(portal.box, outline=ORANGE, width=3)
        draw.text((portal.box[0] + 4, portal.box[1] + 4), "E", fill=ORANGE)
    return vis


def process_episode(episode_dir, out_name="portals", stride=4, config=None, save_crops=True):
    config = config or PortalConfig()
    rgb_dir = os.path.join(episode_dir, "rgb")
    depth_dir = os.path.join(episode_dir, "depth")
    out_dir = os.path.join(episode_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    instruction = os.path.join(episode_dir, "instruction.txt")
    if os.path.exists(instruction):
        with open(instruction, encoding="utf-8") as src, open(os.path.join(out_dir, "instruction.txt"), "w", encoding="utf-8") as dst:
            dst.write(src.read())
    names = sorted(n for n in os.listdir(rgb_dir) if n.endswith(".jpg"))[::stride]
    lines = ["# " + os.path.basename(episode_dir)]
    records = []
    for name in names:
        depth_path = os.path.join(depth_dir, name.replace(".jpg", ".npy"))
        if not os.path.exists(depth_path):
            continue
        depth = load_depth(depth_path, config)
        openings, edges, dropped = detect(depth, config)
        image = Image.open(os.path.join(rgb_dir, name)).convert("RGB")
        draw_portals(image, openings, edges).save(os.path.join(out_dir, name))
        if save_crops:
            for rank, portal in enumerate(openings, 1):
                opening_crop(image, portal).save(os.path.join(out_dir, f"{name[:-4]}_{rank}.jpg"))
        lines.append(f"{name} n={len(openings)} edge={len(edges)}")
        for rank, portal in enumerate(openings, 1):
            lines.append(
                f"  {rank} pot={portal.potential:.2f} w={portal.width_m} in={portal.inside_m} jamb={portal.jamb_m} box={portal.box}"
            )
        for portal in edges:
            lines.append(
                f"  edge pot={portal.potential:.2f} w={portal.width_m} in={portal.inside_m} jamb={portal.jamb_m} box={portal.box}"
            )
        for portal in dropped:
            lines.append(
                f"  drop {portal.reason} pot={portal.potential:.2f} w={portal.width_m} in={portal.inside_m} jamb={portal.jamb_m} box={portal.box}"
            )
        records.append({
            "frame": name,
            "openings": [_portal_dict(p, depth.shape, config) for p in openings],
            "edges": [_portal_dict(p, depth.shape, config) for p in edges],
            "dropped": [_portal_dict(p, depth.shape, config) for p in dropped],
        })
        print(f"{name} n={len(openings)} edge={len(edges)}")
    with open(os.path.join(out_dir, "boxes.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "portals.json"), "w", encoding="utf-8") as handle:
        json.dump({"config": asdict(config), "frames": records}, handle)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Detect openings in collected RGB-D frames.")
    parser.add_argument("--frames", help="Directory of episode folders, each with rgb/ and depth/.")
    parser.add_argument("--episode", help="One episode folder with rgb/ and depth/.")
    parser.add_argument("--out-name", default="portals")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--near-jamb", type=float, default=1.2)
    parser.add_argument("--no-crops", action="store_true")
    args = parser.parse_args()
    if not args.frames and not args.episode:
        parser.error("pass --frames or --episode")
    config = PortalConfig(near_jamb_m=args.near_jamb)
    if args.episode:
        episodes = [args.episode]
        summary_dir = args.episode
    else:
        episodes = [
            os.path.join(args.frames, name)
            for name in sorted(os.listdir(args.frames))
            if os.path.isdir(os.path.join(args.frames, name, "rgb"))
            and os.path.isdir(os.path.join(args.frames, name, "depth"))
        ]
        summary_dir = args.frames
    all_lines = []
    print("episodes", len(episodes))
    for episode in episodes:
        print("\n#", os.path.basename(episode))
        all_lines.extend(process_episode(episode, args.out_name, args.stride, config, not args.no_crops))
    with open(os.path.join(summary_dir, args.out_name + "_boxes.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(all_lines) + "\n")
    print("saved", args.out_name)


if __name__ == "__main__":
    main()
