"""Build a fused portal map for collected episodes and save per-frame views.

Each episode folder needs ``rgb/*.jpg``, ``depth/*.npy`` and ``meta.json``
with a ``position`` and ``rotation_wxyz`` for every frame.

Per frame, ``<out-name>/<frame>.jpg`` shows the RGB with detections numbered
by track id, the egocentric map, and the whole map so far. ``describe.txt``
holds the candidate list after each frame, ``portal_map.json`` the final map.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_portals import PortalConfig, habitat_agent_matrix, load_depth  # noqa: E402
from portal_map import MapConfig, PortalMap, draw_labels  # noqa: E402

NAME_KEYS = ("frame", "name", "rgb", "image", "file")


def _stem(value):
    return os.path.splitext(os.path.basename(str(value)))[0]


def load_meta(episode_dir, names):
    """Map each frame stem to its meta entry.

    Accepts a list or ``{"frames": [...]}`` whose entries name their frame, a
    dict keyed by frame name, or a plain list in the same order as ``names``.
    """
    with open(os.path.join(episode_dir, "meta.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    if isinstance(meta, dict) and "frames" in meta:
        meta = meta["frames"]
    if isinstance(meta, dict):
        keyed = {_stem(k): v for k, v in meta.items()}
    else:
        keyed = {}
        for entry in meta:
            key = next((k for k in NAME_KEYS if k in entry), None)
            if key is not None:
                keyed[_stem(entry[key])] = entry
        if not keyed:
            stems = [_stem(n) for n in names]
            if len(meta) < len(stems):
                raise ValueError(f"{episode_dir}: meta.json has {len(meta)} entries for {len(stems)} frames")
            keyed = dict(zip(stems, meta))
    for stem in list(keyed):
        if stem.isdigit():
            keyed.setdefault(str(int(stem)), keyed[stem])
    return keyed


def _lookup(keyed, stem):
    if stem in keyed:
        return keyed[stem]
    return keyed.get(str(int(stem))) if stem.isdigit() else None


def _composite(rgb, ego, overview):
    height = rgb.height
    ego = ego.resize((height, height))
    overview = overview.resize((height, height))
    out = Image.new("RGB", (rgb.width + 2 * height, height), (0, 0, 0))
    out.paste(rgb, (0, 0))
    out.paste(ego, (rgb.width, 0))
    out.paste(overview, (rgb.width + height, 0))
    return out


def process_episode(episode_dir, out_name, stride, map_config, portal_config, save_frames=True):
    rgb_dir = os.path.join(episode_dir, "rgb")
    depth_dir = os.path.join(episode_dir, "depth")
    out_dir = os.path.join(episode_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(rgb_dir) if n.endswith(".jpg"))
    meta = load_meta(episode_dir, names)
    pmap = PortalMap(map_config, portal_config)
    log = []
    for step, name in enumerate(names[::stride]):
        stem = _stem(name)
        depth_path = os.path.join(depth_dir, stem + ".npy")
        entry = _lookup(meta, stem)
        if not os.path.exists(depth_path) or entry is None:
            print(f"  skip {name}: missing depth or meta")
            continue
        pose = habitat_agent_matrix(entry["position"], entry["rotation_wxyz"])
        depth = load_depth(depth_path, portal_config)
        labels = pmap.update(depth, pose, step, name)
        lines = pmap.describe()
        log.append(f"{name} detections={[tid for _, tid in labels]}")
        log.extend("  " + line for line in lines)
        if save_frames:
            rgb = Image.open(os.path.join(rgb_dir, name)).convert("RGB")
            frame = _composite(
                draw_labels(rgb, labels, pmap),
                pmap.render(egocentric=True),
                pmap.render(egocentric=False),
            )
            frame.save(os.path.join(out_dir, name))

    states = Counter(tr.state for tr in pmap.tracks)
    hits = [tr.hits for tr in pmap.tracks]
    summary = (
        f"{os.path.basename(episode_dir)} frames={pmap.step + 1} tracks={len(pmap.tracks)} "
        + " ".join(f"{k}={v}" for k, v in sorted(states.items()))
        + (f" mean_hits={np.mean(hits):.1f}" if hits else "")
    )
    with open(os.path.join(out_dir, "describe.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(log) + "\n")
    with open(os.path.join(out_dir, "portal_map.json"), "w", encoding="utf-8") as handle:
        json.dump(pmap.to_dict(), handle)
    print(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Fuse per-frame openings into a portal map.")
    parser.add_argument("--frames", help="Directory of episode folders, each with rgb/, depth/ and meta.json.")
    parser.add_argument("--episode", help="One episode folder with rgb/, depth/ and meta.json.")
    parser.add_argument("--out-name", default="portal_map")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--near-jamb", type=float, default=1.2)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--no-frames", action="store_true", help="Only write describe.txt and portal_map.json.")
    args = parser.parse_args()
    if not args.frames and not args.episode:
        parser.error("pass --frames or --episode")
    portal_config = PortalConfig(near_jamb_m=args.near_jamb)
    map_config = MapConfig(resolution=args.resolution)
    if args.episode:
        episodes, summary_dir = [args.episode], args.episode
    else:
        episodes = [
            os.path.join(args.frames, name)
            for name in sorted(os.listdir(args.frames))
            if os.path.isdir(os.path.join(args.frames, name, "rgb"))
            and os.path.isdir(os.path.join(args.frames, name, "depth"))
            and os.path.exists(os.path.join(args.frames, name, "meta.json"))
        ]
        summary_dir = args.frames
    print("episodes", len(episodes))
    summaries = [
        process_episode(ep, args.out_name, args.stride, map_config, portal_config, not args.no_frames)
        for ep in episodes
    ]
    with open(os.path.join(summary_dir, args.out_name + "_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(summaries) + "\n")


if __name__ == "__main__":
    main()
