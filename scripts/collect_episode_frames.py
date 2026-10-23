#!/usr/bin/env python3
"""沿 R2R 参考路径采集 RGB、深度和指令，供门洞地图和语义地图使用。

只采集解析结果里列出的 episode。输出的每个目录是 ``{场景}_{episode_id}``：

    rgb/000.jpg
    depth/000.npy
    meta.json          每帧的 position、rotation_wxyz
    instruction.txt

深度按 Habitat 的归一化值保存。``load_depth`` 看到最大值不超过 1.5 时会乘上
``depth_max``（10 米）。相机和步长用 ``config/vln_r2r.yaml``：640×480、视场角
79 度、前进 0.25 米、转弯 15 度。

在服务器的 GeoVLN 目录下，这 20 条来自 val_unseen：

    python scripts/collect_episode_frames.py \\
        --parsed /path/to/parsed_instructions_qwen3VL-32b.json \\
        --output /root/data1/StreamVLN/data/r2r_frames_20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streamvln"))

from parse_instructions import scene_key  # noqa: E402


def load_wanted(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload["episodes"] if isinstance(payload, dict) else payload
    wanted = []
    seen = set()
    for row in episodes:
        if not isinstance(row, dict) or "scene_id" not in row or "episode_id" not in row:
            continue
        key = (scene_key(row["scene_id"]), str(row["episode_id"]))
        if key in seen:
            continue
        seen.add(key)
        instruction = row.get("instruction") or ""
        if not isinstance(instruction, str):
            instruction = ""
        wanted.append({"scene": key[0], "episode_id": key[1], "instruction": instruction.strip()})
    if not wanted:
        raise SystemExit(f"{path} 里没有带 scene_id 和 episode_id 的条目")
    return wanted


def episode_dirname(scene: str, episode_id: str) -> str:
    return f"{scene}_{episode_id}"


def _rotation_wxyz(rotation) -> list[float]:
    if hasattr(rotation, "w"):
        return [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)]
    import quaternion

    return [float(value) for value in quaternion.as_float_array(rotation)]


def _pose(sim) -> tuple[list[float], list[float]]:
    state = sim.get_agent_state()
    position = [float(value) for value in state.position]
    return position, _rotation_wxyz(state.rotation)


def _save_frame(rgb_dir: Path, depth_dir: Path, index: int, observation, sim) -> dict:
    import numpy as np
    from PIL import Image

    name = f"{index:03d}.jpg"
    Image.fromarray(observation["rgb"]).convert("RGB").save(rgb_dir / name, quality=95)
    depth = np.squeeze(observation["depth"]).astype(np.float32)
    np.save(depth_dir / f"{index:03d}.npy", depth)
    position, rotation = _pose(sim)
    return {"frame": name, "position": position, "rotation_wxyz": rotation}


def _near_waypoint(sim, point, radius: float) -> bool:
    import numpy as np

    position = np.asarray(sim.get_agent_state().position, dtype=np.float64)
    return float(np.linalg.norm(position - np.asarray(point, dtype=np.float64))) <= radius + 0.05


def _already_collected(out_dir: Path) -> bool:
    return (out_dir / "instruction.txt").is_file() and (out_dir / "meta.json").is_file() and any((out_dir / "rgb").glob("*.jpg"))


def collect_one(env, episode, out_dir: Path, parsed_instruction: str, max_steps: int) -> dict:
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth"
    rgb_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)

    env.current_episode = episode
    observation = env.reset()
    reference = list(episode.reference_path or [])
    text = episode.instruction.instruction_text
    if not isinstance(text, str):
        text = parsed_instruction
    text = text.strip()
    (out_dir / "instruction.txt").write_text(text + "\n", encoding="utf-8")

    frames = [_save_frame(rgb_dir, depth_dir, 0, observation, env.sim)]
    goal_radius = 0.5
    follower = ShortestPathFollower(env.sim, goal_radius=goal_radius, return_one_hot=False)
    waypoint = 1
    unreachable = False
    while waypoint < len(reference) and len(frames) < max_steps and not env.episode_over:
        action = int(follower.get_next_action(reference[waypoint]))
        if action == HabitatSimActions.STOP:
            if not _near_waypoint(env.sim, reference[waypoint], goal_radius):
                unreachable = True
                break
            waypoint += 1
            if waypoint == len(reference) - 1:
                goal_radius = 0.25
                follower = ShortestPathFollower(env.sim, goal_radius=goal_radius, return_one_hot=False)
            continue
        observation = env.step(action)
        frames.append(_save_frame(rgb_dir, depth_dir, len(frames), observation, env.sim))

    reached = waypoint >= len(reference) and not unreachable
    payload = {
        "scene_id": scene_key(episode.scene_id),
        "episode_id": str(episode.episode_id),
        "instruction": text,
        "reference_points": len(reference),
        "reached_reference_end": reached,
        "unreachable": unreachable,
        "frames": frames,
    }
    (out_dir / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if parsed_instruction and parsed_instruction != text:
        print(f"  warning: parsed instruction differs from the dataset text for {payload['episode_id']}")
    return {
        "scene_id": payload["scene_id"],
        "episode_id": payload["episode_id"],
        "frames": len(frames),
        "reached": reached,
        "dir": str(out_dir),
    }


def _make_env(args, scenes: list[str]):
    import habitat
    from habitat.config import read_write
    from habitat_baselines.config.default import get_config as get_habitat_config

    import habitat_extensions.measures  # noqa: F401

    config = get_habitat_config(args.config)
    data_path = args.data_path or f"data/datasets/r2r/{args.split}/{args.split}.json.gz"
    with read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.dataset.data_path = data_path
        if args.scenes_dir:
            config.habitat.dataset.scenes_dir = args.scenes_dir
        config.habitat.dataset.content_scenes = scenes
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    return habitat.Env(config=config)


def collect(args) -> None:
    wanted = load_wanted(args.parsed)
    scenes = sorted({row["scene"] for row in wanted})
    by_key = {(row["scene"], row["episode_id"]): row for row in wanted}
    print(f"episodes {len(wanted)} across {len(scenes)} scenes: {', '.join(scenes)}", flush=True)
    env = _make_env(args, scenes)
    found = {}
    for episode in env.episodes:
        key = (scene_key(episode.scene_id), str(episode.episode_id))
        if key in by_key:
            found[key] = episode
    missing = [key for key in by_key if key not in found]
    if missing:
        print("missing from dataset:", ", ".join(f"{scene}/{eid}" for scene, eid in missing))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    ordered = sorted(found, key=lambda key: (key[0], int(key[1]) if key[1].isdigit() else key[1]))
    for index, key in enumerate(ordered, start=1):
        scene, episode_id = key
        out_dir = output / episode_dirname(scene, episode_id)
        if _already_collected(out_dir) and not args.overwrite:
            print(f"[{index}/{len(ordered)}] {scene} {episode_id} skip, already collected", flush=True)
            summaries.append({"scene_id": scene, "episode_id": episode_id, "frames": None, "skipped": True, "dir": str(out_dir)})
            continue
        print(f"[{index}/{len(ordered)}] {scene} {episode_id}", flush=True)
        summary = collect_one(env, found[key], out_dir, by_key[key]["instruction"], args.max_steps)
        summary["skipped"] = False
        summaries.append(summary)
        print(
            f"  frames={summary['frames']} reached_end={summary['reached']}",
            flush=True,
        )
    env.close()
    report = {
        "parsed": str(args.parsed),
        "split": args.split,
        "collected": summaries,
        "missing": [{"scene_id": scene, "episode_id": episode_id} for scene, episode_id in missing],
    }
    (output / "collect_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    done = sum(1 for row in summaries if not row.get("skipped"))
    print(f"collected {done}, skipped {len(summaries) - done}, missing {len(missing)}", flush=True)
    if missing:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集解析结果对应的 R2R 参考路径 RGB-D")
    parser.add_argument("--parsed", type=Path, required=True, help="parsed_instructions.json，只用其中的 scene_id 和 episode_id")
    parser.add_argument("--output", type=Path, required=True, help="轨迹根目录，下面每个 episode 一个文件夹")
    parser.add_argument("--config", default="config/vln_r2r.yaml")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--data-path", default=None, help="默认 data/datasets/r2r/{split}/{split}.json.gz")
    parser.add_argument("--scenes-dir", default=None, help="默认用配置里的 data/scene_datasets/")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-list", action="store_true", help="只打印要采集的 episode，不启动 Habitat")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    wanted = load_wanted(args.parsed)
    if args.dry_list:
        for row in wanted:
            print(f"{episode_dirname(row['scene'], row['episode_id'])}  {row['instruction'][:80]}")
        print(f"{len(wanted)} episodes, {len({row['scene'] for row in wanted})} scenes")
        return
    collect(args)


if __name__ == "__main__":
    main()
