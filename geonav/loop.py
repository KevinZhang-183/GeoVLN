"""Habitat loop. Imports the simulator only when an episode is actually run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from geonav._bootstrap import ROOT, install_habitat
from geonav.episodes import episode_key, load_rows
from geonav.executor import Executor
from geonav.policy import Policy
from geonav.sam3_client import Sam3Client
from geonav.world import World


def run_episodes(args):
    rows = load_rows(args.parsed)
    if args.limit:
        rows = rows[: args.limit]
    by_key = {episode_key(row["scene_id"], row["episode_id"]): row for row in rows}
    install_habitat()
    env = _make_env(args, sorted({key[0] for key in by_key}))
    try:
        found = {}
        for episode in env.episodes:
            key = episode_key(episode.scene_id, episode.episode_id)
            if key in by_key:
                found[key] = episode
        missing = [key for key in by_key if key not in found]
        if missing:
            print("missing from dataset:", ", ".join(f"{scene}/{eid}" for scene, eid in missing), flush=True)
        if not found:
            raise SystemExit("数据集里没有对上解析结果。检查 --data-path、--split 和 --scenes-dir。")
        client = _sam3(args)
        summary = []
        ordered = sorted(found, key=lambda key: (key[0], _episode_order(key[1])))
        for index, key in enumerate(ordered, start=1):
            row = by_key[key]
            print(f"[{index}/{len(ordered)}] {key[0]} {key[1]} mode={args.mode}", flush=True)
            result = run_one(env, found[key], row, args, client)
            summary.append(result)
            print(
                f"  steps={result['steps']} stop={result['stop_reason']} "
                f"success={result['metrics'].get('success')} spl={result['metrics'].get('spl')} "
                f"ne={result['metrics'].get('distance_to_goal')} sam3={result['sam3_calls']}",
                flush=True,
            )
        out = Path(args.output) / args.mode
        out.mkdir(parents=True, exist_ok=True)
        report = {"mode": args.mode, "episodes": summary, "means": _means(summary)}
        (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        means = report["means"]
        print(
            f"episodes={len(summary)} SR={_fmt(means.get('success'))} SPL={_fmt(means.get('spl'))} "
            f"NE={_fmt(means.get('distance_to_goal'))} OSR={_fmt(means.get('oracle_success'))}",
            flush=True,
        )
        print(f"wrote {out / 'summary.json'}", flush=True)
    finally:
        env.close()


def run_one(env, episode, row, args, client):
    actions = _action_ids()
    env.current_episode = episode
    observation = env.reset()
    goal = _goal(episode)
    parsed = row.get("parsed")
    mode = args.mode
    if mode == "landmark" and not parsed:
        print(f"  {episode.episode_id} 没有解析结果，这一条按 explore 走", flush=True)
        mode = "explore"
    policy = Policy(mode, parsed, goal, allow_look=not args.no_look)
    world = World(sam3=client)
    executor = Executor()
    scene, episode_id = episode_key(episode.scene_id, episode.episode_id)
    out_dir = Path(args.output) / args.mode / f"{scene}_{episode_id}"
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    spin_left = 0 if args.mode == "goal" or args.no_look else 24
    spin_done = False
    stop_reason = "max_steps"
    last_reason = None
    while not env.episode_over and step < args.max_steps:
        pose = _pose(env.sim)
        rgb = _rgb(observation)
        depth = _depth_meters(observation["depth"])
        world.update(depth, pose, rgb, step, policy.prompts())
        if world.new_traversals and mode != "goal" and not args.no_look and spin_left == 0:
            spin_left = 24
            policy.allow_look = True
        if spin_left == 0:
            if spin_done:
                policy.allow_look = False
                spin_done = False
            action, decision_reason, stop = _decide(policy, executor, world, pose, step)
            if decision_reason == "look":
                policy.allow_look = False
                spin_left = 24
                action = "turn_left"
            elif _made_progress(decision_reason):
                policy.allow_look = True
            if stop:
                stop_reason = decision_reason
                _save_vis(vis_dir, step, rgb, world, executor, args.vis_stride, force=True)
                if not env.episode_over:
                    env.step(actions["stop"])
                break
        else:
            action = "turn_left"
            decision_reason = "look"
        if spin_left > 0:
            spin_left -= 1
            spin_done = spin_left == 0
        if decision_reason != last_reason:
            print(f"  step {step:03d} {decision_reason}", flush=True)
            last_reason = decision_reason
        _save_vis(vis_dir, step, rgb, world, executor, args.vis_stride, force=False)
        if env.episode_over:
            break
        observation = env.step(actions[action])
        step += 1
    else:
        if not env.episode_over:
            env.step(actions["stop"])
            stop_reason = "max_steps"
    metrics = _plain(env.get_metrics())
    payload = {
        "scene_id": scene,
        "episode_id": episode_id,
        "mode": mode,
        "instruction": row.get("instruction") or "",
        "end_landmark": (parsed or {}).get("end_landmark"),
        "steps": step,
        "stop_reason": stop_reason,
        "sam3_calls": world.sam3_calls,
        "sam3_failures": world.sam3_failures,
        "subtask_index": policy.index,
        "metrics": metrics,
        "portals": [
            {"id": track.id, "state": track.state, "center": [round(float(track.center[0]), 3), round(float(track.center[1]), 3)]}
            for track in world.portals.tracks
        ],
        "instances": [item.to_dict(world.semantic.config.confirm_hits) for item in world.semantic.instances],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episode.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    final = world.render(executor.path, executor.waypoint)
    if final is not None:
        final.save(out_dir / "map.jpg")
    return payload


def _made_progress(reason):
    if reason in ("goal", "backtrack"):
        return True
    return len(reason) > 1 and reason[0] in "PLFN" and reason[1].isdigit()


def _decide(policy, executor, world, pose, step):
    action = None
    decision = None
    for _ in range(4):
        if executor.abandoned and executor.waypoint is not None:
            policy.abandon(executor.waypoint, world)
            executor.clear()
        if executor.reached and executor.waypoint is not None:
            policy.note_reached(executor.waypoint)
            executor.clear()
        decision = policy.decide(world, pose, step)
        if decision.kind != "waypoint":
            break
        executor.set_waypoint(decision.waypoint)
        action = executor.act(pose, world.grid)
        if action is None and (executor.reached or executor.abandoned):
            continue
        break
    if decision is None:
        return "turn_left", "idle", False
    if decision.kind == "stop":
        return "stop", decision.reason, True
    if decision.kind == "look":
        return "turn_left", "look", False
    if decision.kind == "action":
        return decision.action, decision.reason, False
    if action is None:
        action = "turn_left"
    return action, decision.reason, False


def _sam3(args):
    if args.no_sam3:
        return None
    client = Sam3Client(args.sam3_url)
    if client.healthy():
        print(f"SAM3 {args.sam3_url}", flush=True)
        return client
    message = (
        "SAM3 服务没开。在 sam3 环境里先启动:\n"
        "  conda activate sam3\n"
        "  python -m geonav.sam3_server --checkpoint /root/data1/models/sam3_weights/sam3.pt --port 8001"
    )
    if args.mode == "landmark":
        raise SystemExit(message)
    print(message, flush=True)
    print("这一轮不更新地标，继续跑。", flush=True)
    return None


def _make_env(args, scenes):
    import os

    import habitat
    from habitat.config import read_write
    from habitat_baselines.config.default import get_config as get_habitat_config

    import habitat_extensions.measures  # noqa: F401

    config = get_habitat_config(str(args.config))
    with read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.dataset.data_path = str(args.data_path)
        config.habitat.dataset.scenes_dir = str(args.scenes_dir)
        config.habitat.dataset.content_scenes = scenes
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.environment.max_episode_steps = args.max_steps
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("HABITAT_SIM_LOG", "quiet")
    return habitat.Env(config=config)


def _action_ids():
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    names = ("stop", "move_forward", "turn_left", "turn_right")
    mapping = {}
    for name in names:
        if hasattr(HabitatSimActions, name):
            mapping[name] = int(getattr(HabitatSimActions, name))
        else:
            mapping[name] = int(getattr(HabitatSimActions, name.upper()))
    return mapping


def _pose(sim):
    from extract_portals import habitat_agent_matrix

    state = sim.get_agent_state()
    rotation = state.rotation
    if hasattr(rotation, "w"):
        quat = [float(rotation.w), float(rotation.x), float(rotation.y), float(rotation.z)]
    else:
        import quaternion

        quat = [float(value) for value in quaternion.as_float_array(rotation)]
    position = [float(value) for value in state.position]
    return habitat_agent_matrix(position, quat)


def _goal(episode):
    path = list(getattr(episode, "reference_path", None) or [])
    if path:
        return np.asarray(path[-1], dtype=np.float64)
    goals = list(getattr(episode, "goals", None) or [])
    if not goals:
        return None
    position = goals[0].position if hasattr(goals[0], "position") else goals[0]
    return np.asarray(position, dtype=np.float64)


def _rgb(observation):
    image = np.asarray(observation["rgb"])
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _depth_meters(depth, depth_max=10.0):
    depth = np.squeeze(np.asarray(depth, dtype=np.float32))
    finite = depth[np.isfinite(depth)]
    if finite.size and float(np.max(finite)) <= 1.5:
        depth = depth * depth_max
    return depth


def _save_vis(vis_dir, step, rgb, world, executor, stride, force):
    if not force and step % max(1, stride) != 0:
        return
    rendered = world.render(executor.path, executor.waypoint)
    if rendered is None:
        return
    left = Image.fromarray(_rgb({"rgb": rgb})).convert("RGB").resize((640, 480))
    right = rendered.convert("RGB").resize((640, 480))
    canvas = Image.new("RGB", (1280, 480))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (640, 0))
    canvas.save(vis_dir / f"{step:04d}.jpg", quality=90)


def _plain(metrics):
    out = {}
    for key, value in dict(metrics).items():
        if isinstance(value, (dict, list)):
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _means(rows):
    keys = ("success", "spl", "distance_to_goal", "oracle_success")
    means = {}
    for key in keys:
        values = [row["metrics"][key] for row in rows if key in row.get("metrics", {})]
        if values:
            means[key] = sum(values) / len(values)
    return means


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def _episode_order(episode_id):
    text = str(episode_id)
    return (0, int(text)) if text.isdigit() else (1, text)


def _config_path():
    return ROOT / "config" / "vln_r2r.yaml"
