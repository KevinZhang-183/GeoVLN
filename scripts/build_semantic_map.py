"""Run SAM3 along an expert trajectory and fuse the masks into a semantic map.

The portal map is built separately by build_portal_map.py. This script does
not read it. Both write JSON in the episode folder; combining them comes later.

Each episode folder needs rgb/*.jpg, depth/*.npy and meta.json. Prompts come
from parsed_instructions.json, or from --prompts.

    python scripts/build_semantic_map.py \
        --episode /path/to/2azQ1b91cZZ_140 \
        --parsed /path/to/parsed_instructions.json \
        --stride 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_portal_map import _lookup, _stem, load_meta  # noqa: E402
from extract_portals import PortalConfig, habitat_agent_matrix, load_depth  # noqa: E402
from semantic_map import (  # noqa: E402
    SemanticConfig,
    SemanticMap,
    _match_episode,
    mask_points,
    prompts_from_parsed,
)

PALETTE = (
    (0, 180, 70),
    (220, 120, 0),
    (150, 40, 180),
    (0, 110, 210),
    (190, 40, 40),
    (0, 150, 150),
)


DEFAULT_SAM3_WEIGHTS = os.environ.get("SAM3_WEIGHTS") or "/root/data1/models/sam3_weights"
_WEIGHT_NAMES = ("sam3.pt", "sam3.1_multiplex.pt", "model.pt", "checkpoint.pt")


def resolve_checkpoint(path):
    """文件直接用。目录里优先 sam3.pt，只有一个 .pt/.pth 时用那一个。"""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        for name in _WEIGHT_NAMES:
            chosen = candidate / name
            if chosen.is_file():
                return chosen
        weights = sorted(candidate.glob("*.pt")) + sorted(candidate.glob("*.pth"))
        if len(weights) == 1:
            return weights[0]
        if weights:
            names = ", ".join(item.name for item in weights)
            raise SystemExit(f"{candidate} 里有多个权重，请用 --checkpoint 指定: {names}")
    raise SystemExit(f"找不到 SAM3 权重: {candidate}")


def load_processor(checkpoint):
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError as exc:
        raise SystemExit(
            "没有装上 sam3。在服务器上：pip install -e 已克隆的 sam3 仓库。"
            f"\n原始错误: {exc}"
        ) from exc
    checkpoint_path = str(resolve_checkpoint(checkpoint))
    print(f"SAM3 checkpoint: {checkpoint_path}", flush=True)
    try:
        model = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)
    except TypeError:
        model = build_sam3_image_model(checkpoint_path=checkpoint_path)
    return Sam3Processor(model)


def _numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
        # NumPy has no bfloat16. Autocast returns scores in that dtype.
        if value.is_floating_point() and value.dtype != torch.float32:
            value = value.float()
        value = value.cpu().numpy()
    return np.asarray(value)


def segment_prompts(processor, image, prompts):
    """Encode the image once, then run each text prompt. Yields (prompt, masks, scores)."""
    # The fused ViT MLP writes bfloat16 activations while Linear weights stay float32.
    # Autocast casts those weights for the matmul, which is how Meta runs SAM3.
    encoded = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        for prompt in prompts:
            output = processor.set_text_prompt(state=state, prompt=prompt)
            encoded.append((prompt, _numpy(output.get("masks")), _numpy(output.get("scores"))))
    for prompt, masks, scores in encoded:
        if masks is None:
            continue
        if masks.ndim == 4:
            masks = masks[:, 0]
        if masks.ndim == 2:
            masks = masks[None]
        if scores is not None:
            scores = np.reshape(scores, -1)
        yield prompt, masks.astype(bool), scores


def _blend(canvas, mask, color):
    if mask.shape[:2] != canvas.shape[:2]:
        return
    canvas[mask] = (0.55 * canvas[mask] + 0.45 * np.array(color)).astype(np.uint8)


def _prompts_for(episode_dir, parsed_path, scene_id, episode_id, explicit):
    if explicit:
        return [item.strip().lower() for item in explicit.split(",") if item.strip()]
    if parsed_path is None:
        raise SystemExit("pass --parsed or --prompts")
    document = json.loads(Path(parsed_path).read_text(encoding="utf-8"))
    episodes = document["episodes"] if isinstance(document, dict) else document
    row = _match_episode(episodes, episode_dir, scene_id, episode_id)
    parsed = row.get("parsed") or {}
    prompts = prompts_from_parsed(parsed)
    if not prompts:
        raise SystemExit(f"{episode_dir}: parsed episode {row.get('episode_id')} has no object prompts")
    print(f"prompts from {row.get('scene_id')} / {row.get('episode_id')}: {prompts}", flush=True)
    return prompts


def process_episode(episode_dir, out_name, stride, prompts, processor, config, portal_config, save_frames):
    rgb_dir = os.path.join(episode_dir, "rgb")
    depth_dir = os.path.join(episode_dir, "depth")
    out_dir = os.path.join(episode_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(rgb_dir) if name.endswith(".jpg"))
    meta = load_meta(episode_dir, names)
    smap = SemanticMap(config)
    log = [f"# {os.path.basename(episode_dir)}", "prompts: " + ", ".join(prompts)]
    kept_frames = 0
    for name in names[::stride]:
        stem = _stem(name)
        depth_path = os.path.join(depth_dir, stem + ".npy")
        entry = _lookup(meta, stem)
        if not os.path.exists(depth_path) or entry is None:
            print(f"  skip {name}: missing depth or meta", flush=True)
            continue
        pose = habitat_agent_matrix(entry["position"], entry["rotation_wxyz"])
        depth = load_depth(depth_path, portal_config)
        smap.note_agent(pose)
        image = Image.open(os.path.join(rgb_dir, name)).convert("RGB")
        canvas = np.array(image)
        counts = {"new": 0, "updated": 0, "skip": 0}
        for index, (prompt, masks, scores) in enumerate(segment_prompts(processor, image, prompts)):
            color = PALETTE[index % len(PALETTE)]
            for mask_index, mask in enumerate(masks):
                score = 1.0 if scores is None or len(scores) <= mask_index else float(scores[mask_index])
                points = mask_points(mask, depth, config)
                kind = smap.observe_camera(prompt, points, pose, score, name)
                counts[kind] += 1
                if kind != "skip" and save_frames:
                    _blend(canvas, mask, color)
        kept_frames += 1
        log.append(f"{name} new={counts['new']} updated={counts['updated']}")
        print(f"  {name} new={counts['new']} updated={counts['updated']}", flush=True)
        if save_frames:
            Image.fromarray(canvas).save(os.path.join(out_dir, name))

    payload = smap.to_dict(prompts)
    with open(os.path.join(out_dir, "semantic_map.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    smap.render().save(os.path.join(out_dir, "map.jpg"))
    log.append("")
    log.extend(smap.describe())
    with open(os.path.join(out_dir, "describe.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(log) + "\n")
    counts = payload["counts"]
    summary = (
        f"{os.path.basename(episode_dir)} frames={kept_frames} prompts={len(prompts)} "
        f"instances={len(payload['instances'])} confirmed={counts['confirmed']} candidate={counts['candidate']}"
    )
    print(summary, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="沿专家轨迹用 SAM3 建语义地图，不读门洞地图")
    parser.add_argument("--frames", help="一批轨迹目录，每个子目录都有 rgb/、depth/ 和 meta.json")
    parser.add_argument("--episode", help="一条轨迹目录")
    parser.add_argument("--parsed", help="parsed_instructions.json。和 --prompts 二选一")
    parser.add_argument("--scene-id", help="文件夹名对不上解析结果时，用来指定场景")
    parser.add_argument("--episode-id", help="配合 --scene-id")
    parser.add_argument("--prompts", help="逗号分隔的提示词，给定后不再读解析结果")
    parser.add_argument("--out-name", default="semantic_map")
    parser.add_argument("--stride", type=int, default=4, help="每隔多少帧跑一次 SAM3，默认 4")
    parser.add_argument("--min-score", type=float, default=0.45)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_SAM3_WEIGHTS,
        help="SAM3 权重文件或目录，默认 /root/data1/models/sam3_weights，也可用 SAM3_WEIGHTS",
    )
    parser.add_argument("--no-frames", action="store_true", help="不写叠了掩码的 RGB，仍写 map.jpg 和 json")
    args = parser.parse_args()
    if not args.frames and not args.episode:
        parser.error("pass --frames or --episode")
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
    if not episodes:
        raise SystemExit("没有找到带 rgb/、depth/ 和 meta.json 的轨迹")
    config = SemanticConfig(min_score=args.min_score)
    portal_config = PortalConfig()
    print(f"loading SAM3, episodes={len(episodes)} stride={args.stride}", flush=True)
    processor = load_processor(args.checkpoint)
    summaries = []
    for episode in episodes:
        prompts = _prompts_for(episode, args.parsed, args.scene_id, args.episode_id, args.prompts)
        summaries.append(
            process_episode(
                episode, args.out_name, args.stride, prompts, processor, config, portal_config, not args.no_frames
            )
        )
    with open(os.path.join(summary_dir, args.out_name + "_summary.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(summaries) + "\n")


if __name__ == "__main__":
    main()
