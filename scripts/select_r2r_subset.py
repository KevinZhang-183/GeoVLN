#!/usr/bin/env python3
"""从 StreamVLN 的 result.json 里抽出固定的评测子集，并给出这批轨迹的基线指标。

result.json 是每条 episode 一行。文件末尾可能还有一行全量汇总（sucs_all / spls_all），
那一行没有 scene_id，会被跳过。

抽样（seed 固定，重复运行得到同一份名单）：

1. 按 scene_id 的轨迹数比例分配名额（最大余数法）。轨迹总数不少于场景数时，每个场景至少 1 条。
2. 场景内部按基线 success / fail 的比例分配，使子集成功率贴近该场景的全量成功率。
3. 同一成败组内按 steps 从小到大等间隔抽取。steps 是智能体走的步数，不是数据集里的最短路径长度；
   这里只用它把短轨迹和长轨迹都留在子集里。

默认读集群上这次 R2R val-unseen 的结果，写出同目录的 subset_300.json。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_RESULT = "/root/data1/StreamVLN/results/r2r_val_unseen_v1_3/result.json"


def load_episodes(path: Path) -> tuple[list[dict], int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"结果文件是空的: {path}")

    if text.startswith("["):
        records = json.loads(text)
    else:
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} 不是合法 JSON: {exc}") from exc

    episodes: list[dict] = []
    skipped = 0
    for record in records:
        if not isinstance(record, dict) or "scene_id" not in record or "episode_id" not in record:
            skipped += 1
            continue
        episodes.append(record)

    deduped: dict[tuple[str, str], dict] = {}
    for record in episodes:
        deduped[(str(record["scene_id"]), str(record["episode_id"]))] = record
    duplicate_count = len(episodes) - len(deduped)
    if duplicate_count:
        print(f"warning: 去掉重复 episode {duplicate_count} 条，保留每个 (scene_id, episode_id) 的最后一条")
    return list(deduped.values()), skipped


def mean(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(float(row[key]) for row in rows) / len(rows)


def metrics(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "ne": mean(rows, "ne"),
        "os": mean(rows, "os"),
        "sr": mean(rows, "success"),
        "spl": mean(rows, "spl"),
    }


def format_metrics(title: str, stats: dict) -> str:
    return (
        f"{title:<8} n={stats['n']:<5}  "
        f"NE={stats['ne']:.2f}  "
        f"OS={100 * stats['os']:.1f}  "
        f"SR={100 * stats['sr']:.1f}  "
        f"SPL={100 * stats['spl']:.1f}"
    )


def largest_remainder(counts: dict[str, int], n: int) -> dict[str, int]:
    total = sum(counts.values())
    if n > total:
        raise SystemExit(f"要抽 {n} 条，但去重后只有 {total} 条")
    if n == total:
        return dict(counts)

    raw = {scene: n * count / total for scene, count in counts.items()}
    quota = {scene: int(raw[scene]) for scene in raw}
    leftover = n - sum(quota.values())
    order = sorted(raw, key=lambda scene: (-(raw[scene] - quota[scene]), scene))
    for scene in order[:leftover]:
        quota[scene] += 1

    if n >= len(counts):
        missing = sorted(scene for scene, taken in quota.items() if taken == 0)
        for scene in missing:
            donors = [name for name, taken in quota.items() if taken > 1]
            donor = max(donors, key=lambda name: (quota[name], name))
            quota[donor] -= 1
            quota[scene] += 1
    return quota


def is_success(row: dict) -> bool:
    return float(row["success"]) >= 0.5


def split_quota(k: int, n_pos: int, n_neg: int) -> tuple[int, int]:
    if k == 0:
        return 0, 0
    raw_pos = k * n_pos / (n_pos + n_neg)
    k_pos = min(n_pos, int(round(raw_pos)))
    k_neg = k - k_pos
    if k_neg > n_neg:
        k_pos += k_neg - n_neg
        k_neg = n_neg
    if k_pos > n_pos:
        k_neg += k_pos - n_pos
        k_pos = n_pos
    return k_pos, k_neg


def systematic_sample(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (int(row.get("steps") or 0), str(row["episode_id"])))
    if k >= len(ordered):
        return ordered
    step = len(ordered) / k
    start = rng.random() * step
    picked = []
    for i in range(k):
        index = min(int(start + i * step), len(ordered) - 1)
        picked.append(ordered[index])
    return picked


def select_subset(episodes: list[dict], n: int, seed: int) -> tuple[list[dict], dict[str, int]]:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in episodes:
        by_scene[str(row["scene_id"])].append(row)

    quota = largest_remainder({scene: len(rows) for scene, rows in by_scene.items()}, n)
    rng = random.Random(seed)
    selected: list[dict] = []
    for scene in sorted(by_scene):
        rows = by_scene[scene]
        positives = [row for row in rows if is_success(row)]
        negatives = [row for row in rows if not is_success(row)]
        k_pos, k_neg = split_quota(quota[scene], len(positives), len(negatives))
        selected.extend(systematic_sample(positives, k_pos, rng))
        selected.extend(systematic_sample(negatives, k_neg, rng))

    selected.sort(key=lambda row: (str(row["scene_id"]), int(row["episode_id"]) if str(row["episode_id"]).isdigit() else str(row["episode_id"])))
    if len(selected) != n:
        raise SystemExit(f"内部错误: 抽到 {len(selected)} 条，目标是 {n}")
    ids = [(str(row["scene_id"]), str(row["episode_id"])) for row in selected]
    if len(ids) != len(set(ids)):
        raise SystemExit("内部错误: 子集里出现重复 episode")
    return selected, quota


def scene_report(episodes: list[dict], selected: list[dict], quota: dict[str, int]) -> list[dict]:
    full_by_scene: dict[str, list[dict]] = defaultdict(list)
    sub_by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in episodes:
        full_by_scene[str(row["scene_id"])].append(row)
    for row in selected:
        sub_by_scene[str(row["scene_id"])].append(row)

    report = []
    for scene in sorted(full_by_scene):
        full_rows = full_by_scene[scene]
        sub_rows = sub_by_scene[scene]
        report.append(
            {
                "scene_id": scene,
                "full": len(full_rows),
                "selected": quota[scene],
                "full_sr": mean(full_rows, "success"),
                "subset_sr": mean(sub_rows, "success") if sub_rows else None,
            }
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 R2R result.json 抽取固定评测子集并计算基线指标")
    parser.add_argument("--result", type=Path, default=Path(DEFAULT_RESULT), help="每条 episode 一行的 result.json")
    parser.add_argument("--num", type=int, default=300, help="子集轨迹数")
    parser.add_argument("--seed", type=int, default=0, help="抽样随机种子，换种子会得到另一份固定名单")
    parser.add_argument("--output", type=Path, default=None, help="默认写到 result.json 同目录的 subset_300.json")
    args = parser.parse_args()
    if args.num <= 0:
        raise SystemExit("--num 必须是正整数")
    if args.output is None:
        args.output = args.result.with_name(f"subset_{args.num}.json")
    return args


def main() -> None:
    args = parse_args()
    if not args.result.is_file():
        raise SystemExit(f"找不到结果文件: {args.result}")

    episodes, skipped = load_episodes(args.result)
    if not episodes:
        raise SystemExit("没有读到带 scene_id 和 episode_id 的轨迹")

    selected, quota = select_subset(episodes, args.num, args.seed)
    full_stats = metrics(episodes)
    subset_stats = metrics(selected)
    scenes = scene_report(episodes, selected, quota)

    payload = {
        "num": args.num,
        "seed": args.seed,
        "source": str(args.result),
        "skipped_non_episode_records": skipped,
        "method": (
            "按 scene_id 比例分配；场景内按 success/fail 比例分配；"
            "同一组内按 steps 等间隔抽取"
        ),
        "full_metrics": full_stats,
        "subset_metrics": subset_stats,
        "scenes": scenes,
        "episodes": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(format_metrics("full", full_stats))
    print(format_metrics("subset", subset_stats))
    if skipped:
        print(f"跳过非 episode 记录 {skipped} 条（通常是文件末尾的全量汇总）")
    print(f"场景数 {len(scenes)}")
    for scene in scenes:
        subset_sr = "  - " if scene["subset_sr"] is None else f"{100 * scene['subset_sr']:5.1f}"
        print(
            f"  {scene['scene_id']}  "
            f"full {scene['full']:4d}  selected {scene['selected']:3d}  "
            f"SR {100 * scene['full_sr']:5.1f} -> {subset_sr}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
