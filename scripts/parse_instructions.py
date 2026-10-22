#!/usr/bin/env python3
"""把 R2R 指令拆成有序地标链，缓存成 parsed_instructions.json。

子集来自 select_r2r_subset.py。result.json 里已经有 episode_instruction，
所以这一步不需要再打开 Habitat 数据集。没有这条字段时，用 --dataset 按
scene_id + episode_id 去 json/json.gz 里补。

动作只有 go_to / turn_left / turn_right / stop 四类。地标照抄指令里的物体、门或房间，
房间名不拒收，只打标记，留给下游换成门洞或交给 VLM。

在服务器上，先让 vLLM 提供 OpenAI 兼容接口，再：

    export VLM_BASE_URL=http://127.0.0.1:8000/v1
    export VLM_MODEL=Qwen2.5-VL-7B-Instruct
    export VLM_API_KEY=EMPTY
    python scripts/parse_instructions.py \\
        --subset /root/data1/StreamVLN/results/r2r_val_unseen_v1_3/subset_100.json \\
        --output /root/data1/StreamVLN/results/r2r_val_unseen_v1_3/parsed_instructions.json \\
        --review /root/data1/StreamVLN/results/r2r_val_unseen_v1_3/parsed_instructions_review.md
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_r2r_subset import load_episodes  # noqa: E402
from vlm_client import VLMClient, extract_json  # noqa: E402


ACTIONS = ("go_to", "turn_left", "turn_right", "stop")
ACTION_ALIAS = {
    "go": "go_to",
    "goto": "go_to",
    "walk": "go_to",
    "walk_to": "go_to",
    "move_to": "go_to",
    "approach": "go_to",
    "find": "go_to",
    "exit": "go_to",
    "leave": "go_to",
    "enter": "go_to",
    "pass": "go_to",
    "go_past": "go_to",
    "walk_past": "go_to",
    "go_through": "go_to",
    "go_up": "go_to",
    "go_down": "go_to",
    "climb": "go_to",
    "follow": "go_to",
    "left": "turn_left",
    "right": "turn_right",
    "wait": "stop",
    "halt": "stop",
    "stop_at": "stop",
}
TURN_AROUND = {"turn_around", "turn_back", "u_turn"}
ARTICLES = {"the", "a", "an"}
DIRECTION_ONLY = {
    "left",
    "right",
    "forward",
    "straight",
    "ahead",
    "back",
    "backward",
    "around",
    "up",
    "down",
}
ROOM_WORDS = {
    "kitchen",
    "bedroom",
    "bathroom",
    "hallway",
    "hall",
    "corridor",
    "living",
    "dining",
    "office",
    "lobby",
    "garage",
    "balcony",
    "closet",
    "laundry",
    "foyer",
    "entryway",
    "room",
    "den",
    "study",
    "patio",
    "porch",
    "lounge",
}
OBJECT_WORDS = {
    "door",
    "doors",
    "doorway",
    "doorways",
    "stair",
    "stairs",
    "staircase",
    "stairway",
    "table",
    "chair",
    "chairs",
    "oven",
    "ovens",
    "plant",
    "plants",
    "pot",
    "pots",
    "flower",
    "flowers",
    "fireplace",
    "refrigerator",
    "fridge",
    "counter",
    "rug",
    "window",
    "picture",
    "sofa",
    "couch",
    "bed",
    "sink",
    "toilet",
    "cabinet",
    "lamp",
    "mirror",
    "railing",
}

PROMPT = """Split the navigation instruction into steps, in order.

Each step is {"action": ..., "landmark": ...}.
action is one of: go_to, turn_left, turn_right, stop.
landmark is the object, door or room the instruction names, in 1 to 3 words. turn_left and turn_right have landmark "".
"Turn around" is two turn_right steps. The last step is stop, with the place to stop as its landmark.

Reply with JSON only: {"steps": [...]}

Instruction: Walk out of the bedroom, turn left, go past the stairs, and stop next to the potted plant.
{"steps": [{"action": "go_to", "landmark": "bedroom door"}, {"action": "turn_left", "landmark": ""}, {"action": "go_to", "landmark": "stairs"}, {"action": "stop", "landmark": "potted plant"}]}

Instruction: Go down the hallway, turn right into the kitchen, and wait in front of the refrigerator.
{"steps": [{"action": "go_to", "landmark": "hallway"}, {"action": "turn_right", "landmark": ""}, {"action": "go_to", "landmark": "kitchen"}, {"action": "stop", "landmark": "refrigerator"}]}

Instruction: {instruction}
"""


def instruction_of(row: dict) -> str:
    for key in ("episode_instruction", "instruction_text", "instruction"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            text = value.get("instruction_text") or value.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return ""


def scene_key(scene_id: str) -> str:
    parts = str(scene_id).replace("\\", "/").split("/")
    if parts[-1].endswith(".glb") and len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def load_jobs(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
            return payload["episodes"]
    episodes, _skipped = load_episodes(path)
    return episodes


def load_dataset_instructions(path: Path) -> dict[tuple[str, str], str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload["episodes"] if isinstance(payload, dict) else payload
    found = {}
    for episode in episodes:
        text = instruction_of(episode)
        if not text:
            continue
        found[(scene_key(episode.get("scene_id", "")), str(episode.get("episode_id")))] = text
    return found


def build_prompt(instruction: str, errors: list[str] | None = None) -> str:
    prompt = PROMPT.replace("{instruction}", instruction)
    if errors:
        prompt += "\nThe previous reply was rejected:\n- " + "\n- ".join(errors)
        prompt += "\nReply with the corrected JSON only."
    return prompt


def _landmark(value) -> str:
    words = str(value or "").lower().replace(",", " ").strip().strip(".").split()
    while words and words[0] in ARTICLES:
        words = words[1:]
    phrase = " ".join(words)
    return "" if phrase in DIRECTION_ONLY else phrase


def _place_only(landmark: str) -> bool:
    """房间名，或房间名再加方位。没有门、楼梯、家具这类可分割物体。"""
    words = [word for word in landmark.split() if word not in ARTICLES | {"of"}]
    if not words or any(word in OBJECT_WORDS for word in words):
        return False
    return any(word in ROOM_WORDS for word in words)


def _action(value, raw_landmark) -> str | None:
    """归到四类动作。turn_around 单独返回，由调用方展开成两次右转。"""
    name = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if name in ACTIONS or name in TURN_AROUND:
        return name
    if name == "turn":
        side = str(raw_landmark or "").strip().lower()
        return {"left": "turn_left", "right": "turn_right", "around": "turn_around"}.get(side)
    return ACTION_ALIAS.get(name)


def validate(parsed: dict) -> tuple[dict | None, list[str], list[str]]:
    """返回 (规范化结果, 硬错误, 软标记)。硬错误非空时结果为 None。

    能修的都在这里修，只在拿不到动作或一个地标都没有时才判失败。
    """
    if not isinstance(parsed, dict):
        return None, ["reply is not a JSON object"], []
    steps = parsed.get("steps", parsed.get("subtasks"))
    if not isinstance(steps, list) or not steps:
        return None, ['"steps" must be a non-empty list'], []

    errors = []
    flags = []
    out = []
    for index, item in enumerate(steps, start=1):
        if not isinstance(item, dict):
            errors.append(f"step {index} is not an object")
            continue
        raw_landmark = item.get("landmark", "")
        action = _action(item.get("action"), raw_landmark)
        landmark = _landmark(raw_landmark)
        if action is None:
            errors.append(f"step {index} action {item.get('action')!r} is not one of {', '.join(ACTIONS)}")
            continue
        if action in TURN_AROUND:
            out += [{"action": "turn_right", "landmark": ""}, {"action": "turn_right", "landmark": ""}]
            continue
        if action in ("turn_left", "turn_right"):
            if landmark:
                out.append({"action": "go_to", "landmark": landmark})
            out.append({"action": action, "landmark": ""})
            continue
        if action == "go_to" and not landmark:
            continue
        out.append({"action": action, "landmark": landmark})
    if errors:
        return None, errors, flags

    if not out or out[-1]["action"] != "stop":
        flags.append("added missing stop")
        out.append({"action": "stop", "landmark": ""})
    if not out[-1]["landmark"] and len(out) >= 2 and out[-2]["action"] == "go_to":
        out[-1]["landmark"] = out.pop(-2)["landmark"]

    deduped = []
    for step in out:
        if deduped and step["landmark"] and step["landmark"] == deduped[-1]["landmark"]:
            deduped[-1]["action"] = step["action"] if step["action"] == "stop" else deduped[-1]["action"]
            continue
        deduped.append(step)

    landmarks = []
    for step in deduped:
        if step["landmark"] and step["landmark"] not in landmarks:
            landmarks.append(step["landmark"])
    if not landmarks:
        return None, ["no step has a landmark; name the objects, doors or rooms from the instruction"], flags

    for index, step in enumerate(deduped, start=1):
        step["id"] = index
        if len(step["landmark"].split()) > 4:
            flags.append(f"step {index} landmark is long: {step['landmark']}")
        if _place_only(step["landmark"]):
            flags.append(f"step {index} landmark is a room: {step['landmark']}")
    subtasks = [{"id": step["id"], "action": step["action"], "landmark": step["landmark"]} for step in deduped]
    return {"subtasks": subtasks, "landmarks": landmarks, "end_landmark": landmarks[-1]}, [], flags


def parse_one(client: VLMClient, instruction: str, retries: int = 1) -> tuple[dict | None, list[str], str | None]:
    errors: list[str] | None = None
    last_raw = None
    for _ in range(retries + 1):
        raw = client.complete(build_prompt(instruction, errors), max_tokens=400, temperature=0.0, json_mode=True)
        last_raw = raw
        parsed = extract_json(raw or "")
        if parsed is None:
            errors = ["reply was not a JSON object"]
            continue
        normalized, hard, flags = validate(parsed)
        if normalized is not None:
            return normalized, flags, None
        errors = hard
    return None, [], (last_raw or "")[:2000]


def write_review(path: Path, rows: list[dict], review_n: int, seed: int) -> None:
    pool = [row for row in rows if row.get("instruction")]
    rng = random.Random(seed)
    picked = pool if len(pool) <= review_n else rng.sample(pool, review_n)
    picked.sort(key=lambda row: (str(row["scene_id"]), str(row["episode_id"])))
    lines = [
        "# 指令解析抽查",
        "",
        f"共 {len(rows)} 条，这里抽 {len(picked)} 条（seed={seed}）。",
        "看三件事：步骤顺序和转向对不对、地标有没有漏或编造、终点地标对不对。",
        "",
    ]
    for row in picked:
        parsed = row.get("parsed") or {}
        lines.append(f"## {row['scene_id']} / {row['episode_id']}")
        lines.append("")
        lines.append(row["instruction"])
        lines.append("")
        if not parsed:
            lines.append(f"解析失败：{row.get('error') or 'empty'}")
            lines.append("")
            continue
        lines.append(f"终点：{parsed['end_landmark']}")
        lines.append("地标：" + " -> ".join(parsed["landmarks"]))
        for item in parsed["subtasks"]:
            lines.append(f"{item['id']}. {item['action']} {item['landmark']}".rstrip())
        flags = row.get("flags") or []
        lines.append("标记：" + ("；".join(flags) if flags else "无"))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: list[dict]) -> str:
    parsed = [row for row in rows if row.get("parsed")]
    flagged = [row for row in parsed if row.get("flags")]
    room_end = sum(1 for row in parsed if _place_only(row["parsed"]["end_landmark"]))
    return (
        f"parsed {len(parsed)}/{len(rows)}  "
        f"flagged {len(flagged)}  room_end {room_end}  "
        f"failed {len(rows) - len(parsed)}"
    )


def run(args: argparse.Namespace) -> None:
    jobs = load_jobs(args.subset)
    dataset = load_dataset_instructions(args.dataset) if args.dataset else {}
    rows = []
    for job in jobs:
        text = instruction_of(job)
        if not text and dataset:
            text = dataset.get((scene_key(job.get("scene_id", "")), str(job.get("episode_id"))), "")
        if not text:
            continue
        rows.append(
            {
                "scene_id": job.get("scene_id"),
                "episode_id": str(job.get("episode_id")),
                "instruction": text,
            }
        )
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("没有读到带指令的 episode")

    done = {}
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        for row in previous.get("episodes", []):
            if row.get("parsed"):
                done[(str(row["scene_id"]), str(row["episode_id"]))] = row

    client = VLMClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
    client.warmup()
    finished = []
    for index, row in enumerate(rows, start=1):
        key = (str(row["scene_id"]), row["episode_id"])
        if key in done:
            finished.append(done[key])
            continue
        parsed, flags, raw = parse_one(client, row["instruction"], retries=args.retries)
        item = dict(row)
        item["parsed"] = parsed
        item["flags"] = flags
        item["error"] = None if parsed else "model did not return a valid landmark chain"
        if raw and not parsed:
            item["raw"] = raw
        finished.append(item)
        if parsed:
            print(
                f"[{index}/{len(rows)}] {row['scene_id']} {row['episode_id']} -> {parsed['end_landmark']}",
                flush=True,
            )
        else:
            print(f"[{index}/{len(rows)}] {row['scene_id']} {row['episode_id']} -> FAILED", flush=True)
        if index % 10 == 0 or index == len(rows):
            _dump(args, client, finished)

    _dump(args, client, finished)
    if args.review:
        write_review(args.review, finished, args.review_n, args.seed)
        print(f"wrote {args.review}")
    print(summarize(finished))
    print(f"wrote {args.output}")


def _dump(args: argparse.Namespace, client: VLMClient, rows: list[dict]) -> None:
    payload = {
        "model": client.model,
        "base_url": client.base_url,
        "source": str(args.subset),
        "summary": summarize(rows),
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def self_check() -> None:
    fenced = '说明如下：\n```json\n{"subtasks": [], "end_landmark": "chair"}\n```'
    assert extract_json(fenced)["end_landmark"] == "chair"
    assert extract_json("no json here") is None

    normalized, errors, flags = validate(
        {
            "steps": [
                {"action": "exit", "landmark": "the bedroom door"},
                {"action": "turn", "landmark": "left"},
                {"action": "walk", "landmark": "forward"},
                {"action": "turn right", "landmark": "stairs"},
                {"action": "stop", "landmark": "Potted plant."},
            ]
        }
    )
    assert not errors, errors
    assert [(s["action"], s["landmark"]) for s in normalized["subtasks"]] == [
        ("go_to", "bedroom door"),
        ("turn_left", ""),
        ("go_to", "stairs"),
        ("turn_right", ""),
        ("stop", "potted plant"),
    ]
    assert [s["id"] for s in normalized["subtasks"]] == [1, 2, 3, 4, 5]
    assert normalized["landmarks"] == ["bedroom door", "stairs", "potted plant"]
    assert normalized["end_landmark"] == "potted plant"
    assert not flags

    merged, errors, flags = validate(
        {"steps": [{"action": "go_to", "landmark": "kitchen"}, {"action": "turn_around", "landmark": ""},
                   {"action": "go_to", "landmark": "table"}]}
    )
    assert not errors, errors
    assert [s["action"] for s in merged["subtasks"]] == ["go_to", "turn_right", "turn_right", "stop"]
    assert merged["subtasks"][-1]["landmark"] == "table"
    assert any("missing stop" in flag for flag in flags)
    assert any("room" in flag for flag in flags)

    blocked, errors, _flags = validate({"steps": [{"action": "go_to", "landmark": "left"}]})
    assert blocked is None and any("no step has a landmark" in error for error in errors)
    unknown, errors, _flags = validate({"steps": [{"action": "fly", "landmark": "chair"}]})
    assert unknown is None and errors

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subset = root / "subset.json"
        subset.write_text(
            json.dumps(
                {
                    "episodes": [
                        {"scene_id": "abc", "episode_id": 3, "episode_instruction": "Stop by the plant."},
                        {"scene_id": "abc", "episode_id": 4, "episode_instruction": ""},
                    ]
                }
            ),
            encoding="utf-8",
        )
        jobs = [row for row in load_jobs(subset) if instruction_of(row)]
        assert len(jobs) == 1
        review = root / "review.md"
        write_review(
            review,
            [
                {
                    "scene_id": "abc",
                    "episode_id": "3",
                    "instruction": "Stop by the plant.",
                    "parsed": normalized,
                    "flags": flags,
                }
            ],
            50,
            0,
        )
        text = review.read_text(encoding="utf-8")
        assert "potted plant" in text
        assert "kitchen" not in text or "room" in text
    print("self-check ok")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 R2R 指令解析成有序地标链")
    parser.add_argument("--subset", type=Path, help="select_r2r_subset.py 写出的 subset json，或原始 result.json")
    parser.add_argument("--dataset", type=Path, help="可选，Habitat 的 R2R json/json.gz，用来补缺失的指令")
    parser.add_argument("--output", type=Path, help="默认写到子集同目录的 parsed_instructions.json")
    parser.add_argument("--review", type=Path, help="默认写 parsed_instructions_review.md")
    parser.add_argument("--review-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="只解析前 N 条，用来先看提示词效果")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="沿用输出文件里已经解析成功的条目")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--self-check", action="store_true", help="不访问模型，只检查解析和校验")
    parser.add_argument("--dry-run", action="store_true", help="打印第一条指令的提示词后退出")
    args = parser.parse_args()
    if args.self_check:
        return args
    if args.subset is None:
        parser.error("pass --subset, or --self-check")
    if args.output is None:
        args.output = args.subset.with_name("parsed_instructions.json")
    if args.review is None:
        args.review = args.output.with_name("parsed_instructions_review.md")
    return args


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.dry_run:
        jobs = [row for row in load_jobs(args.subset) if instruction_of(row)]
        if not jobs:
            raise SystemExit("子集里没有指令")
        print(build_prompt(instruction_of(jobs[0])))
        return
    run(args)


if __name__ == "__main__":
    main()
