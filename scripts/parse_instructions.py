#!/usr/bin/env python3
"""把 R2R 指令拆成有序地标链，缓存成 parsed_instructions.json。

子集来自 select_r2r_subset.py。result.json 里已经有 episode_instruction，
所以这一步不需要再打开 Habitat 数据集。没有这条字段时，用 --dataset 按
scene_id + episode_id 去 json/json.gz 里补。

动作有 go_to / exit / turn_left / turn_right / turn_around / stop 六类。每步的 landmark
是要到达的目标，near 是只用来定位目标的参照物，side 是目标在左还是右。
exit 的 landmark 是被离开的房间，不进 landmarks 列表。房间名不拒收，终点是房间时打标记。

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
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from select_r2r_subset import load_episodes  # noqa: E402
from vlm_client import VLMClient, extract_json  # noqa: E402


ACTIONS = ("go_to", "exit", "turn_left", "turn_right", "turn_around", "stop")
TURNS = ("turn_left", "turn_right", "turn_around")
ACTION_ALIAS = {
    "go": "go_to",
    "goto": "go_to",
    "walk": "go_to",
    "walk_to": "go_to",
    "move_to": "go_to",
    "approach": "go_to",
    "find": "go_to",
    "leave": "exit",
    "walk_out": "exit",
    "go_out": "exit",
    "get_out": "exit",
    "exit_room": "exit",
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
    "turn_back": "turn_around",
    "u_turn": "turn_around",
    "turn_180": "turn_around",
    "wait": "stop",
    "halt": "stop",
    "stop_at": "stop",
}
ARTICLES = {"the", "a", "an"}
SIDE_PREFIX = re.compile(r"^(?:to\s+)?(?:the\s+)?(left|right)(?:\s+side)?\s+of\s+(.+)$")
SIDE_SUFFIX = re.compile(r"^(.+?)\s+(?:on|to)\s+(?:the|your)\s+(left|right)$")
NEAR_SPLIT = re.compile(r"^(.+?)\s+(?:next to|near|beside|by)\s+(.+)$")
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

Each step is {"action": ..., "landmark": ..., "near": ..., "side": ...}.
action is one of:
- go_to: walk to, into, through or past the landmark.
- exit: leave a room. landmark is the room being left, or "" if it is not named.
- turn_left, turn_right, turn_around: only when the instruction says turn, take a left/right, or turn around. landmark, near and side are "".
- stop: only the last step, where the instruction says to stop or wait. "Stand in front of", "face", and "turn to face" are go_to, not stop.
landmark is the place to reach, in 1 to 3 words: an object, door, doorway or room.
near is another object named only to locate the landmark, else "". In "the doorway by the flower pot", landmark is "doorway" and near is "flower pot".
side is "left" or "right" when the instruction says which side the landmark is on, else "". "On your left" and "to the right of" set side. They are not turns.

Reply with JSON only: {"steps": [...]}

Instruction: Turn around and walk out of the bedroom. Go past the stairs and stop next to the potted plant.
{"steps": [{"action": "turn_around", "landmark": "", "near": "", "side": ""}, {"action": "exit", "landmark": "bedroom", "near": "", "side": ""}, {"action": "go_to", "landmark": "stairs", "near": "", "side": ""}, {"action": "stop", "landmark": "potted plant", "near": "", "side": ""}]}

Instruction: Walk down the hallway to the second doorway on your left. Turn left into the kitchen and wait in front of the refrigerator.
{"steps": [{"action": "go_to", "landmark": "second doorway", "near": "", "side": "left"}, {"action": "turn_left", "landmark": "", "near": "", "side": ""}, {"action": "go_to", "landmark": "kitchen", "near": "", "side": ""}, {"action": "stop", "landmark": "refrigerator", "near": "", "side": ""}]}

Instruction: Go through the doorway to the right of the table, then stop at the door by the flower pot.
{"steps": [{"action": "go_to", "landmark": "doorway", "near": "table", "side": "right"}, {"action": "stop", "landmark": "door", "near": "flower pot", "side": ""}]}

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


def _side(value) -> str:
    text = str(value or "").lower()
    has_left, has_right = "left" in text, "right" in text
    if has_left == has_right:
        return ""
    return "left" if has_left else "right"


def _split_side(landmark: str) -> tuple[str, str]:
    """把 "left of living area"、"doorway on your left" 拆成 (地标, 方位)。"""
    match = SIDE_PREFIX.match(landmark)
    if match:
        return _landmark(match.group(2)), match.group(1)
    match = SIDE_SUFFIX.match(landmark)
    if match:
        return _landmark(match.group(1)), match.group(2)
    return landmark, ""


def _action(value, raw_landmark) -> str | None:
    name = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if name in ACTIONS:
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
        if action is None:
            errors.append(f"step {index} action {item.get('action')!r} is not one of {', '.join(ACTIONS)}")
            continue
        landmark, split_side = _split_side(_landmark(raw_landmark))
        side = _side(item.get("side")) or split_side
        near = _landmark(item.get("near"))
        match = NEAR_SPLIT.match(landmark)
        if match and not near:
            landmark, near = _landmark(match.group(1)), _landmark(match.group(2))
        if near == landmark:
            near = ""
        if action in TURNS:
            if landmark:
                out.append(_step("go_to", landmark, near, side))
            out.append(_step(action))
            continue
        if action == "go_to" and not landmark:
            continue
        out.append(_step(action, landmark, near, side))
    if errors:
        return None, errors, flags

    earlier_stops = 0
    for step in out[:-1]:
        if step["action"] == "stop":
            step["action"] = "go_to"
            earlier_stops += 1
    if earlier_stops:
        flags.append(f"rewrote {earlier_stops} earlier stop steps as go_to")

    if not out or out[-1]["action"] != "stop":
        flags.append("added missing stop")
        out.append(_step("stop"))
    if not out[-1]["landmark"] and len(out) >= 2 and out[-2]["action"] == "go_to":
        previous = out.pop(-2)
        out[-1].update(landmark=previous["landmark"], near=previous["near"], side=previous["side"])

    deduped = []
    for step in out:
        last = deduped[-1] if deduped else None
        # 同一类物体可以出现多次，near 或 side 不同就是另一步。只合并紧挨着的同一次到达。
        if last and step["action"] != "exit" and _same_place(step, last):
            if step["action"] == "stop":
                last["action"] = "stop"
            continue
        deduped.append(step)

    landmarks = []
    seen = set()
    for step in deduped:
        if step["action"] == "exit" or not step["landmark"]:
            continue
        key = (step["landmark"], step["near"], step["side"])
        if key in seen:
            continue
        seen.add(key)
        landmarks.append(step["landmark"])
    if not landmarks:
        return None, ["no step has a landmark; name the objects, doors or rooms from the instruction"], flags

    subtasks = []
    for index, step in enumerate(deduped, start=1):
        if len(step["landmark"].split()) > 4:
            flags.append(f"step {index} landmark is long: {step['landmark']}")
        subtasks.append({"id": index, **step})
    if _place_only(subtasks[-1]["landmark"]):
        flags.append(f"stop landmark is a room: {subtasks[-1]['landmark']}")
    return {"subtasks": subtasks, "landmarks": landmarks, "end_landmark": landmarks[-1]}, [], flags


def _same_place(left: dict, right: dict) -> bool:
    return bool(left["landmark"]) and (
        left["landmark"], left["near"], left["side"]
    ) == (
        right["landmark"], right["near"], right["side"]
    )


def _step(action: str, landmark: str = "", near: str = "", side: str = "") -> dict:
    return {"action": action, "landmark": landmark, "near": near, "side": side}


def parse_one(client: VLMClient, instruction: str, retries: int = 1) -> tuple[dict | None, list[str], str | None]:
    errors: list[str] | None = None
    last_raw = None
    for _ in range(retries + 1):
        raw = client.complete(build_prompt(instruction, errors), max_tokens=700, temperature=0.0, json_mode=True)
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
        "看四件事：步骤顺序和转向对不对、方位修饰有没有被误当成转向、目标和参照物有没有颠倒、终点地标对不对。",
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
            detail = item["landmark"]
            if item.get("side"):
                detail += f" [{item['side']}]"
            if item.get("near"):
                detail += f" (near {item['near']})"
            lines.append(f"{item['id']}. {item['action']} {detail}".rstrip())
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
                {"action": "turn back", "landmark": ""},
                {"action": "leave", "landmark": "the bathroom"},
                {"action": "turn", "landmark": "left"},
                {"action": "walk", "landmark": "forward"},
                {"action": "turn right", "landmark": "stairs"},
                {"action": "go_to", "landmark": "doorway", "near": "Flower pot", "side": "on your right"},
                {"action": "stop", "landmark": "Potted plant."},
            ]
        }
    )
    assert not errors, errors
    assert [(s["action"], s["landmark"], s["near"], s["side"]) for s in normalized["subtasks"]] == [
        ("turn_around", "", "", ""),
        ("exit", "bathroom", "", ""),
        ("turn_left", "", "", ""),
        ("go_to", "stairs", "", ""),
        ("turn_right", "", "", ""),
        ("go_to", "doorway", "flower pot", "right"),
        ("stop", "potted plant", "", ""),
    ]
    assert [s["id"] for s in normalized["subtasks"]] == [1, 2, 3, 4, 5, 6, 7]
    assert normalized["landmarks"] == ["stairs", "doorway", "potted plant"]
    assert normalized["end_landmark"] == "potted plant"
    assert not flags

    merged, errors, flags = validate(
        {"steps": [{"action": "go_to", "landmark": "left of living area"},
                   {"action": "go_to", "landmark": "door next to recycling bin"}]}
    )
    assert not errors, errors
    assert [(s["action"], s["landmark"], s["near"], s["side"]) for s in merged["subtasks"]] == [
        ("go_to", "living area", "", "left"),
        ("stop", "door", "recycling bin", ""),
    ]
    assert any("missing stop" in flag for flag in flags)

    doorways, errors, flags = validate(
        {
            "steps": [
                {"action": "go_to", "landmark": "doorway", "near": "dining room table", "side": "right"},
                {"action": "go_to", "landmark": "doorway", "near": "flower pot", "side": ""},
                {"action": "stop", "landmark": "doorway", "near": "staircase", "side": "left"},
            ]
        }
    )
    assert not errors, errors
    assert [(s["action"], s["near"], s["side"]) for s in doorways["subtasks"]] == [
        ("go_to", "dining room table", "right"),
        ("go_to", "flower pot", ""),
        ("stop", "staircase", "left"),
    ]
    assert doorways["landmarks"] == ["doorway", "doorway", "doorway"]
    assert not flags

    folded, errors, _flags = validate(
        {"steps": [{"action": "go_to", "landmark": "doorway", "near": "flower pot"},
                   {"action": "stop", "landmark": "doorway", "near": "flower pot"}]}
    )
    assert not errors, errors
    assert [(s["action"], s["near"]) for s in folded["subtasks"]] == [("stop", "flower pot")]

    faced, errors, flags = validate(
        {
            "steps": [
                {"action": "go_to", "landmark": "couch", "side": "right"},
                {"action": "stop", "landmark": "grill"},
                {"action": "turn_right", "landmark": ""},
                {"action": "go_to", "landmark": "microwaves"},
                {"action": "stop", "landmark": "doorway"},
            ]
        }
    )
    assert not errors, errors
    assert [s["action"] for s in faced["subtasks"]] == ["go_to", "go_to", "turn_right", "go_to", "stop"]
    assert faced["subtasks"][1]["landmark"] == "grill"
    assert faced["subtasks"][-1]["landmark"] == "doorway"
    assert faced["end_landmark"] == "doorway"
    assert any("earlier stop" in flag for flag in flags)

    room, errors, flags = validate({"steps": [{"action": "stop", "landmark": "kitchen"}]})
    assert room is not None and any("stop landmark is a room" in flag for flag in flags)

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
