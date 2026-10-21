#!/usr/bin/env python3
"""把 R2R 指令拆成有序地标链，缓存成 parsed_instructions.json。

子集来自 select_r2r_subset.py。result.json 里已经有 episode_instruction，
所以这一步不需要再打开 Habitat 数据集。没有这条字段时，用 --dataset 按
scene_id + episode_id 去 json/json.gz 里补。

地标短语要能交给 SAM 做文本分割：具体物体或门洞，不单写房间名或方向。

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


ACTIONS = ("exit", "enter", "go_to", "pass", "turn", "stop")
ACTION_ALIAS = {
    "leave": "exit",
    "walk_out": "exit",
    "go_out": "exit",
    "walk": "go_to",
    "go": "go_to",
    "approach": "go_to",
    "find": "go_to",
    "pass_by": "pass",
    "go_past": "pass",
    "walk_past": "pass",
    "turn_left": "turn",
    "turn_right": "turn",
    "wait": "stop",
    "halt": "stop",
}
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
ROOM_ONLY = {
    "kitchen",
    "bedroom",
    "bathroom",
    "hallway",
    "hall",
    "corridor",
    "living room",
    "dining room",
    "office",
    "lobby",
    "garage",
    "balcony",
    "closet",
    "laundry room",
    "foyer",
    "entryway",
    "room",
}

PROMPT = """Convert one indoor navigation instruction into an ordered chain of visual landmarks.

A later system segments each landmark in RGB with an open-vocabulary segmenter, then walks to them in order. Keep the intermediate landmarks. Do not reduce the instruction to only the final object.

Reply with one JSON object and no other text. Schema:
{{
  "subtasks": [
    {{"id": 1, "action": "exit", "landmark": "bedroom door", "done_when": "passed through the bedroom door"}}
  ],
  "end_landmark": "potted plant",
  "end_context": ["hallway", "past the stairs"]
}}

Rules:
- action is one of: exit, enter, go_to, pass, turn, stop.
- landmark is a short English noun phrase for something that can be segmented: a door, doorway, stairs, furniture, appliance, plant, window, picture, counter, rug. Three words or fewer when possible.
- Do not use a bare direction (left, right, forward) as a landmark.
- Do not use a bare room name (kitchen, bedroom, hallway) as a landmark. If the instruction only names a room, use that room's doorway and put the room name in end_context or done_when.
- The last subtask is where the instruction says to stop. Its action is stop, and end_landmark copies that landmark.
- end_context is a short list of phrases that distinguish this stop location from other copies of the same object.
- Use 2 to 6 subtasks. Fold "walk straight" into the next landmark. A turn still needs the object or doorway that marks where to turn.
- done_when is one short condition for that subtask.

Example 1
Instruction: Walk out of the bedroom, turn left, go past the stairs, and stop next to the potted plant.
{{"subtasks": [{{"id": 1, "action": "exit", "landmark": "bedroom door", "done_when": "passed through the bedroom door"}}, {{"id": 2, "action": "pass", "landmark": "stairs", "done_when": "the stairs are behind the agent"}}, {{"id": 3, "action": "stop", "landmark": "potted plant", "done_when": "within 1.5 m of the potted plant"}}], "end_landmark": "potted plant", "end_context": ["past the stairs", "after leaving the bedroom"]}}

Example 2
Instruction: Go down the hallway, turn right into the kitchen, and stop in front of the refrigerator.
{{"subtasks": [{{"id": 1, "action": "go_to", "landmark": "hallway doorway", "done_when": "entered the hallway"}}, {{"id": 2, "action": "enter", "landmark": "kitchen doorway", "done_when": "passed through the kitchen doorway"}}, {{"id": 3, "action": "stop", "landmark": "refrigerator", "done_when": "in front of the refrigerator"}}], "end_landmark": "refrigerator", "end_context": ["kitchen", "right turn from the hallway"]}}

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
    prompt = PROMPT.format(instruction=instruction)
    if errors:
        prompt += "\nThe previous JSON was rejected:\n- " + "\n- ".join(errors)
        prompt += "\nReply with a corrected JSON object only."
    return prompt


def _clean_phrase(value) -> str:
    return " ".join(str(value).strip().strip(".").split())


def _action(value) -> str | None:
    name = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if name in ACTIONS:
        return name
    return ACTION_ALIAS.get(name)


def validate(parsed: dict) -> tuple[dict | None, list[str], list[str]]:
    """返回 (规范化结果, 硬错误, 软标记)。硬错误非空时结果为 None。"""
    if not isinstance(parsed, dict):
        return None, ["top level is not an object"], []
    subtasks = parsed.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        return None, ["subtasks must be a non-empty list"], []
    if len(subtasks) > 8:
        return None, ["more than 8 subtasks"], []

    errors = []
    flags = []
    cleaned = []
    for index, item in enumerate(subtasks, start=1):
        if not isinstance(item, dict):
            errors.append(f"subtask {index} is not an object")
            continue
        action = _action(item.get("action"))
        landmark = _clean_phrase(item.get("landmark", ""))
        done = _clean_phrase(item.get("done_when", ""))
        if action is None:
            errors.append(f"subtask {index} has an unknown action")
        if not landmark or len(landmark.split()) > 8:
            errors.append(f"subtask {index} landmark is empty or longer than 8 words")
        elif landmark.lower() in DIRECTION_ONLY:
            errors.append(f"subtask {index} landmark is only a direction")
        if not done:
            errors.append(f"subtask {index} is missing done_when")
        if landmark.lower() in ROOM_ONLY:
            flags.append(f"subtask {index} landmark is a room name: {landmark}")
        if landmark and len(landmark.split()) > 4:
            flags.append(f"subtask {index} landmark is long: {landmark}")
        if action and landmark and done:
            cleaned.append({"id": index, "action": action, "landmark": landmark, "done_when": done})

    if errors or len(cleaned) != len(subtasks):
        return None, errors, flags

    end = _clean_phrase(parsed.get("end_landmark", ""))
    if end.lower() != cleaned[-1]["landmark"].lower():
        if end:
            flags.append(f"end_landmark {end!r} replaced by the last landmark")
        end = cleaned[-1]["landmark"]
    if cleaned[-1]["action"] != "stop":
        flags.append("last action is not stop")

    context = parsed.get("end_context", [])
    if isinstance(context, str):
        context = [context] if context.strip() else []
    if not isinstance(context, list):
        return None, ["end_context must be a list"], flags
    context = [_clean_phrase(item) for item in context if _clean_phrase(item)]
    return {"subtasks": cleaned, "end_landmark": end, "end_context": context}, [], flags


def parse_one(client: VLMClient, instruction: str, retries: int = 1) -> tuple[dict | None, list[str], str | None]:
    errors: list[str] | None = None
    last_raw = None
    for _ in range(retries + 1):
        raw = client.complete(build_prompt(instruction, errors), max_tokens=900, temperature=0.0)
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
        "看三件事：子任务顺序对不对、地标是不是能分割的物体、终点地标对不对。",
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
        if parsed["end_context"]:
            lines.append("上下文：" + "；".join(parsed["end_context"]))
        for item in parsed["subtasks"]:
            lines.append(
                f"{item['id']}. {item['action']} | {item['landmark']} | {item['done_when']}"
            )
        flags = row.get("flags") or []
        lines.append("标记：" + ("；".join(flags) if flags else "无"))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(rows: list[dict]) -> str:
    parsed = [row for row in rows if row.get("parsed")]
    flagged = [row for row in parsed if row.get("flags")]
    room = sum(1 for row in parsed if any("room name" in flag for flag in row["flags"]))
    not_stop = sum(1 for row in parsed if any("not stop" in flag for flag in row["flags"]))
    return (
        f"parsed {len(parsed)}/{len(rows)}  "
        f"flagged {len(flagged)}  room_name {room}  last_not_stop {not_stop}  "
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
        state = item["end_landmark"] if parsed else "FAILED"
        if parsed:
            item_end = parsed["end_landmark"]
            print(f"[{index}/{len(rows)}] {row['scene_id']} {row['episode_id']} -> {item_end}", flush=True)
        else:
            print(f"[{index}/{len(rows)}] {row['scene_id']} {row['episode_id']} -> {state}", flush=True)
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

    good = {
        "subtasks": [
            {"id": 9, "action": "leave", "landmark": "bedroom door", "done_when": "passed through"},
            {"id": 2, "action": "stop", "landmark": "potted plant.", "done_when": "within 1.5 m"},
        ],
        "end_landmark": "plant",
        "end_context": "hallway",
    }
    normalized, errors, flags = validate(good)
    assert not errors, errors
    assert normalized["subtasks"][0]["action"] == "exit"
    assert normalized["subtasks"][0]["id"] == 1
    assert normalized["end_landmark"] == "potted plant"
    assert normalized["end_context"] == ["hallway"]
    assert any("replaced" in flag for flag in flags)

    blocked, errors, _flags = validate(
        {"subtasks": [{"action": "go_to", "landmark": "left", "done_when": "turned"}], "end_landmark": "left"}
    )
    assert blocked is None and errors

    room, errors, flags = validate(
        {
            "subtasks": [{"action": "stop", "landmark": "kitchen", "done_when": "inside the kitchen"}],
            "end_landmark": "kitchen",
            "end_context": [],
        }
    )
    assert room is not None and not errors
    assert any("room name" in flag for flag in flags)

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
