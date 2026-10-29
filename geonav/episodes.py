"""Load the parsed episode list. The simulator is not imported here."""

from __future__ import annotations

import json
from pathlib import Path

from geonav._bootstrap import install

install()

from parse_instructions import scene_key  # noqa: E402


def load_rows(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload["episodes"] if isinstance(payload, dict) else payload
    kept = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or "scene_id" not in row or "episode_id" not in row:
            continue
        key = (scene_key(row["scene_id"]), str(row["episode_id"]))
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    if not kept:
        raise SystemExit(f"{path} 里没有带 scene_id 和 episode_id 的条目")
    kept.sort(key=lambda row: (scene_key(row["scene_id"]), _order(row["episode_id"])))
    return kept


def episode_key(scene_id, episode_id):
    return (scene_key(scene_id), str(episode_id))


def _order(episode_id):
    text = str(episode_id)
    return (0, int(text)) if text.isdigit() else (1, text)
