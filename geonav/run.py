"""Run the zero-shot loop in Habitat.

The 20 saved frame folders are not loaded. Episodes come from parsed_instructions.json,
and the simulator loads the matching R2R episodes and scenes.

Start SAM3 in its own env before landmark or explore runs that should see objects:

    conda activate sam3
    python -m geonav.sam3_server --checkpoint /root/data1/models/sam3_weights/sam3.pt --port 8001

Then, in the streamvln env (Python 3.9, Habitat 0.2.4), from the GeoVLN directory:

    python -m geonav.run --mode goal
    python -m geonav.run --mode explore
    python -m geonav.run --mode landmark

Episode files go to --output (default /root/data1/StreamVLN/results/geonav_r2r_100).
Progress is printed on the terminal and appended to <output>/run.log.
A full terminal drops that copy instead of stalling the run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from geonav._bootstrap import RESULT_DIR, ROOT
from geonav.episodes import episode_key, load_rows

PARSED = "/root/data1/StreamVLN/results/r2r_val_unseen_v1_3/parsed_instructions.json"
DATA = "/root/data1/StreamVLN/data/datasets/r2r/val_unseen/val_unseen.json.gz"
SCENES = "/root/data1/StreamVLN/data/scene_datasets"


def main():
    parser = argparse.ArgumentParser(description="在 Habitat 里跑零样本导航闭环")
    parser.add_argument("--mode", choices=("goal", "explore", "landmark"), default="landmark")
    parser.add_argument("--parsed", type=Path, default=Path(PARSED))
    parser.add_argument("--data-path", type=Path, default=Path(DATA))
    parser.add_argument("--scenes-dir", type=Path, default=Path(SCENES))
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "vln_r2r.yaml")
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--output", type=Path, default=RESULT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="只跑解析列表里的前 N 条，0 表示全部")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--vis-stride", type=int, default=10)
    parser.add_argument("--vis-fps", type=int, default=5, help="vis_<轨迹编号>_<模式>.mp4 的帧率，画面来自 vis 里已经按 vis-stride 抽过的帧")
    parser.add_argument(
        "--vis-video",
        type=Path,
        default=None,
        help="把已有的 vis 帧合成 vis_<轨迹编号>_<模式>.mp4，不启动仿真器。可以是一条 episode 目录，或包含多条的结果目录",
    )
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8001")
    parser.add_argument("--no-sam3", action="store_true")
    parser.add_argument("--no-look", action="store_true", help="跳过开局和进房间后的环视")
    parser.add_argument("--dry-list", action="store_true", help="只打印解析文件里的 episode，不启动仿真器")
    args = parser.parse_args()
    if args.vis_video:
        from geonav.video import encode_saved_vis

        encode_saved_vis(args.vis_video, args.vis_fps)
        return
    if args.dry_list:
        rows = load_rows(args.parsed)
        if args.limit:
            rows = rows[: args.limit]
        for row in rows:
            scene, episode_id = episode_key(row["scene_id"], row["episode_id"])
            parsed = row.get("parsed") or {}
            end = parsed.get("end_landmark") or "-"
            print(f"{scene}_{episode_id}  end={end}")
        print(f"{len(rows)} episodes")
        return
    from geonav.loop import run_episodes

    run_episodes(args)


if __name__ == "__main__":
    main()
