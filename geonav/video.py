"""Turn the sampled vis frames of one episode into a short video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def clear_vis(vis_dir):
    """Drop frames left by an earlier run so they are not encoded into this video."""
    vis_dir = Path(vis_dir)
    if not vis_dir.is_dir():
        return
    for path in vis_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            path.unlink()


def write_vis_video(vis_dir, out_path, fps=5):
    """Write ``vis/0000.jpg`` ... into ``out_path``. Returns the path, or None."""
    frames = _frame_paths(vis_dir)
    if not frames:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rate = max(1, int(fps))
    # Keep the last sampled frame on screen long enough to see where the episode ended.
    frames = frames + [frames[-1]] * rate
    for encode in (_encode_av, _encode_cv2, _encode_imageio):
        try:
            if encode(frames, out_path, rate):
                return out_path
        except Exception as exc:
            print(f"  {encode.__name__} failed: {exc}", flush=True)
    print("没有可用的视频编码器。当前环境需要 opencv-python 或 av。", flush=True)
    return None


def encode_saved_vis(path, fps=5):
    """Encode one episode directory, a vis directory, or a tree of episode directories."""
    episodes = _episode_dirs(path)
    if not episodes:
        print(f"没有找到 vis 帧: {path}", flush=True)
        return []
    written = []
    for episode in episodes:
        out = write_vis_video(episode / "vis", episode / vis_video_name(episode), fps)
        if out is not None:
            written.append(out)
            print(f"wrote {out}", flush=True)
    return written


def _episode_dirs(path):
    path = Path(path)
    if not path.exists():
        return []
    if path.is_dir() and path.name == "vis":
        return [path.parent] if _frame_paths(path) else []
    if path.is_dir() and _frame_paths(path / "vis"):
        return [path]
    found = []
    if path.is_dir():
        for vis in sorted(path.rglob("vis")):
            if vis.is_dir() and _frame_paths(vis):
                found.append(vis.parent)
    return found


def vis_video_name(episode_dir):
    """``vis_<episode>_<mode>.mp4`` when the episode sits in a mode directory."""
    episode_id = _trajectory_id(episode_dir)
    mode = Path(episode_dir).parent.name
    if mode in {"goal", "explore", "landmark"}:
        return f"vis_{episode_id}_{mode}.mp4"
    return f"vis_{episode_id}.mp4"


def _trajectory_id(episode_dir):
    """Episode folders are ``{scene}_{episode_id}``."""
    name = Path(episode_dir).name
    return name.rsplit("_", 1)[-1] if "_" in name else name


def _frame_paths(vis_dir):
    vis_dir = Path(vis_dir)
    if not vis_dir.is_dir():
        return []
    return sorted(path for path in vis_dir.glob("*.jpg") if path.stem.isdigit())


def _rgb(path):
    with Image.open(path) as image:
        frame = np.asarray(image.convert("RGB"))
    height, width = frame.shape[:2]
    if height % 2 or width % 2:
        frame = np.pad(frame, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
    return np.ascontiguousarray(frame)


def _encode_av(frames, out_path, fps):
    import av

    first = _rgb(frames[0])
    height, width = first.shape[:2]
    container = av.open(str(out_path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for path in frames:
            video_frame = av.VideoFrame.from_ndarray(_rgb(path), format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return out_path.is_file() and out_path.stat().st_size > 0


def _encode_cv2(frames, out_path, fps):
    import cv2

    first = _rgb(frames[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        return False
    try:
        for path in frames:
            bgr = _rgb(path)[:, :, ::-1]
            writer.write(np.ascontiguousarray(bgr))
    finally:
        writer.release()
    return out_path.is_file() and out_path.stat().st_size > 0


def _encode_imageio(frames, out_path, fps):
    import imageio.v2 as imageio

    images = [_rgb(path) for path in frames]
    imageio.mimsave(out_path, images, fps=fps)
    return out_path.is_file() and out_path.stat().st_size > 0
