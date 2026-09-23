#!/usr/bin/env python3
"""Check VGGT-Omega register tensor shape on 8 frames.

Does not modify StreamVLN. Run inside the separate conda env ``vggt-omega``,
not ``opennav`` or ``streamvln``.

Expected registers shape: (1, 8, 16, 2048)
"""

from __future__ import annotations

import argparse
import glob
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images

DEFAULT_CHECKPOINT = "/root/data1/models/vggt-omega/vggt_omega_1b_512.pt"
EXPECTED_REGISTERS = (1, 8, 16, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Directory of RGB images. Omit to use synthetic frames.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto uses CUDA when present, otherwise exits without loading the model.",
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    return parser.parse_args()


def collect_images(image_dir: Path | None, num_frames: int) -> list[str]:
    if image_dir is None:
        tmp = Path(tempfile.mkdtemp(prefix="vggt_shape_"))
        for index in range(num_frames):
            Image.new("RGB", (640, 480), (index * 20, 40, 80)).save(tmp / f"{index:02d}.png")
        print(f"synthetic_images {tmp}")
        return sorted(str(path) for path in tmp.glob("*.png"))

    paths = sorted(
        path
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
        for path in glob.glob(str(image_dir / pattern))
    )
    if len(paths) < num_frames:
        raise SystemExit(f"{image_dir} has {len(paths)} images, need at least {num_frames}")
    return paths[:num_frames]


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    raise SystemExit(
        "No GPU is visible. The script is in place; rerun it after the A6000 is attached. "
        "Do not pass --device cpu unless you intend to load the 1B model on CPU."
    )


def main() -> None:
    args = parse_args()
    if args.num_frames < 1:
        raise SystemExit("--num-frames must be positive")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    device = resolve_device(args.device)
    image_paths = collect_images(args.image_dir, args.num_frames)

    model = VGGTOmega().to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    del state

    images = load_and_preprocess_images(image_paths, image_resolution=args.image_resolution).to(device)
    with torch.inference_mode():
        predictions = model(images)

    tokens = predictions["camera_and_register_tokens"]
    registers = tokens[:, :, 1:]
    print("device", device)
    print("images", tuple(images.shape))
    print("camera_and_register_tokens", tuple(tokens.shape))
    print("registers", tuple(registers.shape))

    expected = (EXPECTED_REGISTERS[0], args.num_frames, EXPECTED_REGISTERS[2], EXPECTED_REGISTERS[3])
    if tuple(registers.shape) != expected:
        print(f"UNEXPECTED expected {expected}", file=sys.stderr)
        raise SystemExit(1)
    print("shape_ok")


if __name__ == "__main__":
    main()
