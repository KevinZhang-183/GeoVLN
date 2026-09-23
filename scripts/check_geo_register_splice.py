#!/usr/bin/env python3
"""Check that 8 frames of registers become 128 tokens after the visual tokens.

Runs in the streamvln env. Does not load VGGT-Omega or the 7B checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamvln.model.geo_register_projector import (  # noqa: E402
    REGISTERS_PER_FRAME,
    GeoRegisterProjector,
    append_geo_tokens,
)


def main() -> None:
    frames = 8
    visual_len = 64
    visual = torch.zeros(visual_len, 3584)
    registers = torch.zeros(1, frames, REGISTERS_PER_FRAME, 2048)
    projector = GeoRegisterProjector()
    merged = append_geo_tokens(visual, registers, projector)
    extra = frames * REGISTERS_PER_FRAME
    print("visual", tuple(visual.shape))
    print("merged", tuple(merged.shape))
    print("extra", extra)
    if tuple(merged.shape) != (visual_len + extra, 3584):
        raise SystemExit(1)
    print("splice_ok")


if __name__ == "__main__":
    main()
