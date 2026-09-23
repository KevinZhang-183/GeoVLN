#!/usr/bin/env python3
"""Check geometry labels before wiring them into StreamVLN.

The 128 register tokens sit between the image tokens and the action tokens.
Their labels are -100. With registers omitted, the sequence is unchanged.
Does not load VGGT-Omega or the 7B checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamvln.model.geo_register_projector import (  # noqa: E402
    IGNORE_INDEX,
    REGISTERS_PER_FRAME,
    GeoRegisterProjector,
    append_geo_with_labels,
)


def _sequence(image_tokens: torch.Tensor, image_labels: torch.Tensor, action_labels: torch.Tensor, registers, projector):
    image_out, labels_out = append_geo_with_labels(image_tokens, image_labels, registers, projector)
    action_tokens = torch.zeros(action_labels.shape[0], image_tokens.shape[-1])
    tokens = torch.cat([image_out, action_tokens], dim=0)
    labels = torch.cat([labels_out, action_labels], dim=0)
    return tokens, labels


def main() -> None:
    frames = 8
    image_len = 64
    action_labels = torch.tensor([10, 11, 12])
    image_tokens = torch.zeros(image_len, 3584)
    image_labels = torch.full((image_len,), IGNORE_INDEX)
    projector = GeoRegisterProjector()

    off_tokens, off_labels = _sequence(image_tokens, image_labels, action_labels, None, None)
    print("off", tuple(off_tokens.shape))
    if tuple(off_tokens.shape) != (image_len + action_labels.shape[0], 3584):
        raise SystemExit(1)
    if not torch.equal(off_labels, torch.cat([image_labels, action_labels])):
        raise SystemExit(1)
    print("labels_unchanged")

    registers = torch.zeros(1, frames, REGISTERS_PER_FRAME, 2048)
    on_tokens, on_labels = _sequence(image_tokens, image_labels, action_labels, registers, projector)
    extra = frames * REGISTERS_PER_FRAME
    print("on", tuple(on_tokens.shape))
    if tuple(on_tokens.shape) != (image_len + extra + action_labels.shape[0], 3584):
        raise SystemExit(1)
    if not torch.equal(on_labels[:image_len], image_labels):
        raise SystemExit(1)
    if not torch.all(on_labels[image_len:image_len + extra] == IGNORE_INDEX):
        raise SystemExit(1)
    if not torch.equal(on_labels[image_len + extra:], action_labels):
        raise SystemExit(1)
    print("geo_labels", IGNORE_INDEX)
    print("action_labels", action_labels.tolist())
    print("labels_ok")


if __name__ == "__main__":
    main()
