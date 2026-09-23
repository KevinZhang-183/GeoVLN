#!/usr/bin/env python3
"""Call StreamVLN's real input splice with stand-in tensors.

Does not load the 7B checkpoint or VGGT-Omega. Geometry stays off unless
geo_registers is passed. Eight image tokens then gain 16 registers each.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "streamvln"))

from model.stream_video_vln import StreamVLNForCausalLM  # noqa: E402
from streamvln.model.geo_register_projector import GeoRegisterProjector  # noqa: E402
from utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX  # noqa: E402

FRAMES = 8
TOKENS_PER_FRAME = 4
LLM_DIM = 3584


class _SpliceStandIn:
    def __init__(self, image_features: torch.Tensor) -> None:
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(
            tune_mm_mlp_adapter=False,
            mm_use_im_start_end=False,
            tokenizer_model_max_length=None,
            tokenizer_padding_side="right",
        )
        self.image_features = [image_features]
        self.memory_features = [None]

    def get_vision_tower(self):
        return object()

    def encode_rgbd(self, images, depths, poses, intrinsics, time_ids=None, task_ids=None):
        return self.image_features, self.memory_features

    def get_model(self):
        return self

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = torch.zeros(input_ids.shape[0], LLM_DIM)
        embeds[:, 0] = input_ids.to(dtype=torch.float32)
        return embeds


def _call(standin, input_ids, labels, attention_mask, position_ids, registers, projector):
    return StreamVLNForCausalLM.prepare_inputs_labels_for_multimodal(
        standin,
        input_ids,
        position_ids,
        attention_mask,
        None,
        labels,
        images=torch.zeros(1, FRAMES, 3, 8, 8),
        image_sizes=None,
        depths=None,
        poses=None,
        intrinsics=None,
        geo_registers=registers,
        geo_projector=projector,
    )


def main() -> None:
    image_features = torch.zeros(FRAMES, TOKENS_PER_FRAME, LLM_DIM)
    for frame_id in range(FRAMES):
        image_features[frame_id, :, 0] = 1000 + frame_id
    text = [5] + [IMAGE_TOKEN_INDEX] * FRAMES + [10, 11, 12]
    input_ids = torch.tensor([text])
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    standin = _SpliceStandIn(image_features)

    _, off_pos, off_mask, _, off_embeds, off_labels = _call(
        standin, input_ids, labels, attention_mask, position_ids, None, None
    )
    off_len = 1 + FRAMES * TOKENS_PER_FRAME + 3
    print("off", tuple(off_embeds.shape))
    if tuple(off_embeds.shape) != (1, off_len, LLM_DIM):
        raise SystemExit(1)
    if not torch.equal(off_labels[0, :1], torch.tensor([5])):
        raise SystemExit(1)
    if not torch.all(off_labels[0, 1:-3] == IGNORE_INDEX):
        raise SystemExit(1)
    if not torch.equal(off_labels[0, -3:], torch.tensor([10, 11, 12])):
        raise SystemExit(1)
    if int(off_mask.sum()) != off_len or not torch.equal(off_pos[0], torch.arange(off_len)):
        raise SystemExit(1)
    print("off_labels_ok")

    projector = GeoRegisterProjector()
    torch.nn.init.zeros_(projector.proj.weight)
    torch.nn.init.constant_(projector.proj.bias, 7.0)
    registers = torch.zeros(1, FRAMES, 16, 2048)
    _, on_pos, on_mask, _, on_embeds, on_labels = _call(
        standin, input_ids, labels, attention_mask, position_ids, registers, projector
    )
    extra = FRAMES * 16
    on_len = off_len + extra
    print("on", tuple(on_embeds.shape))
    print("extra", extra)
    if tuple(on_embeds.shape) != (1, on_len, LLM_DIM):
        raise SystemExit(1)

    cursor = 1
    for frame_id in range(FRAMES):
        image_slot = on_embeds[0, cursor:cursor + TOKENS_PER_FRAME, 0]
        if not torch.equal(image_slot, torch.full((TOKENS_PER_FRAME,), 1000 + frame_id)):
            raise SystemExit(1)
        cursor += TOKENS_PER_FRAME
        geo_slot = on_embeds[0, cursor:cursor + 16, 0]
        if not torch.equal(geo_slot, torch.full((16,), 7.0)):
            raise SystemExit(1)
        cursor += 16
    if cursor != on_len - 3:
        raise SystemExit(1)
    if not torch.equal(on_embeds[0, -3:, 0], torch.tensor([10.0, 11.0, 12.0])):
        raise SystemExit(1)
    if not torch.equal(on_labels[0, :1], torch.tensor([5])):
        raise SystemExit(1)
    if not torch.all(on_labels[0, 1:-3] == IGNORE_INDEX):
        raise SystemExit(1)
    if not torch.equal(on_labels[0, -3:], torch.tensor([10, 11, 12])):
        raise SystemExit(1)
    if int(on_mask.sum()) != on_len or not torch.equal(on_pos[0], torch.arange(on_len)):
        raise SystemExit(1)
    print("geo_labels", IGNORE_INDEX)
    print("action_labels", [10, 11, 12])
    print("wire_ok")


if __name__ == "__main__":
    main()
