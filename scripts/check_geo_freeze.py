#!/usr/bin/env python3
"""Load StreamVLN, freeze it, and count trainable projector parameters.

Does not run a forward pass, Habitat, or navigation eval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import transformers

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "streamvln"))

from model.stream_video_vln import StreamVLNForCausalLM  # noqa: E402
from streamvln.model.geo_register_projector import GeoRegisterProjector  # noqa: E402

PROJECTOR_PARAMS = 2048 * 3584 + 3584
DEFAULT_MODEL = "/root/data1/models/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("cuda is required")

    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )
    model.requires_grad_(False)
    model.eval()
    model.to("cuda")

    projector = GeoRegisterProjector().to(device="cuda", dtype=torch.bfloat16)
    frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    print("frozen_trainable", frozen)
    print("projector_trainable", trainable)
    print("cuda_gb", round(torch.cuda.memory_allocated() / (1024 ** 3), 2))
    if frozen != 0 or trainable != PROJECTOR_PARAMS:
        raise SystemExit(1)
    print("freeze_ok")


if __name__ == "__main__":
    main()
