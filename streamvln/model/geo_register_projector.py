"""Project VGGT-Omega registers into StreamVLN's token space."""

from __future__ import annotations

import torch
import torch.nn as nn

GEO_DIM = 2048
LLM_DIM = 3584
REGISTERS_PER_FRAME = 16
IGNORE_INDEX = -100


class GeoRegisterProjector(nn.Module):
    """One linear layer. VGGT-Omega and the StreamVLN backbone stay frozen."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(GEO_DIM, LLM_DIM)

    def forward(self, registers: torch.Tensor) -> torch.Tensor:
        if registers.shape[-1] != GEO_DIM:
            raise ValueError(f"expected register dim {GEO_DIM}, got {registers.shape[-1]}")
        flat = registers.reshape(-1, GEO_DIM)
        return self.proj(flat.to(dtype=self.proj.weight.dtype))


def append_geo_tokens(visual_tokens: torch.Tensor, registers: torch.Tensor, projector: GeoRegisterProjector) -> torch.Tensor:
    """Append projected registers after existing visual tokens.

    visual_tokens: (L, 3584)
    registers: (1, T, 16, 2048) or (T, 16, 2048)
    returns: (L + T * 16, 3584)
    """
    if visual_tokens.ndim != 2 or visual_tokens.shape[-1] != LLM_DIM:
        raise ValueError(f"visual_tokens must be (L, {LLM_DIM}), got {tuple(visual_tokens.shape)}")
    geo = projector(registers).to(device=visual_tokens.device, dtype=visual_tokens.dtype)
    return torch.cat([visual_tokens, geo], dim=0)


def append_geo_with_labels(
    visual_tokens: torch.Tensor,
    labels: torch.Tensor,
    registers: torch.Tensor | None,
    projector: GeoRegisterProjector | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert projected registers immediately after one image-token block.

    labels stay aligned with visual_tokens. Geometry positions use IGNORE_INDEX
    so the action loss does not train on them. registers=None leaves both tensors unchanged.
    """
    if labels.ndim != 1 or labels.shape[0] != visual_tokens.shape[0]:
        raise ValueError(
            f"labels must be ({visual_tokens.shape[0]},), got {tuple(labels.shape)}"
        )
    if registers is None:
        return visual_tokens, labels
    if projector is None:
        raise ValueError("projector is required when registers are provided")
    merged = append_geo_tokens(visual_tokens, registers, projector)
    extra = merged.shape[0] - visual_tokens.shape[0]
    geo_labels = torch.full(
        (extra,),
        IGNORE_INDEX,
        device=labels.device,
        dtype=labels.dtype,
    )
    return merged, torch.cat([labels, geo_labels], dim=0)
