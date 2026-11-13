"""RoGLSNet with one relative-DSM bias inside RoAFM attention."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from geoseg.models.RoGLSNet import RoGLSNet


class DSMRelativeRoGLSNet(nn.Module):
    """Keep the optical network intact and penalize attention across height jumps."""

    max_height_strength = 4.0
    initial_height_strength = 0.001

    def __init__(
        self,
        *,
        num_classes: int,
        auxiliary_channels: int = 1,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if auxiliary_channels != 1:
            raise ValueError("DSMRelativeRoGLSNet expects a single DSM channel")
        self.segmenter = RoGLSNet(num_classes=num_classes, pretrained=pretrained)
        probability = self.initial_height_strength / self.max_height_strength
        initial_logit = math.log(probability / (1.0 - probability))
        self.height_strength_logit = nn.Parameter(torch.tensor(initial_logit))

    def height_strength(self) -> torch.Tensor:
        return self.max_height_strength * self.height_strength_logit.sigmoid()

    def forward(self, primary: torch.Tensor, auxiliary: torch.Tensor):
        if primary.ndim != 4 or primary.shape[1] != 3:
            raise ValueError(f"primary must have shape [B,3,H,W], got {tuple(primary.shape)}")
        if auxiliary.ndim != 4 or auxiliary.shape[1] != 1:
            raise ValueError(f"DSM must have shape [B,1,H,W], got {tuple(auxiliary.shape)}")
        if primary.shape[0] != auxiliary.shape[0] or primary.shape[-2:] != auxiliary.shape[-2:]:
            raise ValueError("DSM must be spatially aligned with IRRG")

        features = self.segmenter.backbone(primary)
        return self.segmenter.decode_features(
            features,
            primary.shape[-2:],
            height=auxiliary,
            height_strength=self.height_strength(),
        )


def load_input_fusion_segmenter(model: DSMRelativeRoGLSNet, checkpoint: str | Path) -> None:
    """Load only ``segmenter.*`` tensors from the strong input-fusion checkpoint."""
    path = Path(checkpoint).resolve()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("model_state_dict", state.get("state_dict", state))
    if not isinstance(state, dict):
        raise ValueError("initial checkpoint does not contain a model state dictionary")

    segmenter_state: dict[str, torch.Tensor] = {}
    for raw_name, value in state.items():
        name = raw_name.removeprefix("module.")
        if name.startswith("segmenter."):
            segmenter_state[name.removeprefix("segmenter.")] = value
    if not segmenter_state:
        raise ValueError("initial checkpoint contains no segmenter.* tensors")
    model.segmenter.load_state_dict(segmenter_state, strict=True)
