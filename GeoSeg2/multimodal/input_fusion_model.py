"""Input-level IRRG+DSM baseline used by the completed Vaihingen experiment."""

from __future__ import annotations

import torch
from torch import nn

from geoseg.models.RoGLSNet import RoGLSNet


class InputFusionRoGLSNet(nn.Module):
    """Adapt aligned DSM to three channels and inject it before RoGLSNet."""

    def __init__(
        self,
        *,
        num_classes: int,
        auxiliary_channels: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if auxiliary_channels < 1:
            raise ValueError("auxiliary_channels must be positive")

        hidden = max(8, auxiliary_channels * 4)
        self.auxiliary_adapter = nn.Sequential(
            nn.Conv2d(auxiliary_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Sigmoid(),
        )
        self.segmenter = RoGLSNet(num_classes=num_classes, pretrained=pretrained)

        # The original experiment started exactly from the optical baseline.
        nn.init.zeros_(self.auxiliary_adapter[-1].weight)
        nn.init.zeros_(self.auxiliary_adapter[-1].bias)

    def forward(self, primary: torch.Tensor, auxiliary: torch.Tensor):
        if primary.ndim != 4 or primary.shape[1] != 3:
            raise ValueError(f"primary must have shape [B,3,H,W], got {tuple(primary.shape)}")
        if auxiliary.ndim != 4 or auxiliary.shape[-2:] != primary.shape[-2:]:
            raise ValueError("auxiliary must be BCHW and spatially aligned with primary")
        auxiliary_rgb = self.auxiliary_adapter(auxiliary)
        gate = self.gate(torch.cat((primary, auxiliary_rgb), dim=1))
        return self.segmenter(primary + gate * auxiliary_rgb)
