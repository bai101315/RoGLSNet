"""Lightweight multiscale IRRG+DSM fusion for RoGLSNet."""

from __future__ import annotations

import torch
from torch import nn

from geoseg.models.RoGLSNet import RoGLSNet


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class SeparableDownsample(nn.Module):
    """A low-cost DSM encoder block."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            in_channels,
            in_channels,
            stride=stride,
            groups=in_channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.refine = ConvNormAct(
            out_channels,
            out_channels,
            groups=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.pointwise(self.depthwise(x)))


class AuxiliaryPyramid(nn.Module):
    """Extract DSM features at the four CDFormer feature resolutions."""

    def __init__(self, in_channels: int, widths: tuple[int, ...]) -> None:
        super().__init__()
        if in_channels < 1 or len(widths) != 4:
            raise ValueError("the auxiliary pyramid needs positive input channels and four widths")
        self.stem = ConvNormAct(in_channels, widths[0], stride=2)
        self.stage1 = SeparableDownsample(widths[0], widths[0], stride=2)
        self.stage2 = SeparableDownsample(widths[0], widths[1], stride=2)
        self.stage3 = SeparableDownsample(widths[1], widths[2], stride=2)
        self.stage4 = SeparableDownsample(widths[2], widths[3], stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        feature1 = self.stage1(x)
        feature2 = self.stage2(feature1)
        feature3 = self.stage3(feature2)
        feature4 = self.stage4(feature3)
        return feature1, feature2, feature3, feature4


class GatedFeatureFusion(nn.Module):
    """Inject complementary DSM structure with spatial and channel gates.

    This is a compact asymmetric adaptation of the feature-rectification idea
    used by RGB-X models: the expensive optical stream is kept intact, while a
    small DSM stream supplies a gated residual at every encoder resolution.
    """

    def __init__(self, auxiliary_channels: int, optical_channels: int) -> None:
        super().__init__()
        hidden = max(8, optical_channels // 16)
        self.project = nn.Sequential(
            nn.Conv2d(auxiliary_channels, optical_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(optical_channels),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(optical_channels * 2, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, optical_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        # Identity initialization preserves the pretrained optical network.
        # A per-channel scale can then learn how much DSM information to use.
        self.residual_scale = nn.Parameter(torch.zeros(1, optical_channels, 1, 1))

    def forward(self, optical: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        auxiliary = self.project(auxiliary)
        if optical.shape != auxiliary.shape:
            raise ValueError(
                "optical and projected auxiliary features must be aligned, "
                f"got {tuple(optical.shape)} and {tuple(auxiliary.shape)}"
            )
        channel = self.channel_gate(torch.cat((optical, auxiliary), dim=1))
        spatial_inputs = torch.cat(
            (
                optical.mean(dim=1, keepdim=True),
                optical.amax(dim=1, keepdim=True),
                auxiliary.mean(dim=1, keepdim=True),
                auxiliary.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial = self.spatial_gate(spatial_inputs)
        return optical + self.residual_scale * (channel + spatial) * auxiliary


class AsymmetricFusionRoGLSNet(nn.Module):
    """RoGLSNet with a lightweight four-level DSM branch."""

    optical_channels = (64, 128, 320, 512)
    auxiliary_widths = (32, 64, 96, 128)

    def __init__(
        self,
        *,
        num_classes: int,
        auxiliary_channels: int = 1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.segmenter = RoGLSNet(num_classes=num_classes, pretrained=pretrained)
        self.auxiliary_encoder = AuxiliaryPyramid(
            auxiliary_channels,
            self.auxiliary_widths,
        )
        self.fusion = nn.ModuleList(
            GatedFeatureFusion(auxiliary_width, optical_width)
            for auxiliary_width, optical_width in zip(
                self.auxiliary_widths,
                self.optical_channels,
                strict=True,
            )
        )

    def forward(self, primary: torch.Tensor, auxiliary: torch.Tensor):
        if primary.ndim != 4 or primary.shape[1] != 3:
            raise ValueError(f"primary must be BCHW with three bands, got {primary.shape}")
        if auxiliary.ndim != 4 or auxiliary.shape[-2:] != primary.shape[-2:]:
            raise ValueError("auxiliary must be BCHW and spatially aligned with primary")
        optical_features = self.segmenter.backbone(primary)
        auxiliary_features = self.auxiliary_encoder(auxiliary)
        fused_features = tuple(
            fusion(optical, extra)
            for fusion, optical, extra in zip(
                self.fusion,
                optical_features,
                auxiliary_features,
                strict=True,
            )
        )
        return self.segmenter.decode_features(fused_features, primary.shape[-2:])

