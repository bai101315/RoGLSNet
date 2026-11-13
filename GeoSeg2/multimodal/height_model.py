"""Height-selective RoGLSNet that preserves optical RoAFM queries and keys."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from geoseg.models.RoGLSNet import RoGLSNet


def _group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            _group_norm(out_channels),
            nn.GELU(),
        )


class HeightStructureBlock(nn.Module):
    """Combine height context, local contrast, and learned discontinuities."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.context = ConvGNAct(channels, channels, kernel_size=7, groups=channels)
        self.discontinuity = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _group_norm(channels),
            nn.Sigmoid(),
        )
        self.project = ConvGNAct(channels, channels, kernel_size=3, groups=channels)

    def forward(self, height: torch.Tensor) -> torch.Tensor:
        context = self.context(height)
        contrast = (height - F.avg_pool2d(height, 3, stride=1, padding=1)).abs()
        discontinuity = self.discontinuity(contrast) * height
        return height + self.project(context + contrast + discontinuity)


class HeightStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            ConvGNAct(in_channels, in_channels, stride=2, groups=in_channels),
            ConvGNAct(in_channels, out_channels, kernel_size=1),
        )
        self.structure = HeightStructureBlock(out_channels)

    def forward(self, height: torch.Tensor) -> torch.Tensor:
        return self.structure(self.downsample(height))


class HeightEncoder(nn.Module):
    widths = (32, 64, 96, 128)

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = ConvGNAct(in_channels, self.widths[0], stride=2)
        self.stage1 = HeightStage(self.widths[0], self.widths[0])
        self.stage2 = HeightStage(self.widths[0], self.widths[1])
        self.stage3 = HeightStage(self.widths[1], self.widths[2])
        self.stage4 = HeightStage(self.widths[2], self.widths[3])

    def forward(self, height: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height = self.stem(height)
        height1 = self.stage1(height)
        height2 = self.stage2(height1)
        height3 = self.stage3(height2)
        height4 = self.stage4(height3)
        return height1, height2, height4


class BoundedHeightFusion(nn.Module):
    """A gated residual whose magnitude cannot overwhelm the optical path."""

    def __init__(self, height_channels: int, optical_channels: int) -> None:
        super().__init__()
        self.height_project = nn.Sequential(
            nn.Conv2d(height_channels, optical_channels, kernel_size=1, bias=False),
            _group_norm(optical_channels),
            nn.GELU(),
        )
        hidden = max(16, optical_channels // 4)
        self.gate = nn.Sequential(
            nn.Conv2d(optical_channels * 2, hidden, kernel_size=1, bias=False),
            _group_norm(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, optical_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        initial_probability = 0.1
        self.scale_logit = nn.Parameter(
            torch.full(
                (1, optical_channels, 1, 1),
                math.log(initial_probability / (1.0 - initial_probability)),
            )
        )

    def forward(self, optical: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
        projected = self.height_project(height)
        if projected.shape[-2:] != optical.shape[-2:]:
            projected = F.interpolate(
                projected,
                size=optical.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        gate = self.gate(torch.cat((optical, projected), dim=1))
        strength = 0.5 * self.scale_logit.sigmoid()
        return optical + strength * gate * projected


class BoundaryHeightFusion(nn.Module):
    def __init__(self, height_channels: int, optical_channels: int) -> None:
        super().__init__()
        self.edge_head = nn.Sequential(
            ConvGNAct(height_channels, height_channels, kernel_size=3, groups=height_channels),
            nn.Conv2d(height_channels, 1, kernel_size=1),
        )
        self.height_project = nn.Sequential(
            nn.Conv2d(height_channels, optical_channels, kernel_size=1, bias=False),
            _group_norm(optical_channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(optical_channels * 2 + 1, optical_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        initial_probability = 0.1
        self.scale_logit = nn.Parameter(
            torch.full(
                (1, optical_channels, 1, 1),
                math.log(initial_probability / (1.0 - initial_probability)),
            )
        )

    def forward(
        self,
        optical: torch.Tensor,
        height: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if height.shape[-2:] != optical.shape[-2:]:
            height = F.interpolate(height, optical.shape[-2:], mode="bilinear", align_corners=False)
        edge_logits = self.edge_head(height)
        projected = self.height_project(height)
        edge_probability = edge_logits.sigmoid()
        gate = self.gate(torch.cat((optical, projected, edge_probability), dim=1))
        strength = 0.5 * self.scale_logit.sigmoid()
        return optical + strength * gate * projected, edge_logits, height


class ClassifierConditionedRefinement(nn.Module):
    def __init__(self, height_channels: int, num_classes: int) -> None:
        super().__init__()
        hidden = 64
        self.body = nn.Sequential(
            ConvGNAct(height_channels + num_classes + 1, hidden),
            ConvGNAct(hidden, hidden, groups=hidden),
        )
        self.classifier = nn.Conv2d(hidden, num_classes, kernel_size=1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        initial_probability = 0.1
        self.scale_logit = nn.Parameter(
            torch.tensor(math.log(initial_probability / (1.0 - initial_probability)))
        )

    def forward(
        self,
        logits: torch.Tensor,
        height: torch.Tensor,
        edge_logits: torch.Tensor,
    ) -> torch.Tensor:
        size = height.shape[-2:]
        probabilities = F.interpolate(
            logits.softmax(dim=1), size=size, mode="bilinear", align_corners=False
        )
        edge = edge_logits.sigmoid()
        residual = self.classifier(self.body(torch.cat((probabilities, height, edge), dim=1)))
        residual = F.interpolate(residual, logits.shape[-2:], mode="bilinear", align_corners=False)
        return logits + 0.5 * self.scale_logit.sigmoid() * residual


class HeightSelectiveRoGLSNet(nn.Module):
    """Use DSM as global value, post-attention geometry, and boundary evidence."""

    def __init__(
        self,
        *,
        num_classes: int,
        auxiliary_channels: int = 1,
        pretrained: bool = True,
        output_refinement: bool = False,
    ) -> None:
        super().__init__()
        self.segmenter = RoGLSNet(num_classes=num_classes, pretrained=pretrained)
        self.height_encoder = HeightEncoder(auxiliary_channels)
        self.global_fusion = BoundedHeightFusion(128, 64)
        self.semantic_fusion = BoundedHeightFusion(64, 64)
        self.boundary_fusion = BoundaryHeightFusion(32, 64)
        self.output_refinement = (
            ClassifierConditionedRefinement(32, num_classes)
            if output_refinement
            else None
        )

    def forward(self, primary: torch.Tensor, auxiliary: torch.Tensor):
        if primary.ndim != 4 or primary.shape[1] != 3:
            raise ValueError(f"primary must be BCHW IRRG, got {tuple(primary.shape)}")
        if auxiliary.ndim != 4 or auxiliary.shape[-2:] != primary.shape[-2:]:
            raise ValueError("DSM must be BCHW and aligned with IRRG")

        optical1, optical2, optical3, optical4 = self.segmenter.backbone(primary)
        height1, height2, height4 = self.height_encoder(auxiliary)

        res1 = self.segmenter.pre_conv1(optical1)
        res2 = self.segmenter.pre_conv2(optical2)
        res3 = self.segmenter.pre_conv3(optical3)
        res4 = self.segmenter.pre_conv4(optical4)

        # RoAFM queries and keys are built from untouched optical features.
        global_scene, semantic_aux = self.segmenter.gca(res4)
        global_scene = self.global_fusion(global_scene, height4)

        local_scene = torch.cat(
            (
                res2,
                self.segmenter.up2(res3),
                self.segmenter.up4(res4),
            ),
            dim=1,
        )
        local_scene = self.segmenter.CLC3x3_1(local_scene)
        attended = self.segmenter.fusion(local_scene, global_scene)

        global_up = self.segmenter.up4(global_scene)
        semantic = self.segmenter.CLC3x3_2(torch.cat((global_up, attended), dim=1))
        semantic = self.semantic_fusion(semantic, height2)

        decoded = self.segmenter.wf(semantic, res1)
        decoded, boundary_logits, height_boundary = self.boundary_fusion(decoded, height1)

        shortcut = self.segmenter.shortcut(decoded)
        decoded = self.segmenter.pa(decoded) * decoded + self.segmenter.ca(decoded) * decoded
        decoded = self.segmenter.act(self.segmenter.proj(decoded) + shortcut)
        decoded = F.interpolate(
            decoded,
            size=primary.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        logits = self.segmenter.segmentation_head(decoded)
        if self.output_refinement is not None:
            logits = self.output_refinement(logits, height_boundary, boundary_logits)

        if self.training:
            return logits, semantic_aux, boundary_logits
        return logits

    def fusion_strengths(self) -> dict[str, float]:
        result = {
            "global": float((0.5 * self.global_fusion.scale_logit.sigmoid()).mean()),
            "semantic": float((0.5 * self.semantic_fusion.scale_logit.sigmoid()).mean()),
            "boundary": float((0.5 * self.boundary_fusion.scale_logit.sigmoid()).mean()),
        }
        if self.output_refinement is not None:
            result["refinement"] = float(0.5 * self.output_refinement.scale_logit.sigmoid())
        return result
