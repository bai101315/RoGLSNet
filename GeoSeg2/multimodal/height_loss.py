"""Segmentation, semantic auxiliary, and height-boundary objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .loss import RoGLSNetLoss


class HeightSelectiveLoss(nn.Module):
    def __init__(
        self,
        ignore_index: int,
        *,
        auxiliary_weight: float = 0.4,
        boundary_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.boundary_weight = boundary_weight
        self.segmentation = RoGLSNetLoss(
            ignore_index,
            smoothing=0.0,
            auxiliary_weight=auxiliary_weight,
        )

    def _boundary_target(
        self,
        target: torch.Tensor,
        size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = F.interpolate(
            target.unsqueeze(1).float(), size=size, mode="nearest"
        ).squeeze(1).long()
        valid = labels != self.ignore_index
        boundary = torch.zeros_like(labels, dtype=torch.bool)
        horizontal = valid[:, :, 1:] & valid[:, :, :-1]
        horizontal &= labels[:, :, 1:] != labels[:, :, :-1]
        boundary[:, :, 1:] |= horizontal
        boundary[:, :, :-1] |= horizontal
        vertical = valid[:, 1:, :] & valid[:, :-1, :]
        vertical &= labels[:, 1:, :] != labels[:, :-1, :]
        boundary[:, 1:, :] |= vertical
        boundary[:, :-1, :] |= vertical
        return boundary.unsqueeze(1).float(), valid.unsqueeze(1)

    def _boundary_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        boundary, valid = self._boundary_target(target, logits.shape[-2:])
        if not valid.any():
            return logits.sum() * 0.0
        positives = boundary[valid].sum()
        negatives = valid.sum() - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(max=20.0)
        bce = F.binary_cross_entropy_with_logits(
            logits[valid],
            boundary[valid],
            pos_weight=pos_weight,
        )
        probability = logits.sigmoid() * valid
        boundary = boundary * valid
        intersection = (probability * boundary).sum()
        dice = 1.0 - (2.0 * intersection + 1.0) / (
            probability.sum() + boundary.sum() + 1.0
        )
        return bce + dice

    def forward(
        self,
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(output, tuple) or len(output) != 3:
            raise ValueError("height-selective training output must contain three tensors")
        main_logits, semantic_aux, boundary_logits = output
        segmentation = self.segmentation((main_logits, semantic_aux), target)
        return segmentation + self.boundary_weight * self._boundary_loss(
            boundary_logits,
            target,
        )
