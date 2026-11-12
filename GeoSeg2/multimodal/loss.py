"""Numerically safe RoGLSNet segmentation objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RoGLSNetLoss(nn.Module):
    """Main CE+Dice objective plus 0.4 weighted auxiliary CE."""

    def __init__(
        self,
        ignore_index: int,
        smoothing: float = 0.05,
        auxiliary_weight: float = 0.4,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        self.auxiliary_weight = auxiliary_weight

    def _cross_entropy(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not torch.any(target != self.ignore_index):
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index,
            label_smoothing=self.smoothing,
        )

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0
        safe_target = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2)
        one_hot = one_hot.to(dtype=logits.dtype)
        valid = valid.unsqueeze(1)
        probabilities = logits.softmax(dim=1) * valid
        one_hot = one_hot * valid
        dimensions = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dim=dimensions)
        cardinality = probabilities.sum(dim=dimensions) + one_hot.sum(dim=dimensions)
        present = one_hot.sum(dim=dimensions) > 0
        scores = (2.0 * intersection + 0.05) / (cardinality + 0.05)
        return (1.0 - scores[present]).mean()

    def forward(
        self,
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(output, tuple):
            main_logits, auxiliary_logits = output
            auxiliary_target = F.interpolate(
                target.unsqueeze(1).float(),
                size=auxiliary_logits.shape[-2:],
                mode="nearest",
            ).squeeze(1).long()
            main_loss = self._cross_entropy(main_logits, target)
            main_loss = main_loss + self._dice_loss(main_logits, target)
            return main_loss + self.auxiliary_weight * self._cross_entropy(
                auxiliary_logits,
                auxiliary_target,
            )
        return self._cross_entropy(output, target) + self._dice_loss(output, target)

