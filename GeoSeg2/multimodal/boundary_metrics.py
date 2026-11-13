"""Boundary F1 diagnostics for semantic segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .data import DatasetConfig
from .metrics import SegmentationMetrics


class BoundaryF1:
    """Accumulate one-vs-rest class-boundary F1 with a pixel tolerance."""

    def __init__(self, num_classes: int, ignore_index: int, tolerance: int = 3) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.tolerance = tolerance
        self.matched_prediction = torch.zeros(num_classes, dtype=torch.float64)
        self.prediction_total = torch.zeros(num_classes, dtype=torch.float64)
        self.matched_target = torch.zeros(num_classes, dtype=torch.float64)
        self.target_total = torch.zeros(num_classes, dtype=torch.float64)

    @staticmethod
    def _class_boundary(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        boundary = torch.zeros_like(mask, dtype=torch.bool)
        horizontal = (mask[..., :, 1:] != mask[..., :, :-1]) & (
            valid[..., :, 1:] & valid[..., :, :-1]
        )
        vertical = (mask[..., 1:, :] != mask[..., :-1, :]) & (
            valid[..., 1:, :] & valid[..., :-1, :]
        )
        boundary[..., :, 1:] |= horizontal
        boundary[..., :, :-1] |= horizontal
        boundary[..., 1:, :] |= vertical
        boundary[..., :-1, :] |= vertical
        return boundary

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = logits.argmax(dim=1)
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        kernel = 2 * self.tolerance + 1
        for class_index in range(self.num_classes - 1):
            predicted_boundary = self._class_boundary(prediction == class_index, valid)
            target_boundary = self._class_boundary(target == class_index, valid)
            predicted_dilated = F.max_pool2d(
                predicted_boundary[:, None].float(), kernel, stride=1, padding=self.tolerance
            )[:, 0].bool()
            target_dilated = F.max_pool2d(
                target_boundary[:, None].float(), kernel, stride=1, padding=self.tolerance
            )[:, 0].bool()
            self.matched_prediction[class_index] += (
                predicted_boundary & target_dilated
            ).sum().double().cpu()
            self.prediction_total[class_index] += predicted_boundary.sum().double().cpu()
            self.matched_target[class_index] += (
                target_boundary & predicted_dilated
            ).sum().double().cpu()
            self.target_total[class_index] += target_boundary.sum().double().cpu()

    def compute(self, class_names: tuple[str, ...]) -> dict[str, object]:
        if len(class_names) != self.num_classes:
            raise ValueError("class_names needs one entry per class")
        precision = torch.where(
            self.prediction_total > 0,
            self.matched_prediction / self.prediction_total,
            torch.zeros_like(self.prediction_total),
        )
        recall = torch.where(
            self.target_total > 0,
            self.matched_target / self.target_total,
            torch.zeros_like(self.target_total),
        )
        denominator = precision + recall
        scores = torch.where(
            denominator > 0,
            2.0 * precision * recall / denominator,
            torch.zeros_like(denominator),
        )
        reported = torch.arange(self.num_classes) < self.num_classes - 1
        available = reported & (self.target_total > 0)
        mean = float(scores[available].mean()) if torch.any(available) else 0.0
        return {
            "mean_bf1": mean,
            "tolerance_pixels": self.tolerance,
            "per_class": {
                name: float(scores[index])
                for index, name in enumerate(class_names[:-1])
            },
        }


@torch.no_grad()
def evaluate_with_boundary(
    model: nn.Module,
    loader: DataLoader,
    dataset_config: DatasetConfig,
    device: torch.device,
    amp_enabled: bool,
    *,
    zero_auxiliary: bool = False,
    tolerance: int = 3,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]], dict[str, object]]:
    model.eval()
    metrics = SegmentationMetrics(dataset_config.num_classes, dataset_config.ignore_index)
    boundaries = BoundaryF1(dataset_config.num_classes, dataset_config.ignore_index, tolerance)
    for batch in loader:
        primary = batch["primary"].to(device, non_blocking=True)
        auxiliary = batch["auxiliary"].to(device, non_blocking=True)
        if zero_auxiliary:
            auxiliary = torch.zeros_like(auxiliary)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(primary, auxiliary)
            logits = output[0] if isinstance(output, tuple) else output
        metrics.update(logits, target)
        boundaries.update(logits, target)
    return (
        metrics.compute(),
        metrics.per_class(dataset_config.class_names),
        boundaries.compute(dataset_config.class_names),
    )
