"""Streaming Vaihingen semantic-segmentation metrics."""

from __future__ import annotations

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prediction = logits.argmax(dim=1).detach().cpu().reshape(-1)
        target = target.detach().cpu().reshape(-1)
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        encoded = target[valid] * self.num_classes + prediction[valid]
        self.confusion += torch.bincount(
            encoded,
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float]:
        matrix = self.confusion.float()
        true_positive = matrix.diag()
        support = matrix.sum(1)
        predicted = matrix.sum(0)
        union = support + predicted - true_positive
        denominator_f1 = support + predicted
        # The paper reports means over the five foreground classes. Clutter is
        # included in OA but excluded from mIoU and mF1.
        reported = torch.arange(self.num_classes) < self.num_classes - 1
        valid_iou = reported & (union > 0)
        valid_f1 = reported & (denominator_f1 > 0)
        total = matrix.sum()
        return {
            "miou": float((true_positive[valid_iou] / union[valid_iou]).mean().item()),
            "mf1": float(
                (2 * true_positive[valid_f1] / denominator_f1[valid_f1]).mean().item()
            ),
            "oa": float((true_positive.sum() / total).item()) if total > 0 else 0.0,
        }

    def per_class(self, class_names: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
        if len(class_names) != self.num_classes:
            raise ValueError("class_names needs one entry per class")
        matrix = self.confusion.float()
        true_positive = matrix.diag()
        support = matrix.sum(1)
        predicted = matrix.sum(0)
        union = support + predicted - true_positive
        denominator_f1 = support + predicted
        return {
            name: {
                "iou": float(true_positive[index] / union[index]) if union[index] > 0 else 0.0,
                "f1": float(2 * true_positive[index] / denominator_f1[index])
                if denominator_f1[index] > 0
                else 0.0,
                "support": int(support[index].item()),
            }
            for index, name in enumerate(class_names)
        }

