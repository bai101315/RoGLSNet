"""Optimizer helpers for the RoGLSNet paper recipe."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def parameter_groups(
    model: nn.Module,
    *,
    learning_rate: float,
    backbone_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        learning_rate_for_parameter = (
            backbone_learning_rate
            if name.startswith("segmenter.backbone.")
            else learning_rate
        )
        decay = 0.0 if name.endswith("bias") else weight_decay
        grouped.setdefault((learning_rate_for_parameter, decay), []).append(parameter)
    return [
        {"params": parameters, "lr": group_lr, "weight_decay": decay}
        for (group_lr, decay), parameters in grouped.items()
    ]


class Lookahead(Optimizer):
    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5) -> None:
        if k < 1 or not 0.0 <= alpha <= 1.0:
            raise ValueError("invalid Lookahead configuration")
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups
        self.defaults = optimizer.defaults
        self.state = optimizer.state
        self._slow_weights = [
            parameter.detach().clone()
            for group in self.param_groups
            for parameter in group["params"]
        ]
        self._lookahead_step = 0

    def _parameters(self) -> Iterable[nn.Parameter]:
        for group in self.param_groups:
            yield from group["params"]

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = self.optimizer.step(closure)
        self._lookahead_step += 1
        if self._lookahead_step % self.k == 0:
            for parameter, slow in zip(self._parameters(), self._slow_weights, strict=True):
                slow.add_(parameter.data - slow, alpha=self.alpha)
                parameter.data.copy_(slow)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.optimizer.state_dict(),
            "slow_weights": [weight.detach().cpu() for weight in self._slow_weights],
            "lookahead_step": self._lookahead_step,
            "k": self.k,
            "alpha": self.alpha,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if int(state_dict["k"]) != self.k or float(state_dict["alpha"]) != self.alpha:
            raise ValueError("Lookahead configuration differs from the checkpoint")
        self.optimizer.load_state_dict(state_dict["base_optimizer"])
        saved = state_dict["slow_weights"]
        if len(saved) != len(self._slow_weights):
            raise ValueError("Lookahead parameter count differs from the checkpoint")
        for current, restored in zip(self._slow_weights, saved, strict=True):
            current.copy_(restored.to(device=current.device, dtype=current.dtype))
        self._lookahead_step = int(state_dict["lookahead_step"])
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

