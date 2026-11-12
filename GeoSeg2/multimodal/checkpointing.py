"""Atomic full-state checkpoints for exact continuation."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    epoch: int,
    best_mf1: float,
    best_epoch: int,
    best_metrics: dict[str, float],
    logs: list[dict[str, float | int]],
    elapsed_seconds: float,
    protocol: dict[str, Any],
) -> None:
    atomic_torch_save(
        {
            "format_version": 2,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loader_generator_state": loader_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
            "python_random_state": random.getstate(),
            "best_mf1": best_mf1,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "logs": logs,
            "elapsed_seconds": elapsed_seconds,
            "protocol": protocol,
        },
        path,
    )


def load_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 2:
        raise ValueError("unsupported checkpoint format")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    loader_generator.set_state(payload["loader_generator_state"].cpu())
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload["cuda_rng_state_all"]:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in payload["cuda_rng_state_all"]]
        )
    random.setstate(payload["python_random_state"])
    return payload

