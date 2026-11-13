"""Standalone Vaihingen IRRG+DSM training and evaluation."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .checkpointing import (
    atomic_torch_save,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .data import DatasetConfig, PairedRasterDataset
from .input_fusion_model import InputFusionRoGLSNet
from .loss import RoGLSNetLoss
from .metrics import SegmentationMetrics
from .model import AsymmetricFusionRoGLSNet
from .optim import Lookahead, parameter_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("vaihingen_agf_1024.json"),
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="run one real forward/backward and checkpoint round-trip, then exit",
    )
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="override the config and run the official test split after training",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_live_artifacts(
    output_dir: Path,
    logs: list[dict[str, Any]],
    *,
    total_epochs: int,
    state: str,
    elapsed_seconds: float,
) -> None:
    """Write human-readable status and complete per-epoch history."""
    completed_epochs = int(logs[-1]["epoch"]) if logs else 0
    average_epoch_seconds = elapsed_seconds / completed_epochs if completed_epochs else 0.0
    eta_seconds = average_epoch_seconds * (total_epochs - completed_epochs)
    estimated_end = datetime.now().astimezone() + timedelta(seconds=eta_seconds)
    status = {
        "state": state,
        "completed_epochs": completed_epochs,
        "total_epochs": total_epochs,
        "progress_percent": 100.0 * completed_epochs / total_epochs,
        "elapsed_seconds": elapsed_seconds,
        "average_epoch_seconds": average_epoch_seconds,
        "eta_seconds": eta_seconds,
        "estimated_end_time": estimated_end.isoformat(timespec="seconds"),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "latest": logs[-1] if logs else None,
    }
    _atomic_write_text(
        output_dir / "live_status.json",
        json.dumps(status, indent=2, sort_keys=True),
    )
    _atomic_write_text(
        output_dir / "history.json",
        json.dumps(logs, indent=2, sort_keys=True),
    )

    fieldnames = (
        "epoch",
        "train_loss",
        "val_mf1",
        "val_miou",
        "val_oa",
        "learning_rate",
        "backbone_learning_rate",
        "epoch_seconds",
        "elapsed_seconds",
        "eta_seconds",
        "estimated_end_time",
    )
    csv_temporary = output_dir / "history.csv.tmp"
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(logs)
    csv_temporary.replace(output_dir / "history.csv")

    latest = logs[-1] if logs else {}
    table_rows = [
        "| Epoch | Train loss | Val mF1 | Val mIoU | Val OA | Time |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for record in logs[-10:]:
        table_rows.append(
            "| {epoch} | {train_loss:.4f} | {val_mf1:.4f} | {val_miou:.4f} | "
            "{val_oa:.4f} | {duration} |".format(
                **record,
                duration=_format_duration(float(record.get("epoch_seconds", 0.0))),
            )
        )
    markdown = "\n".join(
        (
            "# Vaihingen IRRG+DSM 训练状态",
            "",
            f"- 状态：`{state}`",
            f"- 进度：**{completed_epochs}/{total_epochs}** "
            f"（{status['progress_percent']:.1f}%）",
            f"- 最新 val mF1：**{float(latest.get('val_mf1', 0.0)):.4f}**",
            f"- 最新 val mIoU：**{float(latest.get('val_miou', 0.0)):.4f}**",
            f"- 最新 val OA：**{float(latest.get('val_oa', 0.0)):.4f}**",
            f"- 已运行：{_format_duration(elapsed_seconds)}",
            f"- 预计剩余：{_format_duration(eta_seconds)}",
            f"- 预计结束：{status['estimated_end_time']}",
            f"- 更新时间：{status['updated_at']}",
            "",
            "## 最近 10 轮",
            "",
            *table_rows,
            "",
            "完整逐轮数据见 `history.csv` 或 `history.json`。",
            "",
        )
    )
    _atomic_write_text(output_dir / "live_status.md", markdown)


def _main_logits(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def build_optimizer(
    model: nn.Module,
    protocol: dict[str, Any],
) -> tuple[Lookahead, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts]:
    base_optimizer = torch.optim.AdamW(
        parameter_groups(
            model,
            learning_rate=float(protocol["learning_rate"]),
            backbone_learning_rate=float(protocol["backbone_learning_rate"]),
            weight_decay=float(protocol["weight_decay"]),
        ),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    optimizer = Lookahead(
        base_optimizer,
        k=int(protocol.get("lookahead_k", 5)),
        alpha=float(protocol.get("lookahead_alpha", 0.5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(protocol.get("scheduler_t0", 15)),
        T_mult=int(protocol.get("scheduler_t_mult", 2)),
    )
    return optimizer, scheduler


def build_model(
    protocol: dict[str, Any],
    dataset_config: DatasetConfig,
    *,
    pretrained: bool,
) -> nn.Module:
    model_name = str(protocol.get("model", "four_stage"))
    model_classes = {
        "four_stage": AsymmetricFusionRoGLSNet,
        "input_fusion": InputFusionRoGLSNet,
    }
    if model_name not in model_classes:
        raise ValueError(f"unknown model: {model_name}")
    return model_classes[model_name](
        num_classes=dataset_config.num_classes,
        auxiliary_channels=dataset_config.auxiliary_channels,
        pretrained=pretrained,
    )


def apply_path_overrides(protocol: dict[str, Any]) -> dict[str, Any]:
    protocol = dict(protocol)
    overrides = {
        "ROGLS_DATASET_MANIFEST": "dataset_manifest",
        "ROGLS_OUTPUT_DIR": "output_dir",
        "ROGLS_INITIAL_CHECKPOINT": "initial_optical_checkpoint",
    }
    for environment_name, key in overrides.items():
        value = os.environ.get(environment_name, "").strip()
        if value:
            protocol[key] = value
    return protocol


def load_optical_initialization(model: nn.Module, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        state = state.get("state_dict", state.get("model_state_dict", state))
    if not isinstance(state, dict):
        raise ValueError("initial checkpoint does not contain a state dictionary")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("segmenter."):
            normalized[key.removeprefix("segmenter.")] = value
        elif not key.startswith(("auxiliary_adapter.", "gate.", "auxiliary_encoder.", "fusion.")):
            normalized[key.removeprefix("net.")] = value
    missing, unexpected = model.segmenter.load_state_dict(normalized, strict=False)
    loaded = len(model.segmenter.state_dict()) - len(missing)
    if loaded < 10:
        raise ValueError("initial checkpoint did not match RoGLSNet")
    print(
        f"optical initialization loaded: tensors={loaded}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}",
        flush=True,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    accumulation_steps: int,
    training_mode: Callable[[nn.Module], None] | None = None,
) -> float:
    model.train()
    if training_mode is not None:
        training_mode(model)
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    batch_count = len(loader)
    for batch_index, batch in enumerate(loader):
        primary = batch["primary"].to(device, non_blocking=True)
        auxiliary = batch["auxiliary"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, batch_count - group_start)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(primary, auxiliary)
            raw_loss = criterion(output, mask)
            loss = raw_loss / group_size
        if not torch.isfinite(raw_loss):
            sample_ids = ", ".join(str(value) for value in batch["sample_id"])
            raise FloatingPointError(f"non-finite loss for samples: {sample_ids}")
        scaler.scale(loss).backward()
        if (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == batch_count:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(raw_loss.detach()) * primary.shape[0]
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_d4(
    model: nn.Module,
    primary: torch.Tensor,
    auxiliary: torch.Tensor,
    amp_enabled: bool,
) -> torch.Tensor:
    original_size = primary.shape[-2:]
    accumulated: torch.Tensor | None = None
    count = 0
    for scale, horizontal, vertical, quarter_turn in itertools.product(
        (0.5, 0.75, 1.0, 1.25, 1.5),
        (False, True),
        (False, True),
        (0, 1),
    ):
        transformed_primary = primary
        transformed_auxiliary = auxiliary
        if scale != 1.0:
            size = tuple(max(1, round(value * scale)) for value in original_size)
            transformed_primary = F.interpolate(
                transformed_primary,
                size=size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            transformed_auxiliary = F.interpolate(
                transformed_auxiliary,
                size=size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        if horizontal:
            transformed_primary = transformed_primary.flip(-1)
            transformed_auxiliary = transformed_auxiliary.flip(-1)
        if vertical:
            transformed_primary = transformed_primary.flip(-2)
            transformed_auxiliary = transformed_auxiliary.flip(-2)
        if quarter_turn:
            transformed_primary = torch.rot90(transformed_primary, 1, (-2, -1))
            transformed_auxiliary = torch.rot90(transformed_auxiliary, 1, (-2, -1))
        with torch.autocast(device_type=primary.device.type, enabled=amp_enabled):
            logits = _main_logits(model(transformed_primary, transformed_auxiliary))
        if quarter_turn:
            logits = torch.rot90(logits, -1, (-2, -1))
        if vertical:
            logits = logits.flip(-2)
        if horizontal:
            logits = logits.flip(-1)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        logits = logits.float()
        accumulated = logits if accumulated is None else accumulated + logits
        count += 1
    if accumulated is None:
        raise RuntimeError("TTA produced no predictions")
    return accumulated / count


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    dataset_config: DatasetConfig,
    device: torch.device,
    amp_enabled: bool,
    *,
    use_d4: bool = False,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    model.eval()
    metrics = SegmentationMetrics(dataset_config.num_classes, dataset_config.ignore_index)
    for batch in loader:
        primary = batch["primary"].to(device, non_blocking=True)
        auxiliary = batch["auxiliary"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if use_d4:
            logits = predict_d4(model, primary, auxiliary, amp_enabled)
        else:
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = _main_logits(model(primary, auxiliary))
        metrics.update(logits, mask)
    return metrics.compute(), metrics.per_class(dataset_config.class_names)


def make_loader(
    dataset: PairedRasterDataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
    )


def run_preflight(protocol: dict[str, Any], dataset_config: DatasetConfig) -> None:
    seed_everything(int(protocol["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedRasterDataset(
        dataset_config,
        "train",
        training=True,
        patch_size=int(protocol["patch_size"]),
    )
    generator = torch.Generator().manual_seed(int(protocol["seed"]))
    loader = make_loader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=True,
        device=device,
        generator=generator,
    )
    model = build_model(
        protocol,
        dataset_config,
        pretrained=bool(protocol.get("pretrained", True)),
    ).to(device)
    criterion = RoGLSNetLoss(
        dataset_config.ignore_index,
        auxiliary_weight=float(protocol.get("auxiliary_loss_weight", 0.4)),
    )
    optimizer, scheduler = build_optimizer(model, protocol)
    amp_enabled = bool(protocol["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch = next(iter(loader))
    primary = batch["primary"].to(device)
    auxiliary = batch["auxiliary"].to(device)
    target = batch["mask"].to(device)
    with torch.autocast(device_type=device.type, enabled=amp_enabled):
        output = model(primary, auxiliary)
        loss = criterion(output, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("preflight produced a non-finite loss")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    with tempfile.TemporaryDirectory(prefix="roglsnet-preflight-") as temporary:
        checkpoint = Path(temporary) / "resume.pt"
        save_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=generator,
            epoch=1,
            best_mf1=0.0,
            best_epoch=1,
            best_metrics={"miou": 0.0, "mf1": 0.0, "oa": 0.0},
            logs=[{"epoch": 1, "train_loss": float(loss.detach())}],
            elapsed_seconds=0.0,
            protocol=protocol,
        )
        restored_model = build_model(
            protocol,
            dataset_config,
            pretrained=False,
        ).to(device)
        restored_optimizer, restored_scheduler = build_optimizer(restored_model, protocol)
        restored_scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
        restored_generator = torch.Generator()
        load_training_checkpoint(
            checkpoint,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            scaler=restored_scaler,
            loader_generator=restored_generator,
            device=device,
        )
        for original, restored in zip(
            model.state_dict().values(),
            restored_model.state_dict().values(),
            strict=True,
        ):
            if not torch.equal(original, restored):
                raise RuntimeError("checkpoint round-trip changed model parameters")

    main_logits = _main_logits(output)
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "sample_id": batch["sample_id"][0],
                "input_shape": list(primary.shape),
                "output_shape": list(main_logits.shape),
                "loss": float(loss.detach()),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "peak_gpu_memory_mib": peak_memory,
                "checkpoint_round_trip": True,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    protocol = apply_path_overrides(
        json.loads(args.config.resolve().read_text(encoding="utf-8"))
    )
    dataset_config = DatasetConfig.load(protocol["dataset_manifest"])
    if args.preflight:
        run_preflight(protocol, dataset_config)
        return

    seed = int(protocol["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(protocol["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume.resolve() if args.resume is not None else None
    result_path = output_dir / "metrics.json"
    if resume_path is None and result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed run: {result_path}")

    train_set = PairedRasterDataset(
        dataset_config,
        "train",
        training=True,
        patch_size=int(protocol["patch_size"]),
    )
    val_set = PairedRasterDataset(
        dataset_config,
        "val",
        training=False,
        patch_size=int(protocol["patch_size"]),
    )
    run_final_test = args.run_final_test or bool(protocol.get("run_final_test", False))
    test_set = (
        PairedRasterDataset(
            dataset_config,
            "test",
            training=False,
            patch_size=int(protocol["patch_size"]),
        )
        if run_final_test
        else None
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(
        train_set,
        batch_size=int(protocol["batch_size"]),
        num_workers=int(protocol["num_workers"]),
        shuffle=True,
        device=device,
        generator=generator,
    )
    val_loader = make_loader(
        val_set,
        batch_size=1,
        num_workers=int(protocol["num_workers"]),
        shuffle=False,
        device=device,
    )
    test_loader = (
        make_loader(
            test_set,
            batch_size=1,
            num_workers=int(protocol["num_workers"]),
            shuffle=False,
            device=device,
        )
        if test_set is not None
        else None
    )

    model = build_model(
        protocol,
        dataset_config,
        pretrained=bool(protocol.get("pretrained", True)),
    )
    initialization = str(protocol.get("initial_optical_checkpoint", "")).strip()
    if resume_path is None and initialization:
        load_optical_initialization(model, Path(initialization).resolve())
    model = model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    criterion = RoGLSNetLoss(
        dataset_config.ignore_index,
        auxiliary_weight=float(protocol.get("auxiliary_loss_weight", 0.4)),
    )
    optimizer, scheduler = build_optimizer(model, protocol)
    amp_enabled = bool(protocol["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    accumulation_steps = int(protocol["gradient_accumulation_steps"])
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")

    latest_checkpoint = checkpoint_dir / "resume_checkpoint.pt"
    best_model_path = checkpoint_dir / "best_model.pt"
    start_epoch = 1
    best_mf1 = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    logs: list[dict[str, Any]] = []
    elapsed_before_resume = 0.0
    if resume_path is not None:
        restored = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=generator,
            device=device,
        )
        if restored["protocol"] != protocol:
            raise ValueError("resume checkpoint uses a different experiment configuration")
        start_epoch = int(restored["epoch"]) + 1
        best_mf1 = float(restored["best_mf1"])
        best_epoch = int(restored["best_epoch"])
        best_metrics = {
            key: float(value)
            for key, value in restored["best_metrics"].items()
        }
        logs = list(restored["logs"])
        elapsed_before_resume = float(restored["elapsed_seconds"])
        print(f"resumed after epoch {start_epoch - 1}", flush=True)

    total_epochs = int(protocol["epochs"])
    checkpoint_interval = int(protocol["checkpoint_interval"])
    started = time.monotonic()
    write_live_artifacts(
        output_dir,
        logs,
        total_epochs=total_epochs,
        state="running",
        elapsed_seconds=elapsed_before_resume,
    )
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_started = time.monotonic()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            amp_enabled,
            accumulation_steps,
        )
        validation, _ = evaluate_loader(
            model,
            val_loader,
            dataset_config,
            device,
            amp_enabled,
        )
        scheduler.step()
        learning_rates = scheduler.get_last_lr()
        epoch_seconds = time.monotonic() - epoch_started
        elapsed_seconds = elapsed_before_resume + time.monotonic() - started
        average_epoch_seconds = elapsed_seconds / epoch
        eta_seconds = average_epoch_seconds * (total_epochs - epoch)
        estimated_end = datetime.now().astimezone() + timedelta(seconds=eta_seconds)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_miou": validation["miou"],
            "val_mf1": validation["mf1"],
            "val_oa": validation["oa"],
            "learning_rate": max(learning_rates),
            "backbone_learning_rate": min(learning_rates),
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed_seconds,
            "eta_seconds": eta_seconds,
            "estimated_end_time": estimated_end.isoformat(timespec="seconds"),
        }
        logs.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        write_live_artifacts(
            output_dir,
            logs,
            total_epochs=total_epochs,
            state="running",
            elapsed_seconds=elapsed_seconds,
        )
        if validation["mf1"] > best_mf1:
            best_mf1 = validation["mf1"]
            best_epoch = epoch
            best_metrics = validation
            atomic_torch_save(model.state_dict(), best_model_path)
        if epoch % checkpoint_interval == 0 or epoch == total_epochs:
            save_training_checkpoint(
                latest_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loader_generator=generator,
                epoch=epoch,
                best_mf1=best_mf1,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                logs=logs,
                elapsed_seconds=elapsed_seconds,
                protocol=protocol,
            )

    training_runtime = elapsed_before_resume + time.monotonic() - started
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    final_test: dict[str, Any] | None = None
    if test_loader is not None:
        test_started = time.monotonic()
        test_metrics, per_class = evaluate_loader(
            model,
            test_loader,
            dataset_config,
            device,
            amp_enabled,
            use_d4=True,
        )
        final_test = {
            **test_metrics,
            "per_class": per_class,
            "tta": "released_d4_multiscale",
            "runtime_seconds": time.monotonic() - test_started,
        }
    peak_gpu_memory = (
        torch.cuda.max_memory_allocated(device) / 1024**2
        if device.type == "cuda"
        else 0.0
    )
    result = {
        "model": "RoGLSNet asymmetric four-stage gated IRRG+DSM fusion",
        "best_validation": {"epoch": best_epoch, **best_metrics},
        "final_test": final_test,
        "training_runtime_seconds": training_runtime,
        "peak_gpu_memory_mib": peak_gpu_memory,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "logs": logs,
    }
    temporary_result = result_path.with_suffix(".json.tmp")
    temporary_result.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_result.replace(result_path)
    write_live_artifacts(
        output_dir,
        logs,
        total_epochs=total_epochs,
        state="completed",
        elapsed_seconds=training_runtime,
    )
    print(json.dumps(result["best_validation"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
