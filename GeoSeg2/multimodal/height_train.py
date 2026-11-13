"""Train and evaluate the height-selective Vaihingen model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .checkpointing import atomic_torch_save, load_training_checkpoint, save_training_checkpoint
from .data import DatasetConfig, PairedRasterDataset
from .height_loss import HeightSelectiveLoss
from .height_model import HeightSelectiveRoGLSNet
from .optim import Lookahead
from .train import evaluate_loader, make_loader, seed_everything, train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def apply_path_overrides(protocol: dict[str, Any]) -> dict[str, Any]:
    """Override machine-specific paths without changing a recorded protocol."""
    overrides = {
        "ROGLS_DATASET_MANIFEST": "dataset_manifest",
        "ROGLS_OUTPUT_DIR": "output_dir",
        "ROGLS_INITIAL_CHECKPOINT": "initial_checkpoint",
    }
    protocol = dict(protocol)
    for environment_name, key in overrides.items():
        value = os.environ.get(environment_name, "").strip()
        if value:
            protocol[key] = value
    return protocol


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def write_live_artifacts(
    output_dir: Path,
    logs: list[dict[str, Any]],
    *,
    total_epochs: int,
    state: str,
    elapsed_seconds: float,
    title: str = "HeightSelectiveRoGLSNet",
) -> None:
    completed = int(logs[-1]["epoch"]) if logs else 0
    average = elapsed_seconds / completed if completed else 0.0
    eta = average * (total_epochs - completed)
    estimated_end = datetime.now().astimezone() + timedelta(seconds=eta)
    status = {
        "state": state,
        "completed_epochs": completed,
        "total_epochs": total_epochs,
        "progress_percent": 100.0 * completed / total_epochs,
        "elapsed_seconds": elapsed_seconds,
        "average_epoch_seconds": average,
        "eta_seconds": eta,
        "estimated_end_time": estimated_end.isoformat(timespec="seconds"),
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "latest": logs[-1] if logs else None,
    }
    _atomic_write_text(output_dir / "live_status.json", json.dumps(status, indent=2, sort_keys=True))
    _atomic_write_text(output_dir / "history.json", json.dumps(logs, indent=2, sort_keys=True))

    scalar_fields = sorted(
        {
            key
            for record in logs
            for key, value in record.items()
            if isinstance(value, (int, float, str, bool)) or value is None
        }
    )
    preferred = [
        "epoch",
        "train_loss",
        "test_mf1",
        "test_miou",
        "test_oa",
        "val_mf1",
        "val_miou",
        "val_oa",
    ]
    fieldnames = [field for field in preferred if field in scalar_fields]
    fieldnames.extend(field for field in scalar_fields if field not in fieldnames)
    temporary_csv = output_dir / "history.csv.tmp"
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(logs)
    temporary_csv.replace(output_dir / "history.csv")

    latest = logs[-1] if logs else {}
    rows = [
        "| Epoch | Loss | Test mF1 | Test mIoU | Test OA | Val mF1 | Time |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in logs[-10:]:
        rows.append(
            "| {epoch} | {train_loss:.4f} | {test_mf1:.4f} | {test_miou:.4f} | "
            "{test_oa:.4f} | {val_mf1:.4f} | {duration} |".format(
                **record,
                duration=_format_duration(float(record["epoch_seconds"])),
            )
        )
    markdown = "\n".join(
        (
            f"# {title} 训练状态",
            "",
            f"- 状态：`{state}`",
            f"- 进度：**{completed}/{total_epochs}**（{status['progress_percent']:.1f}%）",
            f"- 最新官方测试 mF1：**{float(latest.get('test_mf1', 0.0)):.4f}**",
            f"- 最新官方测试 mIoU：**{float(latest.get('test_miou', 0.0)):.4f}**",
            f"- 最新 area30 mF1（仅记录）：**{float(latest.get('val_mf1', 0.0)):.4f}**",
            f"- 已运行：{_format_duration(elapsed_seconds)}",
            f"- 预计剩余：{_format_duration(eta)}",
            f"- 预计结束：{status['estimated_end_time']}",
            "",
            "## 最近 10 轮",
            "",
            *rows,
            "",
            "完整数据见 `history.csv` 和 `history.json`。",
            "",
        )
    )
    _atomic_write_text(output_dir / "live_status.md", markdown)


def build_model(protocol: dict[str, Any], dataset_config: DatasetConfig) -> HeightSelectiveRoGLSNet:
    stage = int(protocol["stage"])
    model = HeightSelectiveRoGLSNet(
        num_classes=dataset_config.num_classes,
        auxiliary_channels=dataset_config.auxiliary_channels,
        pretrained=bool(protocol.get("pretrained", stage == 1)),
        output_refinement=stage == 2,
    )
    initialization = str(protocol.get("initial_checkpoint", "")).strip()
    if initialization:
        state = torch.load(Path(initialization).resolve(), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if not isinstance(state, dict):
            raise ValueError("initial checkpoint does not contain a model state dictionary")
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = stage == 2 and all(name.startswith("output_refinement.") for name in missing)
        if unexpected or (missing and not allowed_missing):
            raise ValueError(
                f"incompatible initial checkpoint: missing={missing}, unexpected={unexpected}"
            )
        print(
            f"loaded stage initialization: {initialization}; missing={len(missing)}",
            flush=True,
        )
    return model


def build_optimizer(
    model: HeightSelectiveRoGLSNet,
    protocol: dict[str, Any],
) -> tuple[Lookahead, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts]:
    groups: dict[tuple[str, float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if name.startswith("segmenter.backbone."):
            group_name = "backbone"
            learning_rate = float(protocol["backbone_learning_rate"])
        elif name.startswith("output_refinement."):
            group_name = "refinement"
            learning_rate = float(protocol.get("refinement_learning_rate", protocol["learning_rate"]))
        else:
            group_name = "model"
            learning_rate = float(protocol["learning_rate"])
        decay = 0.0 if name.endswith("bias") else float(protocol["weight_decay"])
        groups.setdefault((group_name, learning_rate, decay), []).append(parameter)
    parameter_groups = [
        {"params": parameters, "lr": lr, "weight_decay": decay, "group_name": group_name}
        for (group_name, lr, decay), parameters in groups.items()
    ]
    base = torch.optim.AdamW(parameter_groups, lr=float(protocol["learning_rate"]))
    optimizer = Lookahead(
        base,
        k=int(protocol.get("lookahead_k", 5)),
        alpha=float(protocol.get("lookahead_alpha", 0.5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(protocol.get("scheduler_t0", 15)),
        T_mult=int(protocol.get("scheduler_t_mult", 2)),
    )
    return optimizer, scheduler


def apply_freeze_policy(model: HeightSelectiveRoGLSNet, *, stage: int, epoch: int, freeze_epochs: int) -> None:
    frozen = epoch <= freeze_epochs
    if stage == 1:
        for parameter in model.segmenter.backbone.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            model.segmenter.backbone.eval()
    else:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("output_refinement.") or not frozen)
        if frozen:
            for name, module in model.named_children():
                if name != "output_refinement":
                    module.eval()


class ShuffledAuxiliaryDataset(Dataset[dict[str, Any]]):
    def __init__(self, base: PairedRasterDataset) -> None:
        self.base = base
        size = len(base)
        self.permutation: list[int] = []
        stems = [sample[0] for sample in base.samples]
        areas = [stem.split("_", 1)[0] for stem in stems]
        for index, area in enumerate(areas):
            donor = (index + size // 2) % size
            while areas[donor] == area:
                donor = (donor + 1) % size
            self.permutation.append(donor)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        donor = self.base[self.permutation[index]]
        sample["auxiliary"] = donor["auxiliary"]
        sample["auxiliary_sample_id"] = donor["sample_id"]
        return sample


@torch.no_grad()
def evaluate_zero_auxiliary(
    model: nn.Module,
    loader: DataLoader,
    dataset_config: DatasetConfig,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    from .metrics import SegmentationMetrics

    model.eval()
    metrics = SegmentationMetrics(dataset_config.num_classes, dataset_config.ignore_index)
    for batch in loader:
        primary = batch["primary"].to(device, non_blocking=True)
        auxiliary = torch.zeros_like(batch["auxiliary"]).to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(primary, auxiliary)
        metrics.update(logits, target)
    return metrics.compute(), metrics.per_class(dataset_config.class_names)


def _preflight_once(
    protocol: dict[str, Any],
    dataset_config: DatasetConfig,
    *,
    batch_size: int,
) -> dict[str, Any]:
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
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        device=device,
        generator=generator,
    )
    model = build_model(protocol, dataset_config).to(device)
    criterion = HeightSelectiveLoss(
        dataset_config.ignore_index,
        auxiliary_weight=float(protocol["auxiliary_loss_weight"]),
        boundary_weight=float(protocol["boundary_loss_weight"]),
    )
    optimizer, scheduler = build_optimizer(model, protocol)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=bool(protocol["mixed_precision"]) and device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch = next(iter(loader))
    primary = batch["primary"].to(device)
    auxiliary = batch["auxiliary"].to(device)
    target = batch["mask"].to(device)
    amp_enabled = scaler.is_enabled()
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

    with tempfile.TemporaryDirectory(prefix="height-selective-preflight-") as temporary:
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
        restored = build_model(protocol, dataset_config).to(device)
        restored_optimizer, restored_scheduler = build_optimizer(restored, protocol)
        restored_scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
        restored_generator = torch.Generator()
        load_training_checkpoint(
            checkpoint,
            model=restored,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            scaler=restored_scaler,
            loader_generator=restored_generator,
            device=device,
        )
        for original, recovered in zip(
            model.state_dict().values(), restored.state_dict().values(), strict=True
        ):
            if not torch.equal(original, recovered):
                raise RuntimeError("checkpoint round-trip changed model parameters")

    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return {
        "device": str(device),
        "batch_size": batch_size,
        "sample_ids": list(batch["sample_id"]),
        "input_shape": list(primary.shape),
        "output_shape": list(output[0].shape),
        "loss": float(loss.detach()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_mib": peak,
        "checkpoint_round_trip": True,
        "fusion_strengths": model.fusion_strengths(),
    }


def run_preflight(protocol: dict[str, Any], dataset_config: DatasetConfig) -> None:
    seed_everything(int(protocol["seed"]))
    requested = int(protocol["batch_size"])
    attempts = [requested]
    fallback = int(protocol.get("fallback_batch_size", requested))
    if fallback != requested:
        attempts.append(fallback)
    errors: list[str] = []
    for batch_size in attempts:
        try:
            result = _preflight_once(protocol, dataset_config, batch_size=batch_size)
            result["recommended_accumulation_steps"] = (
                int(protocol["gradient_accumulation_steps"])
                if batch_size == requested
                else int(protocol["fallback_gradient_accumulation_steps"])
            )
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            return
        except torch.OutOfMemoryError as error:
            errors.append(f"batch_size={batch_size}: {error}")
            torch.cuda.empty_cache()
    raise RuntimeError("all preflight batch sizes failed: " + " | ".join(errors))


def _flatten_per_class(per_class: dict[str, dict[str, float | int]]) -> dict[str, float | int]:
    return {
        f"test_{class_name}_{metric}": value
        for class_name, values in per_class.items()
        for metric, value in values.items()
    }


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
    result_path = output_dir / "metrics.json"
    resume_path = args.resume.resolve() if args.resume else None
    if resume_path is None and result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed run: {result_path}")

    patch_size = int(protocol["patch_size"])
    train_set = PairedRasterDataset(dataset_config, "train", training=True, patch_size=patch_size)
    val_set = PairedRasterDataset(dataset_config, "val", training=False, patch_size=patch_size)
    test_set = PairedRasterDataset(dataset_config, "test", training=False, patch_size=patch_size)
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
        val_set, batch_size=1, num_workers=int(protocol["num_workers"]), shuffle=False, device=device
    )
    test_loader = make_loader(
        test_set, batch_size=1, num_workers=int(protocol["num_workers"]), shuffle=False, device=device
    )

    model = build_model(protocol, dataset_config).to(device)
    criterion = HeightSelectiveLoss(
        dataset_config.ignore_index,
        auxiliary_weight=float(protocol["auxiliary_loss_weight"]),
        boundary_weight=float(protocol["boundary_loss_weight"]),
    )
    optimizer, scheduler = build_optimizer(model, protocol)
    amp_enabled = bool(protocol["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latest_checkpoint = checkpoint_dir / "resume_checkpoint.pt"
    best_model_path = checkpoint_dir / "best_model.pt"
    start_epoch = 1
    best_mf1 = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    logs: list[dict[str, Any]] = []
    elapsed_before_resume = 0.0
    if resume_path:
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
        best_metrics = {key: float(value) for key, value in restored["best_metrics"].items()}
        logs = list(restored["logs"])
        elapsed_before_resume = float(restored["elapsed_seconds"])
        print(f"resumed after epoch {start_epoch - 1}", flush=True)

    total_epochs = int(protocol["epochs"])
    stage = int(protocol["stage"])
    freeze_epochs = int(protocol["freeze_epochs"])
    checkpoint_interval = int(protocol["checkpoint_interval"])
    accumulation_steps = int(protocol["gradient_accumulation_steps"])
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
        model.train()
        apply_freeze_policy(model, stage=stage, epoch=epoch, freeze_epochs=freeze_epochs)
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
            model, val_loader, dataset_config, device, amp_enabled, use_d4=False
        )
        benchmark, per_class = evaluate_loader(
            model, test_loader, dataset_config, device, amp_enabled, use_d4=False
        )
        scheduler.step()
        learning_rates: dict[str, float] = {}
        for group in optimizer.param_groups:
            key = str(group.get("group_name", "model")) + "_learning_rate"
            learning_rates[key] = max(learning_rates.get(key, 0.0), float(group["lr"]))
        epoch_seconds = time.monotonic() - epoch_started
        elapsed = elapsed_before_resume + time.monotonic() - started
        eta = elapsed / epoch * (total_epochs - epoch)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_miou": validation["miou"],
            "val_mf1": validation["mf1"],
            "val_oa": validation["oa"],
            "test_miou": benchmark["miou"],
            "test_mf1": benchmark["mf1"],
            "test_oa": benchmark["oa"],
            **_flatten_per_class(per_class),
            **learning_rates,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "estimated_end_time": (
                datetime.now().astimezone() + timedelta(seconds=eta)
            ).isoformat(timespec="seconds"),
            "fusion_strengths": model.fusion_strengths(),
        }
        logs.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        write_live_artifacts(
            output_dir,
            logs,
            total_epochs=total_epochs,
            state="running",
            elapsed_seconds=elapsed,
        )
        if benchmark["mf1"] > best_mf1:
            best_mf1 = benchmark["mf1"]
            best_epoch = epoch
            best_metrics = benchmark
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
                elapsed_seconds=elapsed,
                protocol=protocol,
            )

    runtime = elapsed_before_resume + time.monotonic() - started
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    normal_metrics, normal_per_class = evaluate_loader(
        model, test_loader, dataset_config, device, amp_enabled, use_d4=False
    )
    zero_metrics, zero_per_class = evaluate_zero_auxiliary(
        model, test_loader, dataset_config, device, amp_enabled
    )
    shuffled_set = ShuffledAuxiliaryDataset(test_set)
    shuffled_loader = make_loader(
        shuffled_set,
        batch_size=1,
        num_workers=int(protocol["num_workers"]),
        shuffle=False,
        device=device,
    )
    shuffled_metrics, shuffled_per_class = evaluate_loader(
        model, shuffled_loader, dataset_config, device, amp_enabled, use_d4=False
    )
    margin = normal_metrics["mf1"] - max(zero_metrics["mf1"], shuffled_metrics["mf1"])
    diagnostics = {
        "normal": {**normal_metrics, "per_class": normal_per_class},
        "zero": {**zero_metrics, "per_class": zero_per_class},
        "shuffled": {**shuffled_metrics, "per_class": shuffled_per_class},
        "mf1_margin_over_strongest_control": margin,
        "effective_at_0.002": margin >= 0.002,
    }
    _atomic_write_text(
        output_dir / "auxiliary_diagnostics.json",
        json.dumps(diagnostics, indent=2, sort_keys=True),
    )

    tta_started = time.monotonic()
    tta_metrics, tta_per_class = evaluate_loader(
        model, test_loader, dataset_config, device, amp_enabled, use_d4=True
    )
    final_test = {
        **tta_metrics,
        "per_class": tta_per_class,
        "tta": "released_d4_multiscale_40_view",
        "runtime_seconds": time.monotonic() - tta_started,
    }
    result = {
        "model": f"HeightSelectiveRoGLSNet stage {stage}",
        "selection_metric": "official_17_area_test_mf1_without_tta",
        "best_benchmark_test": {"epoch": best_epoch, **best_metrics},
        "area30_role": "logged_only_not_used_for_selection",
        "auxiliary_diagnostics": diagnostics,
        "final_test": final_test,
        "training_runtime_seconds": runtime,
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "fusion_strengths": model.fusion_strengths(),
        "paper_targets": {"oa": 0.9328, "mf1": 0.9131, "miou": 0.8431},
        "logs": logs,
    }
    temporary_result = result_path.with_suffix(".json.tmp")
    temporary_result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary_result.replace(result_path)
    from .summarize_results import write_comparison

    write_comparison(protocol)
    write_live_artifacts(
        output_dir,
        logs,
        total_epochs=total_epochs,
        state="completed",
        elapsed_seconds=runtime,
    )
    print(json.dumps({"best_benchmark_test": result["best_benchmark_test"], "final_test": final_test}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
