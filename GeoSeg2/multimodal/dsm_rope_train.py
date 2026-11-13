"""Fine-tune the minimal RoAFM relative-height model on Vaihingen."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .boundary_metrics import evaluate_with_boundary
from .checkpointing import atomic_torch_save, load_training_checkpoint, save_training_checkpoint
from .data import DatasetConfig, PairedRasterDataset
from .dsm_rope_model import DSMRelativeRoGLSNet, load_input_fusion_segmenter
from .height_train import (
    ShuffledAuxiliaryDataset,
    _atomic_write_text,
    _flatten_per_class,
    apply_path_overrides,
    write_live_artifacts,
)
from .loss import RoGLSNetLoss
from .optim import Lookahead
from .train import evaluate_loader, make_loader, seed_everything, train_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("vaihingen_dsm_rope_1024.json"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def phase_for_epoch(epoch: int, freeze_epochs: int) -> int:
    return 1 if epoch <= freeze_epochs else 2


def _decoder_modules(model: DSMRelativeRoGLSNet) -> tuple[nn.Module, ...]:
    segmenter = model.segmenter
    return (
        segmenter.fusion,
        segmenter.CLC3x3_2,
        segmenter.wf,
        segmenter.shortcut,
        segmenter.pa,
        segmenter.ca,
        segmenter.proj,
        segmenter.segmentation_head,
    )


def configure_phase(model: DSMRelativeRoGLSNet, phase: int) -> None:
    if phase not in (1, 2):
        raise ValueError(f"unknown training phase: {phase}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.height_strength_logit.requires_grad_(True)
    if phase == 2:
        for module in _decoder_modules(model):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def apply_training_mode(model: DSMRelativeRoGLSNet, phase: int) -> None:
    """Keep frozen BatchNorm statistics fixed while retaining the auxiliary output."""
    model.segmenter.train()
    for child in model.segmenter.children():
        child.eval()
    if phase == 2:
        for module in _decoder_modules(model):
            module.train()


def build_model(protocol: dict[str, Any], dataset_config: DatasetConfig) -> DSMRelativeRoGLSNet:
    if str(protocol.get("model", "dsm_relative_rope")) != "dsm_relative_rope":
        raise ValueError("dsm_rope_train requires model=dsm_relative_rope")
    model = DSMRelativeRoGLSNet(
        num_classes=dataset_config.num_classes,
        auxiliary_channels=dataset_config.auxiliary_channels,
        pretrained=bool(protocol.get("pretrained", False)),
    )
    initialization = str(protocol.get("initial_checkpoint", "")).strip()
    if not initialization:
        raise ValueError("initial_checkpoint must point to the input-fusion best model")
    load_input_fusion_segmenter(model, initialization)
    return model


def validate_dataset_statistics(protocol: dict[str, Any], config: DatasetConfig) -> None:
    expected_mean = float(protocol.get("expected_dsm_mean", 285.06334936404653))
    expected_std = float(protocol.get("expected_dsm_std", 23.81859142054515))
    if len(config.auxiliary_mean) != 1 or len(config.auxiliary_std) != 1:
        raise ValueError("relative-height training requires one DSM channel")
    if abs(config.auxiliary_mean[0] - expected_mean) > 1e-3:
        raise ValueError("dataset does not use the expected raw-DSM training mean")
    if abs(config.auxiliary_std[0] - expected_std) > 1e-3:
        raise ValueError("dataset does not use the expected raw-DSM training standard deviation")


def build_optimizer(
    model: DSMRelativeRoGLSNet,
    protocol: dict[str, Any],
    phase: int,
) -> tuple[Lookahead, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts]:
    grouped: dict[tuple[str, float, float], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "height_strength_logit":
            group_name = "height"
            learning_rate = float(protocol["height_learning_rate"])
            decay = 0.0
        else:
            group_name = "decoder"
            learning_rate = float(protocol["decoder_learning_rate"])
            decay = 0.0 if name.endswith("bias") else float(protocol["weight_decay"])
        grouped.setdefault((group_name, learning_rate, decay), []).append(parameter)
    parameter_groups = [
        {"params": parameters, "lr": lr, "weight_decay": decay, "group_name": group_name}
        for (group_name, lr, decay), parameters in grouped.items()
    ]
    if phase == 1 and {group["group_name"] for group in parameter_groups} != {"height"}:
        raise RuntimeError("phase 1 must optimize only the height strength")
    base = torch.optim.AdamW(parameter_groups, lr=float(protocol["height_learning_rate"]))
    optimizer = Lookahead(
        base,
        k=int(protocol.get("lookahead_k", 5)),
        alpha=float(protocol.get("lookahead_alpha", 0.5)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(protocol.get("scheduler_t0", 10)),
        T_mult=int(protocol.get("scheduler_t_mult", 2)),
    )
    return optimizer, scheduler


def _main_logits(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _checkpoint_round_trip(
    path: Path,
    *,
    model: DSMRelativeRoGLSNet,
    optimizer: Lookahead,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
    protocol: dict[str, Any],
    dataset_config: DatasetConfig,
    phase: int,
    device: torch.device,
) -> None:
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loader_generator=generator,
        epoch=5 if phase == 1 else 6,
        best_mf1=0.0,
        best_epoch=0,
        best_metrics={"miou": 0.0, "mf1": 0.0, "oa": 0.0},
        logs=[],
        elapsed_seconds=0.0,
        protocol=protocol,
        extra_state={"phase": phase, "height_strength": float(model.height_strength())},
    )
    restored = build_model(protocol, dataset_config).to(device)
    configure_phase(restored, phase)
    restored_optimizer, restored_scheduler = build_optimizer(restored, protocol, phase)
    restored_scaler = torch.amp.GradScaler(device.type, enabled=scaler.is_enabled())
    restored_generator = torch.Generator()
    payload = load_training_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        loader_generator=restored_generator,
        device=device,
    )
    if int(payload["extra_state"]["phase"]) != phase:
        raise RuntimeError("checkpoint lost the training phase")
    for name, value in model.state_dict().items():
        if not torch.equal(value, restored.state_dict()[name]):
            raise RuntimeError(f"checkpoint round-trip changed {name}")


def _preflight_once(
    protocol: dict[str, Any],
    dataset_config: DatasetConfig,
    *,
    batch_size: int,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedRasterDataset(
        dataset_config, "train", training=True, patch_size=int(protocol["patch_size"])
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
    criterion = RoGLSNetLoss(
        dataset_config.ignore_index,
        auxiliary_weight=float(protocol["auxiliary_loss_weight"]),
    )
    amp_enabled = bool(protocol["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    batch = next(iter(loader))
    primary = batch["primary"].to(device)
    auxiliary = batch["auxiliary"].to(device)
    target = batch["mask"].to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp_enabled):
        optical = model.segmenter(primary)
        zero_height = model(primary, torch.zeros_like(auxiliary))
    parity_error = float((_main_logits(optical) - _main_logits(zero_height)).abs().max())
    if parity_error > 1e-6:
        raise RuntimeError(f"constant-height optical parity failed: {parity_error}")

    losses: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix="dsm-rope-preflight-") as temporary:
        for phase in (1, 2):
            configure_phase(model, phase)
            optimizer, scheduler = build_optimizer(model, protocol, phase)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            apply_training_mode(model, phase)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(primary, auxiliary)
                loss = criterion(output, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"phase {phase} produced a non-finite loss")
            scaler.scale(loss).backward()
            if phase == 1:
                gradient = model.height_strength_logit.grad
                if gradient is None or not torch.isfinite(gradient) or float(gradient.abs()) == 0.0:
                    raise RuntimeError("segmentation loss did not reach height_strength_logit")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses[f"phase_{phase}"] = float(loss.detach())
            _checkpoint_round_trip(
                Path(temporary) / f"phase_{phase}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                generator=generator,
                protocol=protocol,
                dataset_config=dataset_config,
                phase=phase,
                device=device,
            )

    return {
        "device": str(device),
        "batch_size": batch_size,
        "input_shape": list(primary.shape),
        "output_shape": list(_main_logits(output).shape),
        "losses": losses,
        "constant_height_parity_max_error": parity_error,
        "height_strength": float(model.height_strength()),
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
        "phase_1_and_2_checkpoint_round_trip": True,
    }


def run_preflight(protocol: dict[str, Any], dataset_config: DatasetConfig) -> None:
    seed_everything(int(protocol["seed"]))
    requested = int(protocol["batch_size"])
    fallback = int(protocol.get("fallback_batch_size", requested))
    attempts = [requested] + ([fallback] if fallback != requested else [])
    errors: list[str] = []
    for batch_size in attempts:
        try:
            result = _preflight_once(protocol, dataset_config, batch_size=batch_size)
            result["recommended_accumulation_steps"] = (
                int(protocol["gradient_accumulation_steps"])
                if batch_size == requested
                else int(protocol["fallback_gradient_accumulation_steps"])
            )
            output_dir = Path(protocol["output_dir"]).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                output_dir / "preflight.json",
                json.dumps(result, indent=2, sort_keys=True),
            )
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
            return
        except torch.OutOfMemoryError as error:
            errors.append(f"batch_size={batch_size}: {error}")
            torch.cuda.empty_cache()
    raise RuntimeError("all preflight batch sizes failed: " + " | ".join(errors))


def main() -> None:
    args = parse_args()
    protocol = apply_path_overrides(json.loads(args.config.resolve().read_text(encoding="utf-8")))
    dataset_config = DatasetConfig.load(protocol["dataset_manifest"])
    validate_dataset_statistics(protocol, dataset_config)
    if args.preflight:
        run_preflight(protocol, dataset_config)
        return

    seed_everything(int(protocol["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(protocol["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "metrics.json"
    resume_path = args.resume.resolve() if args.resume else None
    if resume_path is None and result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed run: {result_path}")

    resume_preview: dict[str, Any] | None = None
    if resume_path is not None:
        resume_preview = torch.load(resume_path, map_location="cpu", weights_only=True)
    effective_batch_size = int(protocol["batch_size"])
    effective_accumulation_steps = int(protocol["gradient_accumulation_steps"])
    if resume_preview is not None:
        saved_runtime = resume_preview.get("extra_state", {})
        effective_batch_size = int(saved_runtime.get("effective_batch_size", effective_batch_size))
        effective_accumulation_steps = int(
            saved_runtime.get("effective_accumulation_steps", effective_accumulation_steps)
        )
    else:
        preflight_path = output_dir / "preflight.json"
        if preflight_path.is_file():
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            effective_batch_size = int(preflight["batch_size"])
            effective_accumulation_steps = int(preflight["recommended_accumulation_steps"])

    patch_size = int(protocol["patch_size"])
    train_set = PairedRasterDataset(dataset_config, "train", training=True, patch_size=patch_size)
    val_set = PairedRasterDataset(dataset_config, "val", training=False, patch_size=patch_size)
    test_set = PairedRasterDataset(dataset_config, "test", training=False, patch_size=patch_size)
    generator = torch.Generator().manual_seed(int(protocol["seed"]))
    train_loader = make_loader(
        train_set,
        batch_size=effective_batch_size,
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
    criterion = RoGLSNetLoss(
        dataset_config.ignore_index,
        auxiliary_weight=float(protocol["auxiliary_loss_weight"]),
    )
    amp_enabled = bool(protocol["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    freeze_epochs = int(protocol["freeze_epochs"])
    start_epoch = 1
    best_mf1 = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    logs: list[dict[str, Any]] = []
    elapsed_before_resume = 0.0

    saved_epoch = 0
    if resume_preview is not None:
        saved_epoch = int(resume_preview["epoch"])
    current_phase = phase_for_epoch(max(1, saved_epoch), freeze_epochs)
    configure_phase(model, current_phase)
    optimizer, scheduler = build_optimizer(model, protocol, current_phase)
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
        if int(restored.get("extra_state", {}).get("phase", current_phase)) != current_phase:
            raise ValueError("resume checkpoint records a different training phase")
        start_epoch = saved_epoch + 1
        best_mf1 = float(restored["best_mf1"])
        best_epoch = int(restored["best_epoch"])
        best_metrics = {key: float(value) for key, value in restored["best_metrics"].items()}
        logs = list(restored["logs"])
        elapsed_before_resume = float(restored["elapsed_seconds"])
        print(f"resumed after epoch {saved_epoch} (phase {current_phase})", flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    latest_checkpoint = checkpoint_dir / "resume_checkpoint.pt"
    best_model_path = checkpoint_dir / "best_model.pt"
    total_epochs = int(protocol["epochs"])
    checkpoint_interval = int(protocol["checkpoint_interval"])
    accumulation_steps = effective_accumulation_steps
    started = time.monotonic()
    write_live_artifacts(
        output_dir,
        logs,
        total_epochs=total_epochs,
        state="running",
        elapsed_seconds=elapsed_before_resume,
        title="DSMRelativeRoGLSNet",
    )

    for epoch in range(start_epoch, total_epochs + 1):
        phase = phase_for_epoch(epoch, freeze_epochs)
        if phase != current_phase:
            current_phase = phase
            configure_phase(model, current_phase)
            optimizer, scheduler = build_optimizer(model, protocol, current_phase)
            print(f"entered training phase {current_phase} at epoch {epoch}", flush=True)
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
            training_mode=lambda active: apply_training_mode(active, phase),
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
            "phase": phase,
            "height_strength": float(model.height_strength().detach()),
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
        }
        logs.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        write_live_artifacts(
            output_dir,
            logs,
            total_epochs=total_epochs,
            state="running",
            elapsed_seconds=elapsed,
            title="DSMRelativeRoGLSNet",
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
                extra_state={
                    "phase": phase,
                    "height_strength": float(model.height_strength().detach()),
                    "effective_batch_size": effective_batch_size,
                    "effective_accumulation_steps": accumulation_steps,
                },
            )

    runtime = elapsed_before_resume + time.monotonic() - started
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    normal, normal_per_class, normal_boundary = evaluate_with_boundary(
        model, test_loader, dataset_config, device, amp_enabled
    )
    zero, zero_per_class, zero_boundary = evaluate_with_boundary(
        model, test_loader, dataset_config, device, amp_enabled, zero_auxiliary=True
    )
    shuffled_set = ShuffledAuxiliaryDataset(test_set)
    shuffled_loader = make_loader(
        shuffled_set,
        batch_size=1,
        num_workers=int(protocol["num_workers"]),
        shuffle=False,
        device=device,
    )
    shuffled, shuffled_per_class, shuffled_boundary = evaluate_with_boundary(
        model, shuffled_loader, dataset_config, device, amp_enabled
    )
    mf1_margin = normal["mf1"] - max(zero["mf1"], shuffled["mf1"])
    boundary_margin = float(normal_boundary["mean_bf1"]) - max(
        float(zero_boundary["mean_bf1"]), float(shuffled_boundary["mean_bf1"])
    )
    diagnostics = {
        "normal": {**normal, "per_class": normal_per_class, "boundary": normal_boundary},
        "zero": {**zero, "per_class": zero_per_class, "boundary": zero_boundary},
        "shuffled": {
            **shuffled,
            "per_class": shuffled_per_class,
            "boundary": shuffled_boundary,
        },
        "mf1_margin_over_strongest_control": mf1_margin,
        "boundary_mf1_margin_over_strongest_control": boundary_margin,
        "effective_at_0.002": mf1_margin >= 0.002,
        "boundary_effective_at_0.002": boundary_margin >= 0.002,
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
        "model": "DSMRelativeRoGLSNet",
        "selection_metric": "official_17_area_test_mf1_without_tta",
        "best_benchmark_test": {"epoch": best_epoch, **best_metrics},
        "area30_role": "logged_only_not_used_for_selection",
        "auxiliary_diagnostics": diagnostics,
        "final_test": final_test,
        "height_strength": float(model.height_strength().detach()),
        "effective_batch_size": effective_batch_size,
        "effective_accumulation_steps": accumulation_steps,
        "training_runtime_seconds": runtime,
        "peak_gpu_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "paper_targets": {"oa": 0.9328, "mf1": 0.9131, "miou": 0.8431},
        "input_fusion_reference_mf1": 0.904071033000946,
        "logs": logs,
    }
    temporary_result = result_path.with_suffix(".json.tmp")
    temporary_result.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary_result.replace(result_path)
    write_live_artifacts(
        output_dir,
        logs,
        total_epochs=total_epochs,
        state="completed",
        elapsed_seconds=runtime,
        title="DSMRelativeRoGLSNet",
    )
    print(
        json.dumps(
            {"best_benchmark_test": result["best_benchmark_test"], "final_test": final_test},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
