"""Compare normal, zeroed, and spatially unrelated DSM on the official test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import DatasetConfig, PairedRasterDataset
from .boundary_metrics import evaluate_with_boundary
from .dsm_rope_model import DSMRelativeRoGLSNet
from .height_train import (
    ShuffledAuxiliaryDataset,
    apply_path_overrides,
)
from .input_fusion_model import InputFusionRoGLSNet
from .model import AsymmetricFusionRoGLSNet
from .train import make_loader, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("four_stage", "input_fusion", "dsm_relative_rope"),
        default="four_stage",
        help="architecture stored in the checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = apply_path_overrides(
        json.loads(args.config.resolve().read_text(encoding="utf-8"))
    )
    dataset_config = DatasetConfig.load(protocol["dataset_manifest"])
    seed_everything(int(protocol["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(protocol["mixed_precision"]) and device.type == "cuda"
    dataset = PairedRasterDataset(
        dataset_config,
        "test",
        training=False,
        patch_size=int(protocol["patch_size"]),
    )
    loader = make_loader(
        dataset,
        batch_size=1,
        num_workers=int(protocol["num_workers"]),
        shuffle=False,
        device=device,
    )
    model_classes = {
        "four_stage": AsymmetricFusionRoGLSNet,
        "input_fusion": InputFusionRoGLSNet,
        "dsm_relative_rope": DSMRelativeRoGLSNet,
    }
    model_class = model_classes[args.model]
    model = model_class(
        num_classes=dataset_config.num_classes,
        auxiliary_channels=dataset_config.auxiliary_channels,
        pretrained=False,
    ).to(device)
    state = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    normal, normal_per_class, normal_boundary = evaluate_with_boundary(
        model, loader, dataset_config, device, amp_enabled
    )
    zero, zero_per_class, zero_boundary = evaluate_with_boundary(
        model, loader, dataset_config, device, amp_enabled, zero_auxiliary=True
    )
    shuffled = ShuffledAuxiliaryDataset(dataset)
    shuffled_loader = make_loader(
        shuffled,
        batch_size=1,
        num_workers=int(protocol["num_workers"]),
        shuffle=False,
        device=device,
    )
    shuffled_metrics, shuffled_per_class, shuffled_boundary = evaluate_with_boundary(
        model,
        shuffled_loader,
        dataset_config,
        device,
        amp_enabled,
    )
    margin = normal["mf1"] - max(zero["mf1"], shuffled_metrics["mf1"])
    boundary_margin = float(normal_boundary["mean_bf1"]) - max(
        float(zero_boundary["mean_bf1"]), float(shuffled_boundary["mean_bf1"])
    )
    result = {
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "tta": False,
        "normal": {**normal, "per_class": normal_per_class, "boundary": normal_boundary},
        "zero": {**zero, "per_class": zero_per_class, "boundary": zero_boundary},
        "shuffled": {
            **shuffled_metrics,
            "per_class": shuffled_per_class,
            "boundary": shuffled_boundary,
        },
        "mf1_margin_over_strongest_control": margin,
        "boundary_mf1_margin_over_strongest_control": boundary_margin,
        "effective_at_0.002": margin >= 0.002,
        "boundary_effective_at_0.002": boundary_margin >= 0.002,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
