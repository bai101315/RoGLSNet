"""Compare normal, zeroed, and spatially unrelated DSM on the official test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import DatasetConfig, PairedRasterDataset
from .height_train import (
    ShuffledAuxiliaryDataset,
    apply_path_overrides,
    evaluate_zero_auxiliary,
)
from .input_fusion_model import InputFusionRoGLSNet
from .model import AsymmetricFusionRoGLSNet
from .train import evaluate_loader, make_loader, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=("four_stage", "input_fusion"),
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
    model_class = (
        InputFusionRoGLSNet if args.model == "input_fusion" else AsymmetricFusionRoGLSNet
    )
    model = model_class(
        num_classes=dataset_config.num_classes,
        auxiliary_channels=dataset_config.auxiliary_channels,
        pretrained=False,
    ).to(device)
    state = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)

    normal, normal_per_class = evaluate_loader(
        model, loader, dataset_config, device, amp_enabled, use_d4=False
    )
    zero, zero_per_class = evaluate_zero_auxiliary(
        model, loader, dataset_config, device, amp_enabled
    )
    shuffled = ShuffledAuxiliaryDataset(dataset)
    shuffled_loader = make_loader(
        shuffled,
        batch_size=1,
        num_workers=int(protocol["num_workers"]),
        shuffle=False,
        device=device,
    )
    shuffled_metrics, shuffled_per_class = evaluate_loader(
        model,
        shuffled_loader,
        dataset_config,
        device,
        amp_enabled,
        use_d4=False,
    )
    margin = normal["mf1"] - max(zero["mf1"], shuffled_metrics["mf1"])
    result = {
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "tta": False,
        "normal": {**normal, "per_class": normal_per_class},
        "zero": {**zero, "per_class": zero_per_class},
        "shuffled": {**shuffled_metrics, "per_class": shuffled_per_class},
        "mf1_margin_over_strongest_control": margin,
        "effective_at_0.002": margin >= 0.002,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
