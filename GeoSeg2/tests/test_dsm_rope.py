from __future__ import annotations

import torch

from geoseg.models.RoGLSNet import Fusion
from multimodal.boundary_metrics import BoundaryF1
from multimodal.dsm_rope_model import DSMRelativeRoGLSNet
from multimodal.dsm_rope_train import configure_phase


def test_relative_height_bias_only_penalizes_cross_height_pairs() -> None:
    fusion = Fusion(dim=16, num_heads=2, window_size=4, rope_mixed=False)
    height = torch.zeros(1, 1, 4, 4)
    height[:, :, :, 2:] = 2.0

    bias = fusion.relative_height_bias(height, torch.tensor(1.0), (4, 4))

    assert bias.shape == (1, 1, 16, 16)
    assert bias[0, 0, 0, 1].item() == 0.0
    assert bias[0, 0, 0, 2].item() == -2.0
    assert bias[0, 0, 2, 3].item() == 0.0


def test_constant_height_matches_legacy_fusion_and_strength_gets_gradient() -> None:
    torch.manual_seed(7)
    fusion = Fusion(dim=16, num_heads=2, window_size=4, rope_mixed=False).eval()
    local = torch.randn(2, 16, 8, 8)
    global_scene = torch.randn(2, 16, 2, 2)
    constant_height = torch.full((2, 1, 64, 64), 3.5)
    strength = torch.tensor(0.7, requires_grad=True)

    legacy = fusion(local, global_scene)
    constant = fusion(local, global_scene, constant_height, strength)

    torch.testing.assert_close(constant, legacy, rtol=0.0, atol=1e-6)
    stepped_height = constant_height.clone()
    # Put the edge inside an RoAFM window; attention does not cross window borders.
    stepped_height[:, :, :, 24:] += 2.0
    output = fusion(local, global_scene, stepped_height, strength)
    output.square().mean().backward()
    assert strength.grad is not None
    assert torch.isfinite(strength.grad)
    assert strength.grad.abs().item() > 0.0


def test_training_phases_only_unfreeze_the_requested_parameters() -> None:
    model = DSMRelativeRoGLSNet(num_classes=6, pretrained=False)

    configure_phase(model, 1)
    phase_one = {name for name, value in model.named_parameters() if value.requires_grad}
    assert phase_one == {"height_strength_logit"}

    configure_phase(model, 2)
    phase_two = {name for name, value in model.named_parameters() if value.requires_grad}
    assert "height_strength_logit" in phase_two
    assert any(name.startswith("segmenter.fusion.") for name in phase_two)
    assert any(name.startswith("segmenter.segmentation_head.") for name in phase_two)
    assert not any(name.startswith("segmenter.backbone.") for name in phase_two)
    assert not any(name.startswith("segmenter.gca.") for name in phase_two)
    assert not any(name.startswith("segmenter.CLC3x3_1.") for name in phase_two)


def test_boundary_f1_is_one_for_a_perfect_prediction() -> None:
    target = torch.zeros(1, 16, 16, dtype=torch.long)
    target[:, :, 8:] = 1
    logits = torch.full((1, 6, 16, 16), -10.0)
    logits.scatter_(1, target[:, None], 10.0)
    metric = BoundaryF1(num_classes=6, ignore_index=6, tolerance=3)

    metric.update(logits, target)
    result = metric.compute(("surface", "building", "low", "tree", "car", "clutter"))

    assert result["mean_bf1"] == 1.0
