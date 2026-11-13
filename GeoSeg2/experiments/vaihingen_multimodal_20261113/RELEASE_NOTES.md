# Vaihingen multimodal handoff

This release contains the two checkpoints needed to continue the completed experiments:

- `input_fusion_best_model.pt`: highest absolute score (`OA 91.09`, `mF1 90.41`, `mIoU 82.70` with D4), but the DSM control failed. Use its `segmenter` as the strong optical initialization.
- `height_selective_stage2_best_model.pt`: selective-height stage 2 (`OA 90.71`, `mF1 89.85`, `mIoU 81.79` with D4); its normal DSM beat zero/shuffled controls by `0.531` percentage points.

Verify both files with the committed `SHA256SUMS`. Dataset imagery is intentionally excluded because Vaihingen is distributed under the official ISPRS data terms. Code, exact protocol snapshots, per-epoch histories, metrics, and the RTX 3090 continuation plan are in the tagged commit.
