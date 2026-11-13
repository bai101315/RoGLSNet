# Vaihingen multimodal experiment record

This directory is the immutable, lightweight record of the experiments completed on 2026-11-13. It intentionally contains metrics and per-epoch histories but no ISPRS imagery and no model binaries.

## Contents

- `input_fusion/`: highest absolute score plus the control showing that it barely uses DSM.
- `four_stage/`: four-level DSM residual fusion result and DSM controls.
- `height_stage1/`, `height_stage2/`: complete metrics, per-epoch CSV, and DSM controls.
- `comparison/`: final side-by-side table.
- `manifests/`: dataset protocol snapshots, without raster data.

The committed `metrics.json` files are the source of truth; rounded headline values are summarized in `comparison/comparison.md`. Model binaries are distributed as GitHub Release assets because each file is larger than GitHub's normal Git-object limit.

Important protocol note: the requested exploratory runs evaluated the 17 official areas every epoch and selected by their no-TTA mF1. Consequently these numbers are useful for engineering comparison but are not an untouched, unbiased final-test estimate. `area30` was logged only.
