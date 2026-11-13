"""Build tile-normalized DSM patches without touching the existing dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import tifffile


PATCH_PATTERN = re.compile(r"^area(?P<area>\d+)_y(?P<y>\d+)_x(?P<x>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-dsm-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"unexpected symlink target: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"refusing to replace: {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)


def _normalized_tile(path: Path) -> tuple[np.ndarray, float, float]:
    tile = tifffile.imread(path).astype(np.float32)
    if tile.ndim != 2:
        raise ValueError(f"DSM tile must be two-dimensional: {path} {tile.shape}")
    finite = np.isfinite(tile)
    if not finite.any():
        raise ValueError(f"DSM tile contains no finite pixels: {path}")
    low, high = np.percentile(tile[finite], (5.0, 95.0))
    if high <= low:
        raise ValueError(f"DSM percentile range is empty: {path}")
    normalized = np.zeros_like(tile, dtype=np.float32)
    normalized[finite] = np.clip((tile[finite] - low) / (high - low), 0.0, 1.0)
    return normalized, float(low), float(high)


def _extract_patch(tile: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    patch = np.zeros((size, size), dtype=np.float16)
    source = tile[y : y + size, x : x + size]
    patch[: source.shape[0], : source.shape[1]] = source.astype(np.float16)
    return patch


def main() -> None:
    args = parse_args()
    source_manifest = args.source_manifest.resolve()
    raw_dsm_dir = args.raw_dsm_dir.resolve()
    output_dir = args.output_dir.resolve()
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_root = source_manifest.parent
    patch_size = int(payload["patch_size"])
    output_dir.mkdir(parents=True, exist_ok=True)

    percentiles: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "test"):
        source_split = source_root / split
        output_split = output_dir / split
        _relative_symlink(source_split / payload["primary_dir"], output_split / "primary")
        _relative_symlink(source_split / payload["mask_dir"], output_split / "masks")
        auxiliary_dir = output_split / "auxiliary"
        auxiliary_dir.mkdir(parents=True, exist_ok=True)

        primary_files = sorted((source_split / payload["primary_dir"]).iterdir())
        areas: dict[int, list[tuple[Path, int, int]]] = {}
        for primary_path in primary_files:
            match = PATCH_PATTERN.match(primary_path.stem)
            if match is None:
                continue
            area = int(match.group("area"))
            areas.setdefault(area, []).append(
                (primary_path, int(match.group("y")), int(match.group("x")))
            )

        for area, patches in sorted(areas.items()):
            raw_path = raw_dsm_dir / f"dsm_09cm_matching_area{area}.tif"
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            normalized, low, high = _normalized_tile(raw_path)
            percentiles[str(area)] = {"p05": low, "p95": high}
            for primary_path, y, x in patches:
                destination = auxiliary_dir / f"{primary_path.stem}.npy"
                expected = _extract_patch(normalized, y, x, patch_size)
                if destination.exists():
                    current = np.load(destination, allow_pickle=False)
                    if current.shape != expected.shape or not np.array_equal(current, expected):
                        raise FileExistsError(f"existing normalized patch differs: {destination}")
                    continue
                np.save(destination, expected, allow_pickle=False)

    normalized_manifest = dict(payload)
    normalized_manifest.update(
        {
            "auxiliary_channels": 1,
            "auxiliary_dir": "auxiliary",
            "auxiliary_mean": [0.0],
            "auxiliary_std": [1.0],
            "dsm_normalization": {
                "scope": "original_area_before_patching",
                "clip_percentiles": [5.0, 95.0],
                "range": [0.0, 1.0],
                "per_area_percentiles": percentiles,
            },
            "source_manifest": str(source_manifest),
        }
    )
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(normalized_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
