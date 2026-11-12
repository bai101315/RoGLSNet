"""Aligned offline-patch dataset for IRRG and DSM."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    primary_dir: str
    auxiliary_dir: str
    mask_dir: str
    num_classes: int
    ignore_index: int
    auxiliary_channels: int
    auxiliary_mean: tuple[float, ...]
    auxiliary_std: tuple[float, ...]
    class_names: tuple[str, ...]

    @classmethod
    def load(cls, manifest_path: str | Path) -> "DatasetConfig":
        path = Path(manifest_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        mean = tuple(float(value) for value in payload["auxiliary_mean"])
        std = tuple(float(value) for value in payload["auxiliary_std"])
        channels = int(payload["auxiliary_channels"])
        if len(mean) != channels or len(std) != channels or any(value <= 0 for value in std):
            raise ValueError("auxiliary_mean/std must have one valid value per channel")
        return cls(
            root=path.parent,
            primary_dir=str(payload.get("primary_dir", "primary")),
            auxiliary_dir=str(payload.get("auxiliary_dir", "auxiliary")),
            mask_dir=str(payload.get("mask_dir", "masks")),
            num_classes=int(payload["num_classes"]),
            ignore_index=int(payload["ignore_index"]),
            auxiliary_channels=channels,
            auxiliary_mean=mean,
            auxiliary_std=std,
            class_names=tuple(str(name) for name in payload["class_names"]),
        )


def _read_raster(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile

            array = tifffile.imread(path)
        except ImportError:
            array = np.asarray(Image.open(path))
    else:
        array = np.asarray(Image.open(path))
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"expected a HWC or CHW raster at {path}, got {array.shape}")
    if array.shape[0] <= 32 and array.shape[-1] > 32:
        array = np.moveaxis(array, 0, -1)
    return np.asarray(array)


def _files_by_stem(directory: Path) -> dict[str, Path]:
    allowed = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
    result = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in allowed
    }
    if not result:
        raise ValueError(f"no supported rasters found in {directory}")
    return result


class PairedRasterDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        *,
        training: bool,
        patch_size: int,
    ) -> None:
        self.config = config
        self.training = training
        self.patch_size = patch_size
        split_root = config.root / split
        primary = _files_by_stem(split_root / config.primary_dir)
        auxiliary = _files_by_stem(split_root / config.auxiliary_dir)
        masks = _files_by_stem(split_root / config.mask_dir)
        shared = sorted(primary.keys() & auxiliary.keys() & masks.keys())
        if set(shared) != set(primary) or set(shared) != set(auxiliary) or set(shared) != set(masks):
            raise ValueError(f"{split}: primary, auxiliary and masks need identical stems")
        self.samples = [
            (stem, primary[stem], auxiliary[stem], masks[stem])
            for stem in shared
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        stem, primary_path, auxiliary_path, mask_path = self.samples[index]
        primary = _read_raster(primary_path)
        auxiliary = _read_raster(auxiliary_path)
        mask = _read_raster(mask_path)[..., 0]
        primary_dtype = primary.dtype
        if primary.shape[-1] != 3:
            raise ValueError(f"{primary_path} must contain three IRRG bands")
        if auxiliary.shape[-1] != self.config.auxiliary_channels:
            raise ValueError(f"{auxiliary_path} has an unexpected channel count")
        if primary.shape[:2] != auxiliary.shape[:2] or primary.shape[:2] != mask.shape:
            raise ValueError(f"unaligned paired sample: {stem}")
        if mask.shape != (self.patch_size, self.patch_size):
            raise ValueError(
                f"{stem}: expected an offline {self.patch_size}x{self.patch_size} patch, "
                f"got {mask.shape}"
            )
        if not np.any(mask != self.config.ignore_index):
            raise ValueError(f"{stem}: source patch contains no supervised pixels")

        original = (primary, auxiliary, mask)
        if self.training:
            primary, auxiliary, mask = self._augment(primary, auxiliary, mask)
            # A scaled crop at a padded tile edge can occasionally contain only
            # ignore pixels. Fall back to the aligned source patch in that case.
            if not np.any(mask != self.config.ignore_index):
                primary, auxiliary, mask = original

        primary = primary.astype(np.float32)
        if np.issubdtype(primary_dtype, np.integer):
            primary /= float(np.iinfo(primary_dtype).max)
        image_mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
        image_std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
        primary = (primary - image_mean) / image_std
        auxiliary = auxiliary.astype(np.float32)
        auxiliary_mean = np.asarray(self.config.auxiliary_mean, dtype=np.float32)
        auxiliary_std = np.asarray(self.config.auxiliary_std, dtype=np.float32)
        auxiliary = (auxiliary - auxiliary_mean) / auxiliary_std
        return {
            "sample_id": stem,
            "primary": torch.from_numpy(primary.transpose(2, 0, 1)).float(),
            "auxiliary": torch.from_numpy(auxiliary.transpose(2, 0, 1)).float(),
            "mask": torch.from_numpy(mask.astype(np.int64)),
        }

    def _augment(
        self,
        primary: np.ndarray,
        auxiliary: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = random.choice((0.5, 0.75, 1.0, 1.25, 1.5))
        if scale != 1.0:
            size = max(1, round(self.patch_size * scale))
            primary = self._resize(primary, (size, size), mode="bilinear")
            auxiliary = self._resize(auxiliary, (size, size), mode="bilinear")
            mask = self._resize(mask, (size, size), mode="nearest")
        primary, auxiliary, mask = self._crop_or_pad(primary, auxiliary, mask)
        if random.random() < 0.5:
            primary, auxiliary, mask = primary[:, ::-1], auxiliary[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            primary, auxiliary, mask = primary[::-1], auxiliary[::-1], mask[::-1]
        rotations = random.randint(0, 3)
        if rotations:
            primary = np.rot90(primary, rotations)
            auxiliary = np.rot90(auxiliary, rotations)
            mask = np.rot90(mask, rotations)
        return tuple(
            np.ascontiguousarray(array)
            for array in (primary, auxiliary, mask)
        )

    @staticmethod
    def _resize(array: np.ndarray, size: tuple[int, int], *, mode: str) -> np.ndarray:
        had_channels = array.ndim == 3
        if not had_channels:
            array = array[..., None]
        tensor = torch.from_numpy(
            np.ascontiguousarray(array.transpose(2, 0, 1)).copy()
        ).float()
        options: dict[str, bool | str] = {"mode": mode}
        if mode != "nearest":
            options.update(align_corners=False, antialias=True)
        resized = F.interpolate(tensor.unsqueeze(0), size=size, **options)[0]
        result = resized.permute(1, 2, 0).numpy()
        if mode == "nearest":
            result = result.astype(array.dtype)
        return result if had_channels else result[..., 0]

    def _crop_or_pad(
        self,
        primary: np.ndarray,
        auxiliary: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = mask.shape
        top = random.randint(0, height - self.patch_size) if height > self.patch_size else 0
        left = random.randint(0, width - self.patch_size) if width > self.patch_size else 0
        bottom = min(top + self.patch_size, height)
        right = min(left + self.patch_size, width)
        primary = primary[top:bottom, left:right]
        auxiliary = auxiliary[top:bottom, left:right]
        mask = mask[top:bottom, left:right]
        if mask.shape == (self.patch_size, self.patch_size):
            return primary, auxiliary, mask

        pad_height = self.patch_size - mask.shape[0]
        pad_width = self.patch_size - mask.shape[1]
        pad_top = random.randint(0, pad_height)
        pad_left = random.randint(0, pad_width)
        spatial_padding = (
            (pad_top, pad_height - pad_top),
            (pad_left, pad_width - pad_left),
        )
        primary = np.pad(primary, spatial_padding + ((0, 0),), constant_values=0)
        auxiliary = np.pad(auxiliary, spatial_padding + ((0, 0),), constant_values=0)
        mask = np.pad(mask, spatial_padding, constant_values=self.config.ignore_index)
        return primary, auxiliary, mask
