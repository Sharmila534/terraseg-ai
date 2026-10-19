from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _index(directory: str | Path) -> tuple[dict[str, Path], list[str]]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    found: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS):
        key = path.stem.casefold()
        if key in found:
            duplicates.append(f"{found[key].name} and {path.name}")
        else:
            found[key] = path
    return found, duplicates


def audit_pairs(image_dir: str | Path, mask_dir: str | Path) -> dict:
    images, image_duplicates = _index(image_dir)
    masks, mask_duplicates = _index(mask_dir)
    common = sorted(images.keys() & masks.keys())
    return {
        "image_count": len(images), "mask_count": len(masks), "paired_count": len(common),
        "unmatched_images": [images[k].name for k in sorted(images.keys() - masks.keys())],
        "unmatched_masks": [masks[k].name for k in sorted(masks.keys() - images.keys())],
        "duplicate_images": image_duplicates, "duplicate_masks": mask_duplicates,
    }


def matched_pairs(image_dir: str | Path, mask_dir: str | Path, strict: bool = True) -> list[tuple[str, Path, Path]]:
    report = audit_pairs(image_dir, mask_dir)
    if report["duplicate_images"] or report["duplicate_masks"]:
        raise ValueError(f"Duplicate filename stems: {report}")
    if strict and (report["unmatched_images"] or report["unmatched_masks"]):
        raise ValueError(f"Unmatched image/mask files: {report}")
    images, _ = _index(image_dir)
    masks, _ = _index(mask_dir)
    return [(key, images[key], masks[key]) for key in sorted(images.keys() & masks.keys())]


def make_splits(ids: Sequence[str], seed: int, train_fraction: float = .70, validation_fraction: float = .15) -> dict[str, list[str]]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and leave a positive test fraction")
    ordered = list(ids)
    if len(set(ordered)) != len(ordered):
        raise ValueError("Sample IDs must be unique before splitting")
    n = len(ordered)
    if n < 3:
        raise ValueError(f"At least 3 matched image-mask pairs are required for train/validation/test; found {n}")
    random.Random(seed).shuffle(ordered)
    n_train = max(1, int(n * train_fraction))
    n_val = max(1, int(n * validation_fraction))
    # Preserve at least one example in each partition for small datasets.
    while n_train + n_val >= n:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Could not form non-empty train/validation/test partitions")
    return {"train": ordered[:n_train], "validation": ordered[n_train:n_train+n_val], "test": ordered[n_train+n_val:]}


def save_splits(splits: dict, path: str | Path, seed: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "splits": splits}, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["splits"]


class BijieDataset(Dataset):
    def __init__(self, image_dir, mask_dir, ids=None, image_size=256, augment=False, imagenet_normalization=False):
        self.pairs = matched_pairs(image_dir, mask_dir, strict=True)
        if ids is not None:
            wanted = set(ids)
            self.pairs = [p for p in self.pairs if p[0] in wanted]
            if {p[0] for p in self.pairs} != wanted:
                raise ValueError("Split contains IDs missing from the paired dataset")
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.imagenet_normalization = bool(imagenet_normalization)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        _, image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        if self.augment:
            if random.random() < .5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < .5:
                image, mask = TF.vflip(image), TF.vflip(mask)
        image_t = TF.to_tensor(image)
        mask_t = (TF.to_tensor(mask) > .5).float()
        if not torch.isfinite(image_t).all() or image_t.min() < 0 or image_t.max() > 1:
            raise ValueError(f"Invalid image tensor range: {image_path}")
        if not torch.logical_or(mask_t == 0, mask_t == 1).all():
            raise ValueError(f"Mask is not binary: {mask_path}")
        if self.imagenet_normalization:
            mean = torch.tensor([.485, .456, .406])[:, None, None]
            std = torch.tensor([.229, .224, .225])[:, None, None]
            image_t = (image_t - mean) / std
        return image_t, mask_t
