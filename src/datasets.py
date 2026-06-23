"""Dataset helpers for industrial defect image classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_imagefolder_layout(data_root: Path, class_names: list[str]) -> None:
    """Validate the expected ImageFolder split and class directory layout."""
    expected_splits = ("train", "val", "test")

    missing_paths: list[Path] = []
    for split in expected_splits:
        split_dir = data_root / split
        if not split_dir.exists():
            missing_paths.append(split_dir)
            continue

        for class_name in class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                missing_paths.append(class_dir)

    if missing_paths:
        formatted = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "The dataset layout is incomplete. Missing paths:\n" + formatted
        )


def create_dataloaders(config: dict[str, Any]) -> dict[str, Any]:
    """Create train, validation, and test DataLoaders.

    TODO:
    - Build transforms from `src.transforms`.
    - Load ImageFolder datasets.
    - Create DataLoaders with batch size and num workers from config.
    """
    raise NotImplementedError("DataLoader creation is not implemented yet.")

