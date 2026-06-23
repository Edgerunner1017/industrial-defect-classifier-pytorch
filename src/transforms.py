"""Image transform builders for training and evaluation."""

from __future__ import annotations

from typing import Any


def build_train_transforms(config: dict[str, Any]) -> Any:
    """Build training image augmentations.

    TODO:
    - Resize or random resized crop to the configured image size.
    - Add horizontal flip if suitable for the selected defect dataset.
    - Normalize with ImageNet mean and std for pretrained models.
    """
    raise NotImplementedError("Training transforms are not implemented yet.")


def build_eval_transforms(config: dict[str, Any]) -> Any:
    """Build deterministic validation/test image preprocessing."""
    raise NotImplementedError("Evaluation transforms are not implemented yet.")

