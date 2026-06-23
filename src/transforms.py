"""Image transform builders for training and evaluation."""

from __future__ import annotations

from typing import Any, Mapping


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_image_size(config: Mapping[str, Any]) -> int:
    return int(config.get("data", {}).get("image_size", 224))


def build_train_transforms(config: Mapping[str, Any]):
    """Build training image augmentations."""
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to build image transforms. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    image_size = _get_image_size(config)

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transforms(config: Mapping[str, Any]):
    """Build deterministic validation/test image preprocessing."""
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError(
            "torchvision is required to build image transforms. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    image_size = _get_image_size(config)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
