"""Dataset helpers for industrial defect image classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def validate_imagefolder_layout(data_root: str | Path, class_names: list[str]) -> None:
    """Validate the expected ImageFolder split and class directory layout."""
    data_root = Path(data_root)
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


def _validate_dataset_classes(
    split_name: str, actual_classes: list[str], expected_classes: list[str]
) -> None:
    actual = set(actual_classes)
    expected = set(expected_classes)

    if actual == expected:
        return

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    details: list[str] = []
    if missing:
        details.append(f"missing classes: {missing}")
    if unexpected:
        details.append(f"unexpected classes: {unexpected}")

    raise ValueError(
        f"Class mismatch in {split_name} split. " + "; ".join(details)
    )


def create_dataloaders(config: Mapping[str, Any]) -> dict[str, Any]:
    """Create train, validation, and test DataLoaders."""
    try:
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder
    except ImportError as exc:
        raise ImportError(
            "PyTorch and torchvision are required to create DataLoaders. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    from transforms import build_eval_transforms, build_train_transforms

    data_config = config.get("data", {})
    training_config = config.get("training", {})

    data_root = Path(data_config.get("root", "data"))
    class_names = list(data_config.get("classes", []))
    if not class_names:
        raise ValueError("Config must define data.classes.")

    validate_imagefolder_layout(data_root, class_names)

    train_dataset = ImageFolder(
        root=str(data_config.get("train_dir", data_root / "train")),
        transform=build_train_transforms(config),
    )
    val_dataset = ImageFolder(
        root=str(data_config.get("val_dir", data_root / "val")),
        transform=build_eval_transforms(config),
    )
    test_dataset = ImageFolder(
        root=str(data_config.get("test_dir", data_root / "test")),
        transform=build_eval_transforms(config),
    )

    for split_name, dataset in (
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        _validate_dataset_classes(split_name, dataset.classes, class_names)

    batch_size = int(training_config.get("batch_size", 32))
    num_workers = int(training_config.get("num_workers", 0))

    return {
        "train_loader": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val_loader": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test_loader": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "class_names": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
    }
