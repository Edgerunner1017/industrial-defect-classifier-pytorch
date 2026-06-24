"""Split the flat NEU-CLS dataset into ImageFolder train/val/test folders."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


PREFIX_TO_CLASS = {
    "Cr": "crack",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled_scale",
    "Sc": "scratch",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare NEU-CLS as an ImageFolder dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to the flat NEU-CLS folder containing .bmp files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Output dataset root. Defaults to data/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio per class.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio per class.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied image files.",
    )
    return parser


def class_name_from_file(path: Path) -> str:
    prefix = path.stem.split("_", maxsplit=1)[0]
    if prefix not in PREFIX_TO_CLASS:
        raise ValueError(f"Unknown NEU-CLS filename prefix {prefix!r}: {path.name}")
    return PREFIX_TO_CLASS[prefix]


def collect_images(source: Path) -> dict[str, list[Path]]:
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source}")

    grouped: dict[str, list[Path]] = defaultdict(list)
    for image_path in sorted(source.glob("*.bmp")):
        grouped[class_name_from_file(image_path)].append(image_path)

    expected_classes = set(PREFIX_TO_CLASS.values())
    actual_classes = set(grouped)
    if actual_classes != expected_classes:
        missing = sorted(expected_classes - actual_classes)
        unexpected = sorted(actual_classes - expected_classes)
        raise ValueError(
            f"Unexpected class set. Missing={missing}, unexpected={unexpected}"
        )

    return dict(grouped)


def split_class_images(
    images: list[Path],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy: train > 0, val > 0, train + val < 1.")

    shuffled = list(images)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def copy_split(
    class_name: str,
    split_name: str,
    images: list[Path],
    output_root: Path,
    overwrite: bool,
) -> int:
    target_dir = output_root / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for source_path in images:
        target_path = target_dir / source_path.name
        if target_path.exists() and not overwrite:
            raise FileExistsError(
                f"Target file already exists: {target_path}. "
                "Use --overwrite to regenerate the split."
            )

        shutil.copy2(source_path, target_path)
        copied += 1

    return copied


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)

    grouped = collect_images(args.source)
    summary: dict[str, dict[str, int]] = {}

    for class_name in sorted(grouped):
        split = split_class_images(
            grouped[class_name],
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            rng=rng,
        )
        summary[class_name] = {}

        for split_name, images in split.items():
            summary[class_name][split_name] = copy_split(
                class_name,
                split_name,
                images,
                args.output,
                overwrite=args.overwrite,
            )

    print("NEU-CLS split complete")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Seed: {args.seed}")
    print()
    print("Class counts:")

    for class_name in sorted(summary):
        counts = summary[class_name]
        total = sum(counts.values())
        print(
            f"- {class_name}: "
            f"train={counts['train']}, val={counts['val']}, "
            f"test={counts['test']}, total={total}"
        )


if __name__ == "__main__":
    main()
