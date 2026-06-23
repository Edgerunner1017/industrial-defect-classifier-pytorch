"""Single-image prediction entry point for the industrial defect classifier."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict the defect class for a single image."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML config used for the experiment.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained checkpoint file.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image that should be classified.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from utils import load_config, print_config_summary

    config = load_config(args.config)
    print_config_summary(config)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Image: {args.image}")

    raise NotImplementedError(
        "Single-image preprocessing and inference are planned but not "
        "implemented yet."
    )


if __name__ == "__main__":
    main()

