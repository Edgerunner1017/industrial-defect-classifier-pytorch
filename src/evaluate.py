"""Evaluation entry point for a trained industrial defect classifier."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained industrial surface defect classifier."
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
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from utils import load_config, print_config_summary

    config = load_config(args.config)
    print_config_summary(config)
    print(f"\nCheckpoint to evaluate: {args.checkpoint}")

    raise NotImplementedError(
        "Evaluation metrics and confusion matrix generation are planned but "
        "not implemented yet."
    )


if __name__ == "__main__":
    main()

