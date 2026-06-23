"""Training entry point for the industrial defect classifier.

This file currently defines the command-line interface and the planned data
flow. The full training loop will be implemented in the next development pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an industrial surface defect image classifier."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML experiment config, for example configs/resnet18.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the config and print the planned training flow without training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from utils import load_config, print_config_summary

    config = load_config(args.config)
    print_config_summary(config)

    print("\nPlanned training flow:")
    print("1. Build train and validation transforms.")
    print("2. Load ImageFolder datasets and DataLoaders.")
    print("3. Build the configured torchvision model.")
    print("4. Train for the configured number of epochs.")
    print("5. Save the best checkpoint by validation metric.")

    if args.dry_run:
        print("\nDry run complete. No training was started.")
        return

    raise NotImplementedError(
        "The training loop is planned but not implemented yet. "
        "Use --dry-run to validate the current skeleton."
    )


if __name__ == "__main__":
    main()

