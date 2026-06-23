"""Shared utilities for configuration, logging, and reproducibility."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load config files. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must define a mapping: {config_path}")

    return config


def print_config_summary(config: dict[str, Any]) -> None:
    """Print a compact summary of the current experiment config."""
    project = config.get("project", {})
    data = config.get("data", {})
    model = config.get("model", {})
    training = config.get("training", {})

    print("Experiment summary")
    print("------------------")
    print(f"Project: {project.get('name', 'unknown')}")
    print(f"Experiment: {project.get('experiment_name', 'unknown')}")
    print(f"Model: {model.get('name', 'unknown')}")
    print(f"Classes: {len(data.get('classes', []))}")
    print(f"Image size: {data.get('image_size', 'unknown')}")
    print(f"Epochs: {training.get('epochs', 'unknown')}")
    print(f"Batch size: {training.get('batch_size', 'unknown')}")


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments.

    TODO:
    - Seed Python `random`, NumPy, and PyTorch.
    - Configure deterministic PyTorch behavior where appropriate.
    """
    raise NotImplementedError("Seed setup is not implemented yet.")

