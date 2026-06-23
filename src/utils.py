"""Shared utilities for configuration, logging, and reproducibility."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = Path(config_path)

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


def print_config_summary(config: Mapping[str, Any]) -> None:
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


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)

    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "NumPy is required for reproducibility setup. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for reproducibility setup. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: Mapping[str, Any]):
    """Return the torch device requested by the config."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for device selection. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    training = config.get("training", {})
    requested = str(training.get("device", "auto")).lower()

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA, but CUDA is not available.")
        return torch.device("cuda")

    raise ValueError(
        f"Unsupported device value: {requested!r}. Use 'auto', 'cpu', or 'cuda'."
    )


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: Mapping[str, Any], path: str | Path) -> Path:
    """Save a mapping as pretty-printed JSON."""
    output_path = Path(path)
    ensure_dir(output_path.parent)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")

    return output_path


def save_checkpoint(state: Mapping[str, Any], path: str | Path) -> Path:
    """Save a PyTorch checkpoint dictionary."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to save checkpoints. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    checkpoint_path = Path(path)
    ensure_dir(checkpoint_path.parent)
    torch.save(dict(state), checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, device: str | Any = "cpu") -> dict[str, Any]:
    """Load a PyTorch checkpoint dictionary."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load checkpoints. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {checkpoint_path}")

    return checkpoint
