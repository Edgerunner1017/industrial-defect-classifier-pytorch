"""Shared utilities for configuration, logging, and reproducibility."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy
import torch


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
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

def get_device(config:dict) -> torch.device:
    """Get a PyTorch device.
    Args:
        config: 配置字典，包含 training.device 字段
            支持的值: "auto", "cpu", "cuda", "cuda:0", "cuda:1" 等

    Returns:
        torch.device: PyTorch 设备对象"""
    device_str = config.get("device", {}).get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(device_str)

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(data, path):

def
