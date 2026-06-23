"""Model factory for industrial defect image classification."""

from __future__ import annotations

from typing import Any, Mapping


def create_model(config: Mapping[str, Any]):
    """Create the configured image classification model."""
    try:
        import torch.nn as nn
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise ImportError(
            "PyTorch and torchvision are required to create models. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    model_config = config.get("model", {})
    model_name = str(model_config.get("name", "resnet18")).lower()
    pretrained = bool(model_config.get("pretrained", True))
    num_classes = int(model_config.get("num_classes", 0))

    if num_classes <= 0:
        raise ValueError("Config value model.num_classes must be a positive integer.")

    if model_name != "resnet18":
        raise ValueError(f"Unsupported model: {model_name!r}. Only 'resnet18' is supported.")

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
