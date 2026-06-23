"""Model factory for industrial defect image classification."""

from __future__ import annotations

from typing import Any


def create_model(config: dict[str, Any]) -> Any:
    """Create the configured image classification model.

    TODO:
    - Support torchvision ResNet18 as the first baseline.
    - Replace the classification head according to `model.num_classes`.
    - Add EfficientNet-B0 support after the baseline is stable.
    """
    model_name = config.get("model", {}).get("name", "resnet18")
    raise NotImplementedError(f"Model factory for {model_name!r} is not implemented yet.")

