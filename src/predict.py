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

    from models import create_model
    from transforms import build_eval_transforms
    from utils import get_device, load_checkpoint, load_config, print_config_summary

    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "PyTorch and Pillow are required for prediction. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if not args.image.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    config = load_config(args.config)
    print_config_summary(config)

    device = get_device(config)
    checkpoint = load_checkpoint(args.checkpoint, device=device)

    model = create_model(config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    class_names = list(
        checkpoint.get("class_names", config.get("data", {}).get("classes", []))
    )
    if not class_names:
        raise ValueError("Class names are missing from checkpoint and config.")

    transform = build_eval_transforms(config)
    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_k = min(3, len(class_names))
    top_probs, top_indices = probabilities.topk(top_k)

    best_index = int(top_indices[0].item())
    print(f"\nPrediction: {class_names[best_index]}")
    print(f"Confidence: {float(top_probs[0].item()):.4f}")
    print("\nTop-3:")

    for rank, (probability, index) in enumerate(zip(top_probs, top_indices), start=1):
        class_index = int(index.item())
        print(f"{rank}. {class_names[class_index]}: {float(probability.item()):.4f}")


if __name__ == "__main__":
    main()
