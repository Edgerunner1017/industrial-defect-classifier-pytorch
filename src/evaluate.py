"""Evaluation entry point for a trained industrial defect classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


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


def _collect_predictions(model, dataloader, device) -> tuple[list[int], list[int]]:
    import torch

    y_true: list[int] = []
    y_pred: list[int] = []

    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    return y_true, y_pred


def _save_confusion_matrix(
    matrix: Any, class_names: list[str], output_path: Path
) -> None:
    import matplotlib.pyplot as plt

    from utils import ensure_dir

    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > matrix.max() / 2 else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()

    from datasets import create_dataloaders
    from models import create_model
    from utils import (
        ensure_dir,
        get_device,
        load_checkpoint,
        load_config,
        print_config_summary,
        save_json,
    )

    try:
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for evaluation metrics. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    config = load_config(args.config)
    print_config_summary(config)

    device = get_device(config)
    dataloaders = create_dataloaders(config)
    checkpoint = load_checkpoint(args.checkpoint, device=device)

    model = create_model(config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    class_names = list(checkpoint.get("class_names", dataloaders["class_names"]))
    y_true, y_pred = _collect_predictions(model, dataloaders["test_loader"], device)

    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    output_dir = ensure_dir(config.get("outputs", {}).get("output_dir", "outputs"))
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "macro_f1": float(macro_f1),
        "num_samples": len(y_true),
        "class_names": class_names,
        "checkpoint": str(args.checkpoint),
    }

    metrics_path = save_json(metrics, output_dir / "test_metrics.json")
    confusion_path = output_dir / "confusion_matrix.png"
    _save_confusion_matrix(matrix, class_names, confusion_path)

    print("\nEvaluation complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {confusion_path}")


if __name__ == "__main__":
    main()
