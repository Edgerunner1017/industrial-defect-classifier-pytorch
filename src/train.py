"""Training entry point for the industrial defect classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


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


def _create_optimizer(model, config: dict[str, Any]):
    import torch

    training = config.get("training", {})
    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    learning_rate = float(training.get("learning_rate", 0.001))
    weight_decay = float(training.get("weight_decay", 0.0))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name!r}.")


def _create_scheduler(optimizer, config: dict[str, Any]):
    import torch

    training = config.get("training", {})
    scheduler_name = str(training.get("scheduler", "cosine")).lower()
    epochs = int(training.get("epochs", 20))

    if scheduler_name in {"none", "null"}:
        return None

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    raise ValueError(f"Unsupported scheduler: {scheduler_name!r}.")


def _run_epoch(model, dataloader, criterion, device, optimizer=None) -> dict[str, float]:
    import torch

    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    if total == 0:
        raise ValueError("Dataloader produced no samples.")

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def main() -> None:
    args = build_parser().parse_args()

    from datasets import create_dataloaders, validate_imagefolder_layout
    from models import create_model
    from utils import (
        ensure_dir,
        get_device,
        load_config,
        print_config_summary,
        save_checkpoint,
        save_json,
        set_seed,
    )

    config = load_config(args.config)
    print_config_summary(config)

    data_config = config.get("data", {})
    class_names = list(data_config.get("classes", []))

    if args.dry_run:
        print("\nDry run checks:")
        print(f"- Config loaded from: {args.config}")
        print(f"- Expected classes: {class_names}")
        print(f"- Dataset root: {data_config.get('root', 'data')}")
        try:
            validate_imagefolder_layout(data_config.get("root", "data"), class_names)
        except FileNotFoundError as exc:
            print(f"- Dataset layout check: missing data\n{exc}")
        else:
            print("- Dataset layout check: ok")
        print("\nDry run complete. No training was started.")
        return

    import torch

    set_seed(int(config.get("project", {}).get("seed", 42)))
    device = get_device(config)
    dataloaders = create_dataloaders(config)
    model = create_model(config).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = _create_optimizer(model, config)
    scheduler = _create_scheduler(optimizer, config)

    output_dir = ensure_dir(config.get("outputs", {}).get("output_dir", "outputs"))
    checkpoint_dir = ensure_dir(
        config.get("outputs", {}).get("checkpoint_dir", "checkpoints")
    )
    checkpoint_name = config.get("outputs", {}).get(
        "best_checkpoint_name", "resnet18_best.pth"
    )
    checkpoint_path = checkpoint_dir / checkpoint_name

    epochs = int(config.get("training", {}).get("epochs", 20))
    best_val_accuracy = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(
            model,
            dataloaders["train_loader"],
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = _run_epoch(
            model,
            dataloaders["val_loader"],
            criterion,
            device,
        )

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                    if scheduler is not None
                    else None,
                    "best_val_accuracy": best_val_accuracy,
                    "class_names": dataloaders["class_names"],
                    "class_to_idx": dataloaders["class_to_idx"],
                    "config": config,
                },
                checkpoint_path,
            )

    save_json(
        {
            "best_val_accuracy": best_val_accuracy,
            "history": history,
            "checkpoint": str(checkpoint_path),
        },
        output_dir / "training_metrics.json",
    )

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
