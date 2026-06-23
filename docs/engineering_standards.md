# Engineering Standards

This document defines the engineering conventions for this repository.

## General Principles

- Keep the first version simple and reproducible.
- Prefer readable code over clever abstractions.
- Keep experiment settings in config files.
- Avoid committing datasets, checkpoints, caches, and temporary outputs.
- Every script should be runnable from the repository root.

## Command Style

Planned command examples:

```bash
python src/train.py --config configs/resnet18.yaml
python src/evaluate.py --config configs/resnet18.yaml --checkpoint checkpoints/best.pth
python src/predict.py --config configs/resnet18.yaml --checkpoint checkpoints/best.pth --image demo/sample.jpg
```

Each command should provide helpful error messages when files are missing.

## Configuration Rules

Config files should control:

```text
dataset root
class names
model name
image size
batch size
epochs
learning rate
optimizer
scheduler
random seed
num workers
device
output directory
checkpoint directory
```

Avoid hiding important experiment settings inside source code.

## Reproducibility Rules

Training scripts should:

- Set a random seed.
- Save the used config into the output directory.
- Log train and validation metrics per epoch.
- Save the best checkpoint according to validation macro F1 or validation accuracy.
- Record the final test metrics separately from validation metrics.

## Metric Rules

The project should report:

```text
accuracy
precision
recall
macro F1
confusion matrix
per-class accuracy
```

Accuracy alone is not enough because industrial defect datasets may be imbalanced.

## File Naming Rules

Use clear lowercase names:

```text
resnet18.yaml
efficientnet_b0.yaml
confusion_matrix.png
sample_predictions.png
metrics.json
best.pth
```

Prefer underscores over spaces.

## Git Rules

Commit source code, configs, and documentation.

Do not commit:

```text
data/train/
data/val/
data/test/
checkpoints/*.pth
checkpoints/*.pt
outputs/*
__pycache__/
.ipynb_checkpoints/
```

Small representative images may be committed later only if they are license-safe and useful for the README.

## README Quality Bar

Before the repository is considered presentable, `README.md` should include:

- Project overview.
- Dataset preparation.
- Installation.
- Training command.
- Evaluation command.
- Prediction command.
- Result table.
- Confusion matrix image.
- Example predictions.
- Project structure.
- Roadmap.

