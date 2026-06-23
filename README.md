# Industrial Defect Classifier PyTorch

A PyTorch image classification project for industrial metal surface defect recognition.

This repository is designed as a learning and portfolio project. The goal is not only to train a model, but also to build a reproducible computer vision engineering workflow that can be presented on GitHub.

## Project Goal

Build an end-to-end image classification system for metal surface defects:

- Prepare an ImageFolder-style industrial defect dataset.
- Train a ResNet18 baseline with PyTorch.
- Evaluate the model with accuracy, precision, recall, macro F1, and confusion matrix.
- Run single-image prediction from the command line.
- Keep experiments reproducible and easy to review.

## Target Task

Input:

```text
One industrial surface image
```

Output:

```text
Predicted defect class and confidence score
```

Recommended first-version classes:

```text
scratch
crack
inclusion
pitted_surface
rolled_scale
patches
```

The exact class names may change depending on the selected dataset. Any mapping should be documented in `data/README.md`.

## Planned Repository Structure

```text
industrial-defect-classifier-pytorch/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- configs/
|   |-- README.md
|   |-- resnet18.yaml
|   `-- efficientnet_b0.yaml
|-- data/
|   `-- README.md
|-- src/
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|   |-- datasets.py
|   |-- models.py
|   |-- transforms.py
|   `-- utils.py
|-- notebooks/
|   `-- data_exploration.ipynb
|-- outputs/
|   `-- README.md
|-- checkpoints/
|   `-- README.md
`-- docs/
    |-- project_structure.md
    |-- engineering_standards.md
    `-- roadmap.md
```

## Development Phases

1. Define the dataset format, class names, split rules, and evaluation metrics.
2. Create the repository structure and engineering standards.
3. Implement the minimum training pipeline with ResNet18.
4. Add evaluation, confusion matrix, and prediction scripts.
5. Compare models such as ResNet18 and EfficientNet-B0.
6. Polish the README, experiment logs, and GitHub presentation.

## Current Status

Step 1 is complete:

- Task direction: metal surface defect classification.
- Dataset format: ImageFolder.
- Baseline model: ResNet18.
- Main metrics: accuracy, macro F1, and confusion matrix.

Step 2 is complete:

- Repository structure.
- Documentation rules.
- Engineering conventions.

Step 3 is in progress:

- Minimal command-line skeleton.
- ResNet18 baseline config draft.
- Source module placeholders for dataset, model, transforms, and utilities.

## Documentation

- [Project Structure](docs/project_structure.md)
- [Engineering Standards](docs/engineering_standards.md)
- [Roadmap](docs/roadmap.md)
- [Dataset Guide](data/README.md)
- [Outputs Guide](outputs/README.md)
- [Checkpoints Guide](checkpoints/README.md)
