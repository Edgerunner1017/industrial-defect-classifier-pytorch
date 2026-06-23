# Roadmap

This roadmap breaks the project into practical development milestones.

## Phase 1: Dataset Definition

Status: complete.

Decisions:

```text
Task: metal surface defect image classification
Dataset format: ImageFolder
Input size: 224x224
Baseline model: ResNet18
Main metrics: accuracy, macro F1, confusion matrix
```

Recommended class names:

```text
scratch
crack
inclusion
pitted_surface
rolled_scale
patches
```

## Phase 2: Repository Structure

Status: in progress.

Goals:

- Create public-facing README.
- Define source code responsibilities.
- Define dataset, output, and checkpoint rules.
- Define engineering standards.

Completion criteria:

- `README.md` exists.
- `docs/project_structure.md` exists.
- `docs/engineering_standards.md` exists.
- `data/README.md` exists.
- `outputs/README.md` exists.
- `checkpoints/README.md` exists.

## Phase 3: Minimum Training Pipeline

Status: planned.

Goals:

- Implement `src/datasets.py`.
- Implement `src/models.py`.
- Implement `src/transforms.py`.
- Implement `src/train.py`.
- Add `configs/resnet18.yaml`.

Completion criteria:

```bash
python src/train.py --config configs/resnet18.yaml
```

The command should train a ResNet18 model and save the best checkpoint.

## Phase 4: Evaluation and Prediction

Status: planned.

Goals:

- Implement `src/evaluate.py`.
- Implement `src/predict.py`.
- Generate `metrics.json`.
- Generate `confusion_matrix.png`.
- Generate sample prediction visualizations.

Completion criteria:

```bash
python src/evaluate.py --config configs/resnet18.yaml --checkpoint checkpoints/best.pth
python src/predict.py --config configs/resnet18.yaml --checkpoint checkpoints/best.pth --image demo/sample.jpg
```

## Phase 5: Model Comparison

Status: planned.

Goals:

- Add EfficientNet-B0 config.
- Compare ResNet18 and EfficientNet-B0.
- Update README result table.

Suggested table:

```text
Model          Accuracy   Macro F1   Notes
ResNet18       TBD        TBD        baseline
EfficientNetB0 TBD        TBD        lightweight comparison
```

## Phase 6: GitHub Polish

Status: planned.

Goals:

- Complete README.
- Add result images.
- Add experiment notes.
- Add known limitations.
- Add future work.

Future extensions:

- Object detection with YOLO.
- Semantic segmentation for defect regions.
- ONNX export.
- TensorRT deployment.
- Simple web or desktop demo.

