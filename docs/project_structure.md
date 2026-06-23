# Project Structure

This document defines the intended repository structure and the responsibility of each directory.

## Root Files

```text
README.md
requirements.txt
.gitignore
```

`README.md` is the public entry point of the project. It should explain the task, installation steps, training commands, evaluation commands, results, and future work.

`requirements.txt` will list the minimum Python dependencies required to run the project.

`.gitignore` prevents datasets, model weights, cache files, and generated outputs from being committed.

## configs/

The `configs/` directory stores experiment configuration files.

Planned files:

```text
configs/
|-- README.md
|-- resnet18.yaml
`-- efficientnet_b0.yaml
```

Each config file should define:

```text
dataset paths
class names
model name
input size
batch size
epochs
learning rate
optimizer
scheduler
random seed
output directory
checkpoint directory
```

Training and evaluation scripts should prefer config values over hard-coded constants.

## data/

The `data/` directory is reserved for local datasets.

Datasets should not be committed to Git unless the files are tiny and license-safe. The repository should only include `data/README.md`, which explains where to download the dataset and how to arrange files.

Expected ImageFolder layout:

```text
data/
|-- train/
|   |-- scratch/
|   |-- crack/
|   `-- ...
|-- val/
|   |-- scratch/
|   |-- crack/
|   `-- ...
`-- test/
    |-- scratch/
    |-- crack/
    `-- ...
```

## src/

The `src/` directory will contain project source code.

Planned modules:

```text
src/
|-- train.py
|-- evaluate.py
|-- predict.py
|-- datasets.py
|-- models.py
|-- transforms.py
`-- utils.py
```

Responsibilities:

```text
train.py       training entry point
evaluate.py    checkpoint evaluation entry point
predict.py     single-image inference entry point
datasets.py    dataset loading helpers
models.py      model construction helpers
transforms.py  image preprocessing and augmentation
utils.py       logging, seed, metrics, and file utilities
```

## notebooks/

The `notebooks/` directory is optional and should only be used for exploration.

Notebook code should not become the main training logic. Once an idea is stable, it should be moved into `src/`.

## outputs/

The `outputs/` directory stores generated experiment artifacts.

Typical files:

```text
outputs/
|-- metrics.json
|-- confusion_matrix.png
|-- training_curves.png
`-- sample_predictions.png
```

Large or frequently regenerated files should not be committed unless they are useful for the README.

## checkpoints/

The `checkpoints/` directory stores trained model weights locally.

Large `.pth` or `.pt` files should not be committed directly. If a final model needs to be shared, use GitHub Releases or another external storage method and document the link in the README.

## docs/

The `docs/` directory stores planning and engineering documents.

Current documents:

```text
docs/project_structure.md
docs/engineering_standards.md
docs/roadmap.md
```
