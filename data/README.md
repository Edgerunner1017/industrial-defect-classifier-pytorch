# Dataset Guide

This directory is reserved for local datasets.

Do not commit full datasets to Git. Keep dataset files local and document the source, license, and preparation steps here.

## Selected Direction

The first version of this project uses metal or steel surface defect image classification.

Recommended classes:

```text
scratch
crack
inclusion
pitted_surface
rolled_scale
patches
```

The actual class names may vary by dataset. If the original dataset uses different names, record the mapping here.

Example mapping:

```text
crazing -> crack
scratches -> scratch
rolled-in_scale -> rolled_scale
pitted_surface -> pitted_surface
inclusion -> inclusion
patches -> patches
```

## Expected Directory Layout

Use ImageFolder format:

```text
data/
|-- train/
|   |-- scratch/
|   |-- crack/
|   |-- inclusion/
|   |-- pitted_surface/
|   |-- rolled_scale/
|   `-- patches/
|-- val/
|   |-- scratch/
|   |-- crack/
|   |-- inclusion/
|   |-- pitted_surface/
|   |-- rolled_scale/
|   `-- patches/
`-- test/
    |-- scratch/
    |-- crack/
    |-- inclusion/
    |-- pitted_surface/
    |-- rolled_scale/
    `-- patches/
```

## Split Rule

If the dataset has no official split, use:

```text
train: 70%
val: 15%
test: 15%
```

Use stratified splitting so each class appears in all splits when possible.

## Dataset Notes

Current dataset:

```text
Dataset name: NEU-CLS surface defect dataset
Source URL: local download
License: check the original download source before redistribution
Number of classes: 6
Number of images: 1800
Image format: BMP
Original local path: C:\Users\Wuzheng\Desktop\NEU-CLS\NEU-CLS
Prepared local path: data/
Official split: none in the downloaded folder
Prepared split: train 70%, val 15%, test 15%
Split seed: 42
```

Prepared class counts:

```text
class           train  val  test  total
crack             210   45    45    300
inclusion         210   45    45    300
patches           210   45    45    300
pitted_surface    210   45    45    300
rolled_scale      210   45    45    300
scratch           210   45    45    300
```

The dataset was prepared with:

```bash
python scripts/split_neu_cls.py --source "C:\Users\Wuzheng\Desktop\NEU-CLS\NEU-CLS" --output data
```
