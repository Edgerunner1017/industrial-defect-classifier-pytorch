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
├── train/
│   ├── scratch/
│   ├── crack/
│   ├── inclusion/
│   ├── pitted_surface/
│   ├── rolled_scale/
│   └── patches/
├── val/
│   ├── scratch/
│   ├── crack/
│   ├── inclusion/
│   ├── pitted_surface/
│   ├── rolled_scale/
│   └── patches/
└── test/
    ├── scratch/
    ├── crack/
    ├── inclusion/
    ├── pitted_surface/
    ├── rolled_scale/
    └── patches/
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

Fill this section after selecting the exact dataset:

```text
Dataset name:
Source URL:
License:
Number of classes:
Number of images:
Image size:
Official split:
Local path:
```

