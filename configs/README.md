# Config Guide

This directory will store experiment configuration files.

Planned configs:

```text
resnet18.yaml
efficientnet_b0.yaml
```

Each config should define:

```text
project name
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
checkpoint directory
output directory
```

The first implementation should avoid hard-coded experiment settings in Python files whenever a config value is appropriate.

