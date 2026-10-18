# Algorithm Dependent Representation Learning

Experiments on how matrix learning rates affect representations in two-layer linear networks, comparing trained representations with theoretical predictions.

## Setup

Requires uv and an NVIDIA GPU with a CUDA 13.0-compatible driver. The project uses Python 3.13 and PyTorch 2.14.0.

```bash
uv python install 3.13
uv sync --locked
```

## Run

```bash
uv run --locked python run.py --config config_matrices.yaml --run-id matrices01
uv run --locked python plot.py --run-id matrices01
```

The configuration compares SGD, random diagonal learning-rate matrices, and two dense positive definite matrix pairs. Each condition runs three repetitions of 20,000 updates. Use a new run ID for each experiment suite.

Training records, checkpoints, and figures are saved in `records/`. The 2×2 figure shows prediction error, the gradient penalty (Entropy), relative Gram error, and trace-normalized Gram distance. Entropy is a diagnostic and is not added to the training loss.

## Configurations

- [config_matrices.yaml](config_matrices.yaml): vary both layers' learning-rate matrices.
- [config.yaml](config.yaml): SGD and changes to one layer's learning-rate matrix at a time.
- [config_anisotropic.yaml](config_anisotropic.yaml): use nonidentity input and noise covariances.
