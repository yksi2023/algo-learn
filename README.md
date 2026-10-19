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

The configuration compares SGD, random diagonal learning-rate matrices, and two dense positive definite matrix pairs. Each condition runs three repetitions of 20,000 updates.

For an SGD-only experiment with five repetitions on the same task:

```bash
uv run --locked python run.py --config config_sgd.yaml --run-id sgd01
uv run --locked python plot.py --run-id sgd01
```

Repetitions use different initialization and data seeds; the task is fixed. Conditions share the same seed pair within each repetition. Use a new run ID for each experiment suite.

Training records and checkpoints are saved in `records/<run-id>/`, alongside `curves.png` and `curves.pdf`. Curves show the mean across seeds, with error bars of ±1 sample standard deviation. The 2×2 figure shows prediction error, the gradient penalty (Entropy), relative Gram error, and trace-normalized Gram distance. Entropy is recorded as a diagnostic and does not enter the training loss.

To move existing flat records into experiment folders, run `uv run --locked python organize_records.py` once.

## Configurations

- [config_matrices.yaml](config_matrices.yaml): vary both layers' learning-rate matrices.
- [config_sgd.yaml](config_sgd.yaml): SGD with identity learning-rate matrices and five seed pairs.
- [config.yaml](config.yaml): SGD and changes to one layer's learning-rate matrix at a time.
- [config_anisotropic.yaml](config_anisotropic.yaml): use nonidentity input and noise covariances.
