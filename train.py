"""Online SGD with paired random streams and resumable training records."""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
import time

import torch

from evaluate import evaluate
from problem import sample_batch


CSV_FIELDS = (
    "step", "fit_excess", "entropy", "k_relative_error", "shape_error", "elapsed_seconds",
)


def initialize(problem: dict, config: dict, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw CPU weights using the supplied initialization generator."""
    scale = config["train"]["init_scale"]
    d, m, h = (problem[key] for key in ("d", "m", "h"))
    W1 = torch.randn((h, d), generator=rng, dtype=torch.float64, device="cpu")
    W2 = torch.randn((m, h), generator=rng, dtype=torch.float64, device="cpu")
    W1.mul_(scale / math.sqrt(d))
    W2.mul_(scale / math.sqrt(h))
    return W1, W2


def train_step(
    W1: torch.Tensor,
    W2: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    rule: dict,
    eta: float,
) -> float:
    """Update both layers using gradients evaluated at the same old weights.

    The loss sums over output coordinates and averages over the batch.
    """
    with torch.enable_grad():
        first = W1.detach().requires_grad_(True)
        second = W2.detach().requires_grad_(True)
        prediction = X @ first.T @ second.T
        loss = (prediction - Y).square().sum(dim=1).mean()
        G1, G2 = torch.autograd.grad(loss, (first, second))
    with torch.no_grad():
        next_W1 = W1 - eta * (rule["Lambda1"] @ G1)
        next_W2 = W2 - eta * (rule["Lambda2"] @ G2)
        if not (torch.isfinite(next_W1).all() & torch.isfinite(next_W2).all()):
            raise FloatingPointError("The weight update produced a non-finite value.")
        W1.copy_(next_W1)
        W2.copy_(next_W2)
    return loss.item()


def _on_device(values: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in values.items()
    }


def _atomic_save(state: dict, path: Path) -> None:
    """Keep the preceding checkpoint until the new file has been written."""
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(problem: dict, rule: dict, target: dict, config: dict, state: dict | None = None) -> dict:
    """Train one run; the caller manages configuration and run IDs.

    Resume restores the saved sampling state and rewrites CSV from checkpoint
    history, so records after the last saved checkpoint are discarded.
    """
    settings = config["train"]
    steps, batch_size, eta = (settings[key] for key in ("steps", "batch_size", "eta"))
    eval_every = settings["logging"]["eval_every"]
    checkpoint_every = settings["logging"]["checkpoint_every"]
    device = torch.device(config["runtime"]["device"])
    run_id = config["runtime"]["run_id"]
    records_dir = Path(config["runtime"]["records_dir"])
    checkpoint_path = records_dir / f"{run_id}.pt"
    csv_path = records_dir / f"{run_id}.csv"

    if state is None:
        init_rng = torch.Generator(device="cpu").manual_seed(config["seeds"]["init"])
        data_rng = torch.Generator(device="cpu").manual_seed(config["seeds"]["data"])
        W1, W2 = initialize(problem, config, init_rng)
        initial_W1, initial_W2 = W1.clone(), W2.clone()
        history, snapshots = [], []
        start_step, prior_elapsed = 0, 0.0
    else:
        W1, W2 = state["W1"].clone(), state["W2"].clone()
        initial_W1, initial_W2 = state["initial_W1"], state["initial_W2"]
        history, snapshots = copy.deepcopy(state["history"]), copy.deepcopy(state["snapshots"])
        # Older checkpoints did not record entropy at every evaluation step.
        for row in history:
            row.setdefault("entropy", math.nan)
        start_step, prior_elapsed = state["step"], state["elapsed_seconds"]
        data_rng = torch.Generator(device="cpu").set_state(state["data_rng_state"])

    W1, W2 = W1.to(device), W2.to(device)
    device_problem = _on_device(problem, device)
    device_rule = _on_device(rule, device)
    device_target = _on_device(target, device)
    started = time.perf_counter()

    def metric_record(step: int) -> dict:
        return {
            "step": step,
            **evaluate(W1, W2, device_problem, device_rule, device_target, batch_size=batch_size, eta=eta),
            "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
        }

    def save_checkpoint(step: int) -> dict:
        snapshot = {
            "step": step,
            "W1": W1.detach().cpu().clone(),
            "W2": W2.detach().cpu().clone(),
        }
        if snapshots and snapshots[-1]["step"] == step:
            snapshots[-1] = snapshot
        else:
            snapshots.append(snapshot)
        saved = {
            "format_version": 1,
            "config": copy.deepcopy(config),
            "problem": problem,
            "rule": rule,
            "target": target,
            **snapshot,
            "initial_W1": initial_W1,
            "initial_W2": initial_W2,
            "data_rng_state": data_rng.get_state(),
            "elapsed_seconds": history[-1]["elapsed_seconds"],
            "snapshots": copy.deepcopy(snapshots),
            "history": copy.deepcopy(history),
        }
        _atomic_save(saved, checkpoint_path)
        return saved

    if state is None:
        history.append(metric_record(0))
    records_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        final_state = save_checkpoint(start_step) if state is None or steps == start_step else state
        for step in range(start_step + 1, steps + 1):
            X, Y = sample_batch(problem, batch_size, data_rng)
            try:
                train_step(W1, W2, X.to(device), Y.to(device), device_rule, eta)
            except FloatingPointError as exc:
                raise FloatingPointError(f"Step {step}: {exc}") from exc
            checkpoint_due = step % checkpoint_every == 0 or step == steps
            if step % eval_every == 0 or checkpoint_due:
                record = metric_record(step)
                history.append(record)
                writer.writerow(record)
                handle.flush()
            if checkpoint_due:
                final_state = save_checkpoint(step)
    return final_state
