"""Training integration checks require a CUDA GPU on the target server."""

import copy
import csv
import math

import pytest
import torch

from evaluate import evaluate
from problem import make_problem, make_rule, sample_batch
from theory import compute_target
import train as training


def small_config(tmp_path, run_id="check", *, steps=10, eval_every=3, checkpoint_every=4):
    return {
        "problem": {
            "dimensions": {"d": 5, "m": 3, "h": 2, "rank": 2},
            "V": {"kind": "svd", "spectrum": "linear", "max": 2.0, "min": 1.0},
            "Sigma_x": {"kind": "identity"},
            "Sigma_epsilon": {"kind": "identity"},
        },
        "rules": {"sgd": {"Lambda1": {"kind": "identity"}, "Lambda2": {"kind": "identity"}}},
        "train": {
            "eta": 0.001,
            "batch_size": 7,
            "steps": steps,
            "init_scale": 0.1,
            "logging": {"eval_every": eval_every, "checkpoint_every": checkpoint_every, "tail_fraction": 0.2},
        },
        "seeds": {"problem": 11, "init": 101, "data": 201},
        "runtime": {
            "device": "cuda:0", "dtype": "float64", "records_dir": str(tmp_path),
            "run_id": run_id, "suite_id": "checks",
        },
        "rule_name": "sgd",
        "repeat": 0,
    }


def build_task(config):
    problem = make_problem(config, seed=config["seeds"]["problem"])
    rule = make_rule(config["rules"]["sgd"], problem)
    return problem, rule, compute_target(problem, rule)


def read_history(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_two_updates_match_hand_gradients_with_rectangular_weights_and_dense_spd():
    rng = torch.Generator(device="cpu").manual_seed(123)
    W1 = torch.randn((2, 5), generator=rng, dtype=torch.float64)
    W2 = torch.randn((3, 2), generator=rng, dtype=torch.float64)
    X = torch.randn((7, 5), generator=rng, dtype=torch.float64)
    Y = torch.randn((7, 3), generator=rng, dtype=torch.float64)
    first = torch.tensor([[1.0, 0.2], [0.2, 2.0]], dtype=torch.float64)
    second = torch.tensor([[2.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.5]], dtype=torch.float64)
    rule = {"Lambda1": first, "Lambda2": second}
    expected_W1, expected_W2 = W1.clone(), W2.clone()
    eta = 0.001

    # Existing .grad buffers must have no influence on either update.
    W1.requires_grad_(True)
    W2.requires_grad_(True)
    W1.grad = torch.full_like(W1, 100.0)
    W2.grad = torch.full_like(W2, -100.0)
    for _ in range(2):
        error = X @ expected_W1.T @ expected_W2.T - Y
        G2 = (2.0 / len(X)) * error.T @ (X @ expected_W1.T)
        G1 = (2.0 / len(X)) * expected_W2.T @ error.T @ X
        expected_loss = float(error.square().sum() / len(X))
        expected_W1 = expected_W1 - eta * first @ G1
        expected_W2 = expected_W2 - eta * second @ G2
        actual_loss = training.train_step(W1, W2, X, Y, rule, eta)
        assert actual_loss == pytest.approx(expected_loss, rel=1e-12)
        torch.testing.assert_close(W1, expected_W1, rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(W2, expected_W2, rtol=1e-10, atol=1e-12)
    assert torch.equal(W1.grad, torch.full_like(W1, 100.0))
    assert torch.equal(W2.grad, torch.full_like(W2, -100.0))


def test_initialization_uses_documented_variances_and_only_supplied_rng(tmp_path):
    config = small_config(tmp_path)
    problem, _, _ = build_task(config)
    rng = torch.Generator(device="cpu").manual_seed(42)
    reference_rng = torch.Generator(device="cpu").manual_seed(42)
    actual_W1, actual_W2 = training.initialize(problem, config, rng)
    expected_W1 = torch.randn((2, 5), generator=reference_rng, dtype=torch.float64) * (0.1 / 5**0.5)
    expected_W2 = torch.randn((3, 2), generator=reference_rng, dtype=torch.float64) * (0.1 / 2**0.5)
    assert torch.equal(actual_W1, expected_W1)
    assert torch.equal(actual_W2, expected_W2)
    assert torch.equal(rng.get_state(), reference_rng.get_state())


def test_paired_rules_receive_identical_initial_weights_and_samples(tmp_path, monkeypatch):
    config = small_config(tmp_path, "baseline", steps=2)
    problem, rule, target = build_task(config)
    batches = []
    train_step = training.train_step
    devices = []

    def recording_sampler(problem, n, rng):
        X, Y = sample_batch(problem, n, rng)
        batches.append((X.clone(), Y.clone()))
        return X, Y

    def recording_step(W1, W2, X, Y, rule, eta):
        tensors = (W1, W2, X, Y, rule["Lambda1"], rule["Lambda2"])
        assert all(value.device == torch.device("cuda:0") for value in tensors)
        devices.append(W1.device)
        return train_step(W1, W2, X, Y, rule, eta)

    monkeypatch.setattr(training, "sample_batch", recording_sampler)
    monkeypatch.setattr(training, "train_step", recording_step)
    first = training.train(problem, rule, target, config)
    changed_config = copy.deepcopy(config)
    changed_config["runtime"]["run_id"] = "changed"
    changed_rule = {"Lambda1": rule["Lambda1"], "Lambda2": 4 * rule["Lambda2"]}
    second = training.train(problem, changed_rule, compute_target(problem, changed_rule), changed_config)
    assert len(devices) == 4
    for name in ("initial_W1", "initial_W2"):
        assert torch.equal(first[name], second[name])
    for index in range(2):
        for expected, actual in zip(batches[index], batches[index + 2]):
            assert torch.equal(expected, actual)


def test_eval_frequency_does_not_change_weights_or_rng(tmp_path):
    config = small_config(tmp_path, "frequent", steps=11, eval_every=1)
    problem, rule, target = build_task(config)
    first = training.train(problem, rule, target, config)
    changed_config = copy.deepcopy(config)
    changed_config["runtime"]["run_id"] = "sparse"
    changed_config["train"]["logging"]["eval_every"] = 5
    second = training.train(problem, rule, target, changed_config)
    for name in ("W1", "W2", "data_rng_state"):
        assert torch.equal(first[name], second[name])
    assert first["history"][0]["step"] == second["history"][0]["step"] == 0
    assert first["history"][-1]["step"] == second["history"][-1]["step"] == 11
    assert {row["step"] for row in second["history"]} == {0, 4, 5, 8, 10, 11}


def test_resume_matches_uninterrupted_training_and_repairs_csv_tail(tmp_path):
    config = small_config(tmp_path, "whole", steps=10)
    problem, rule, target = build_task(config)
    whole = training.train(problem, rule, target, config)
    split_config = copy.deepcopy(config)
    split_config["runtime"]["run_id"] = "split"
    split_config["train"]["steps"] = 6
    training.train(problem, rule, target, split_config)
    saved = torch.load(tmp_path / "split.pt", map_location="cpu", weights_only=True)
    with (tmp_path / "split.csv").open("a", encoding="utf-8") as handle:
        handle.write("999,1,1,1,1000\n")
    resumed_config = copy.deepcopy(split_config)
    resumed_config["train"]["steps"] = 10
    resumed = training.train(problem, rule, target, resumed_config, state=saved)
    for name in ("W1", "W2", "initial_W1", "initial_W2", "data_rng_state"):
        assert torch.equal(whole[name], resumed[name])
    assert resumed["config"]["train"]["steps"] == 10
    assert resumed["step"] == 10
    rows = read_history(tmp_path / "split.csv")
    assert [int(row["step"]) for row in rows] == [0, 3, 4, 6, 8, 9, 10]
    assert len({row["step"] for row in rows}) == len(rows)
    for first, second in zip(whole["history"], resumed["history"]):
        for name in ("step", "fit_excess", "entropy", "k_relative_error", "shape_error"):
            assert first[name] == second[name]


def test_saved_metrics_match_each_weight_snapshot(tmp_path):
    config = small_config(tmp_path, steps=9)
    problem, rule, target = build_task(config)
    training.train(problem, rule, target, config)
    saved = torch.load(tmp_path / "check.pt", map_location="cpu", weights_only=True)
    csv_records = {int(row["step"]): row for row in read_history(tmp_path / "check.csv")}
    assert [snapshot["step"] for snapshot in saved["snapshots"]] == [0, 4, 8, 9]
    for snapshot in saved["snapshots"]:
        calculated = evaluate(
            snapshot["W1"], snapshot["W2"], saved["problem"], saved["rule"], saved["target"],
            batch_size=config["train"]["batch_size"], eta=config["train"]["eta"],
        )
        for name, value in calculated.items():
            assert float(csv_records[snapshot["step"]][name]) == pytest.approx(value, rel=1e-10, abs=1e-12)
    for name in ("W1", "W2", "initial_W1", "initial_W2"):
        assert saved[name].device.type == "cpu"
        assert saved[name].dtype == torch.float64
        assert not saved[name].requires_grad


def test_training_does_not_consume_the_global_cpu_rng(tmp_path):
    config = small_config(tmp_path, steps=2)
    problem, rule, target = build_task(config)
    before = torch.random.get_rng_state().clone()
    training.train(problem, rule, target, config)
    assert torch.equal(before, torch.random.get_rng_state())


def test_resume_older_checkpoint_keeps_unrecorded_entropy_missing(tmp_path):
    config = small_config(tmp_path, steps=2)
    problem, rule, target = build_task(config)
    state = training.train(problem, rule, target, config)
    for row in state["history"]:
        del row["entropy"]
    config["train"]["steps"] = 4
    resumed = training.train(problem, rule, target, config, state=state)
    assert all(math.isnan(row["entropy"]) for row in resumed["history"] if row["step"] <= 2)
    assert all(math.isfinite(row["entropy"]) for row in resumed["history"] if row["step"] > 2)


def test_nonfinite_update_keeps_last_valid_checkpoint(tmp_path, monkeypatch):
    config = small_config(tmp_path, steps=8, eval_every=1, checkpoint_every=3)
    problem, rule, target = build_task(config)
    count = 0

    def failing_sampler(problem, n, rng):
        nonlocal count
        count += 1
        X, Y = sample_batch(problem, n, rng)
        if count == 5:
            Y.fill_(float("inf"))
        return X, Y

    monkeypatch.setattr(training, "sample_batch", failing_sampler)
    with pytest.raises(FloatingPointError, match="Step 5"):
        training.train(problem, rule, target, config)
    saved = torch.load(tmp_path / "check.pt", map_location="cpu", weights_only=True)
    assert saved["step"] == 3
    assert torch.isfinite(saved["W1"]).all()
    assert torch.isfinite(saved["W2"]).all()
    assert saved["history"][-1]["step"] == 3
    assert read_history(tmp_path / "check.csv")[-1]["step"] == "4"


def test_nonfinite_step_rejects_both_candidates_before_mutation():
    W1 = torch.ones((2, 3), dtype=torch.float64)
    W2 = torch.ones((4, 2), dtype=torch.float64)
    X = torch.ones((2, 3), dtype=torch.float64)
    Y = torch.zeros((2, 4), dtype=torch.float64)
    original_W1, original_W2 = W1.clone(), W2.clone()
    rule = {"Lambda1": torch.eye(2, dtype=torch.float64), "Lambda2": torch.full((4, 4), float("inf"), dtype=torch.float64)}
    with pytest.raises(FloatingPointError):
        training.train_step(W1, W2, X, Y, rule, 0.01)
    assert torch.equal(W1, original_W1)
    assert torch.equal(W2, original_W2)


def test_zero_step_run_saves_initial_metrics_and_final_state(tmp_path):
    config = small_config(tmp_path, steps=0)
    problem, rule, target = build_task(config)
    state = training.train(problem, rule, target, config)
    assert state["step"] == 0
    assert [row["step"] for row in state["history"]] == [0]
    assert [row["step"] for row in state["snapshots"]] == [0]
    assert torch.equal(state["W1"], state["initial_W1"])
    assert torch.equal(state["W2"], state["initial_W2"])


def test_zero_gram_has_undefined_shape_error_and_can_resume(tmp_path, monkeypatch):
    config = small_config(tmp_path, steps=0)
    problem, rule, target = build_task(config)

    def zero_initialize(problem, config, rng):
        return (
            torch.zeros((problem["h"], problem["d"]), dtype=torch.float64),
            torch.zeros((problem["m"], problem["h"]), dtype=torch.float64),
        )

    monkeypatch.setattr(training, "initialize", zero_initialize)
    initial = training.train(problem, rule, target, config)
    assert math.isnan(initial["history"][0]["shape_error"])
    config["train"]["steps"] = 2
    resumed = training.train(problem, rule, target, config, state=initial)
    assert resumed["step"] == 2
    for row in resumed["history"]:
        assert math.isfinite(row["fit_excess"])
        assert row["k_relative_error"] == pytest.approx(1.0)
        assert math.isnan(row["shape_error"])
    assert [row["shape_error"] for row in read_history(tmp_path / "check.csv")] == ["nan", "nan"]
