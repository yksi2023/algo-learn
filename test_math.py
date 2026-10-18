"""Algebra and data checks to run on the server with ``uv run pytest``.

These checks cover the proposition and interfaces without asserting that SGD
must converge to the theoretical selection target.
"""

import copy
import json
import math

import pytest
import torch

from evaluate import evaluate
from problem import make_problem, make_rule, sample_batch
from theory import compute_target, construct_minimizer, gradient_penalty


def config(d=4, m=4, h=8, rank=4):
    return {
        "problem": {
            "dimensions": {"d": d, "m": m, "h": h, "rank": rank},
            "V": {"kind": "diagonal", "min": 1, "max": 4},
            "Sigma_x": {"kind": "identity"},
            "Sigma_epsilon": {"kind": "identity"},
        },
        "seeds": {"problem": 11},
        "numerics": {"rtol": 1e-10, "atol": 1e-12, "gap_rtol": 1e-8},
    }


def assert_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_diagonal_known_target_and_second_layer_scaling():
    problem = make_problem(config())
    rule = make_rule({}, problem)
    target = compute_target(problem, rule)
    assert_close(target["K_theory"], torch.diag(torch.tensor([4, 3, 2, 1], dtype=torch.float64)))
    scaled_rule = make_rule({"Lambda2": {"kind": "scaled_identity", "scale": 4}}, problem)
    scaled_target = compute_target(problem, scaled_rule)
    assert_close(scaled_target["K_theory"], target["K_theory"] / 2)
    assert scaled_target["min_penalty"] == pytest.approx(2 * target["min_penalty"], rel=1e-10)


def test_diagonal_first_layer_weighting():
    problem = make_problem(config())
    rule = make_rule({"Lambda1": {"kind": "diagonal", "values": [1, 1, 1, 4, 5, 6, 7, 8]}}, problem)
    target = compute_target(problem, rule)
    assert_close(target["K_theory"], torch.diag(torch.tensor([4, 3, 2, 2], dtype=torch.float64)))
    active_rule = make_rule({"Lambda1": {"kind": "active_linear", "min": 1, "max": 4, "rest": 5}}, problem)
    assert_close(torch.diag(active_rule["Lambda1"]), torch.tensor([1, 2, 3, 4, 5, 5, 5, 5], dtype=torch.float64))


def test_dense_rectangular_rank_deficient_minimizer():
    cfg = config(d=5, m=3, h=2, rank=2)
    cfg["problem"].update({
        "V": {"kind": "svd", "values": [3.0, 1.2]},
        "Sigma_x": {"kind": "spd", "min": 0.3, "max": 2.0},
        "Sigma_epsilon": {"kind": "spd", "min": 0.7, "max": 1.8},
    })
    problem = make_problem(cfg)
    rule = make_rule({
        "Lambda1": {"kind": "spd", "min": 0.8, "max": 1.9},
        "Lambda2": {"kind": "spd", "min": 0.9, "max": 3.0},
    }, problem)
    target = compute_target(problem, rule)
    W1, W2 = construct_minimizer(problem, rule, target)
    assert W1.shape == (2, 5)
    assert W2.shape == (3, 2)
    assert_close(W2 @ W1, problem["V"])
    assert_close(W1.T @ W1, target["K_theory"])
    assert gradient_penalty(problem, rule, W1, W2) == pytest.approx(target["min_penalty"], rel=1e-10, abs=1e-12)
    assert int(torch.linalg.matrix_rank(target["K_theory"])) == 2
    metrics = evaluate(W1, W2, problem, rule, target, batch_size=16, eta=0.001)
    assert set(metrics) == {"fit_excess", "k_relative_error", "shape_error", "entropy"}
    assert metrics["fit_excess"] < 1e-20
    assert metrics["k_relative_error"] < 1e-10
    assert metrics["shape_error"] < 1e-10
    for name in ("Sigma_x", "Sigma_epsilon"):
        assert_close(problem[f"{name}_sqrt"] @ problem[f"{name}_sqrt"], problem[name])
        identity = torch.eye(problem[name].shape[0], dtype=torch.float64)
        assert_close(problem[f"{name}_sqrt"] @ problem[f"{name}_inv_sqrt"], identity)


def test_named_task_streams_are_independent():
    cfg = config()
    cfg["problem"]["V"]["kind"] = "svd"
    first = make_problem(cfg)
    changed = copy.deepcopy(cfg)
    changed["problem"]["Sigma_x"] = {"kind": "spd", "min": 0.5, "max": 2}
    changed["problem"]["Sigma_epsilon"] = {"kind": "spd", "min": 1, "max": 3}
    second = make_problem(changed)
    assert torch.equal(first["V"], second["V"])
    third_config = copy.deepcopy(changed)
    third_config["problem"]["V"] = {"kind": "diagonal", "values": [8, 6, 3, 1]}
    third = make_problem(third_config)
    assert torch.equal(second["Sigma_x"], third["Sigma_x"])
    assert torch.equal(second["Sigma_epsilon"], third["Sigma_epsilon"])


@pytest.mark.parametrize("kind", ["diagonal", "spd"])
def test_uniform_matrix_spectrum_is_seeded_and_preserves_global_rng(kind):
    problem = make_problem(config())
    spec = {"kind": kind, "spectrum": "uniform", "min": 0.5, "max": 3.0, "seed": 17}
    before = torch.random.get_rng_state().clone()
    first = make_rule({"Lambda1": spec}, problem)["Lambda1"]
    repeated = make_rule({"Lambda1": spec}, problem)["Lambda1"]
    changed = make_rule({"Lambda1": spec | {"seed": 18}}, problem)["Lambda1"]
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert torch.equal(torch.random.get_rng_state(), before)
    eigenvalues = torch.linalg.eigvalsh(first)
    assert bool((eigenvalues >= 0.5).all() & (eigenvalues <= 3.0).all())


def test_uniform_diagonal_and_dense_matrices_share_the_seeded_spectrum():
    problem = make_problem(config())
    spec = {"spectrum": "uniform", "min": 0.5, "max": 3.0, "seed": 17}
    diagonal = make_rule({"Lambda1": spec | {"kind": "diagonal"}}, problem)["Lambda1"]
    dense = make_rule({"Lambda1": spec | {"kind": "spd"}}, problem)["Lambda1"]
    values = torch.diag(diagonal)
    assert_close(torch.linalg.eigvalsh(dense), values.sort().values)
    assert not torch.equal(values, values.sort().values)
    assert not torch.allclose(values.sort().values, torch.linspace(0.5, 3.0, 8, dtype=torch.float64))
    assert torch.count_nonzero(dense - torch.diag(torch.diag(dense))) > 0


def test_sample_batch_uses_only_supplied_rng_and_symmetric_roots():
    global_state = torch.random.get_rng_state().clone()
    cfg = config()
    cfg["problem"]["V"]["kind"] = "svd"
    cfg["problem"]["Sigma_x"] = {"kind": "spd", "min": 0.5, "max": 2}
    cfg["problem"]["Sigma_epsilon"] = {"kind": "spd", "min": 1, "max": 3}
    problem = make_problem(cfg)
    rng = torch.Generator().manual_seed(207)
    reference_rng = torch.Generator().manual_seed(207)
    X, Y = sample_batch(problem, 7, rng)
    expected_X = torch.randn(7, 4, generator=reference_rng, dtype=torch.float64) @ problem["Sigma_x_sqrt"]
    expected_noise = torch.randn(7, 4, generator=reference_rng, dtype=torch.float64) @ problem["Sigma_epsilon_sqrt"]
    assert_close(X, expected_X)
    assert_close(Y, expected_X @ problem["V"].T + expected_noise)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    assert torch.equal(rng.get_state(), reference_rng.get_state())


def test_evaluation_is_pure_and_zero_representation_has_no_shape():
    problem = make_problem(config())
    rule = make_rule({}, problem)
    target = compute_target(problem, rule)
    W1 = torch.zeros(8, 4, dtype=torch.float64)
    W2 = torch.zeros(4, 8, dtype=torch.float64)
    before = torch.random.get_rng_state().clone()
    metrics = evaluate(W1, W2, problem, rule, target, batch_size=16, eta=0.001)
    assert metrics["fit_excess"] == pytest.approx(30)
    assert metrics["k_relative_error"] == pytest.approx(1)
    assert math.isnan(metrics["shape_error"])
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.count_nonzero(W1) == 0
    assert torch.count_nonzero(W2) == 0


@pytest.mark.parametrize("suffix", ["json", "csv", "pt"])
def test_matrix_files_are_relative_to_config_directory(tmp_path, suffix):
    values = [[2.123456789012345, 0.0], [0.0, 1.0]]
    path = tmp_path / f"matrix.{suffix}"
    if suffix == "json":
        path.write_text(json.dumps({"matrix": values}), encoding="utf-8")
        spec = {"kind": "file", "path": path.name, "key": "matrix"}
    elif suffix == "csv":
        path.write_text("2.123456789012345,0\n0,1\n", encoding="utf-8")
        spec = {"kind": "file", "path": path.name}
    else:
        torch.save({"matrix": torch.tensor(values, dtype=torch.float64)}, path)
        spec = {"kind": "file", "path": path.name, "key": "matrix"}
    cfg = config(d=2, m=2, h=2, rank=2)
    cfg["problem"]["V"] = spec
    problem = make_problem(cfg, base_dir=tmp_path)
    assert torch.equal(problem["V"], torch.tensor(values, dtype=torch.float64))


@pytest.mark.parametrize("bad_matrix, message", [
    ([[1, 0], [0, 0]], "positive definite"),
    ([[1, 0], [0, -1]], "positive definite"),
    ([[1, 0.5], [0, 1]], "symmetric"),
    ([[1, 0, 0], [0, 1, 0]], "shape"),
    ([[1, 0], [0, float("nan")]], "non-finite"),
])
def test_invalid_covariances_are_rejected(bad_matrix, message):
    cfg = config(d=2, m=2, h=2, rank=2)
    cfg["problem"]["Sigma_x"] = {"kind": "values", "values": bad_matrix}
    with pytest.raises(ValueError, match=message):
        make_problem(cfg)


def test_declared_rank_and_hidden_width_are_checked():
    cfg = config(d=2, m=2, h=2, rank=2)
    cfg["problem"]["V"] = {"kind": "values", "values": [[1, 0], [0, 0]]}
    with pytest.raises(ValueError, match="numerical rank"):
        make_problem(cfg)
    cfg["problem"]["dimensions"]["h"] = 1
    with pytest.raises(ValueError, match="rank"):
        make_problem(cfg)


@pytest.mark.parametrize("values", [[1.0, 1.0], [1.0 + 1e-10, 1.0]])
def test_repeated_weighted_singular_values_are_rejected(values):
    cfg = config(d=2, m=2, h=2, rank=2)
    cfg["problem"]["V"] = {"kind": "diagonal", "values": values}
    problem = make_problem(cfg)
    with pytest.raises(ValueError, match="repeated or too close"):
        compute_target(problem, make_rule({}, problem))


def test_penalty_helper_rejects_off_constraint_weights():
    problem = make_problem(config())
    rule = make_rule({}, problem)
    with pytest.raises(ValueError, match="requires W2 @ W1"):
        gradient_penalty(problem, rule, torch.zeros(8, 4, dtype=torch.float64), torch.zeros(4, 8, dtype=torch.float64))


def test_rank_one_tasks_are_supported():
    problem = make_problem(config(d=3, m=2, h=4, rank=1))
    rule = make_rule({}, problem)
    target = compute_target(problem, rule)
    W1, W2 = construct_minimizer(problem, rule, target)
    assert_close(W2 @ W1, problem["V"])
    assert_close(W1.T @ W1, target["K_theory"])


@pytest.mark.parametrize("spec, message", [
    ({"kind": "identity", "scale": 4}, "inapplicable fields"),
    ({"kind": "spd", "specrum": "linear"}, "inapplicable fields"),
    ({"kind": "spd", "values": [1, 2, 3, 4], "min": 1}, "cannot be combined"),
    ({"kind": "spd", "values": [1, 2, 3, 4], "spectrum": "linear"}, "cannot be combined"),
])
def test_invalid_or_ignored_matrix_fields_are_rejected(spec, message):
    cfg = config()
    cfg["problem"]["Sigma_x"] = spec
    with pytest.raises(ValueError, match=message):
        make_problem(cfg)


def test_v_and_rule_specs_reject_inapplicable_fields():
    cfg = config()
    cfg["problem"]["V"] = {"kind": "svd", "values": [4, 3, 2, 1], "max": 4}
    with pytest.raises(ValueError, match="cannot be combined"):
        make_problem(cfg)
    problem = make_problem(config())
    with pytest.raises(ValueError, match="inapplicable fields"):
        make_rule({"Lambda2": {"kind": "identity", "scale": 4}}, problem)
