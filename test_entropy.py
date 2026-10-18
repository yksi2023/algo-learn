"""Exact Gaussian integration checks for the gradient penalty along training."""

import itertools
import math

import pytest
import torch

from evaluate import evaluate, expected_gradient_penalty
from problem import make_problem, make_rule
from theory import compute_target, construct_minimizer, gradient_penalty


@pytest.fixture
def dense_task():
    problem = make_problem({
        "problem": {
            "dimensions": {"d": 2, "m": 3, "h": 4, "rank": 2},
            "V": {"kind": "svd", "values": [2.1, 0.8]},
            "Sigma_x": {"kind": "spd", "min": 0.7, "max": 1.8},
            "Sigma_epsilon": {"kind": "spd", "min": 0.4, "max": 1.3},
        },
        "seeds": {"problem": 817},
    })
    rule = make_rule({
        "Lambda1": {"kind": "spd", "min": 0.6, "max": 2.0},
        "Lambda2": {"kind": "spd", "min": 0.8, "max": 1.7},
    }, problem)
    rng = torch.Generator().manual_seed(119)
    W1 = 0.4 * torch.randn(4, 2, generator=rng, dtype=torch.float64)
    W2 = 0.6 * torch.randn(3, 4, generator=rng, dtype=torch.float64)
    return problem, rule, W1, W2


@pytest.mark.parametrize("batch_size", [1, 2])
def test_off_fit_penalty_matches_exact_gaussian_quadrature(dense_task, batch_size):
    problem, rule, W1, W2 = dense_task
    # These nodes reproduce all normal moments through degree five. The
    # squared gradient is quartic, so this expectation has no sampling error.
    nodes = torch.tensor([-math.sqrt(3), 0, math.sqrt(3)], dtype=torch.float64)
    weights = torch.tensor([1 / 6, 2 / 3, 1 / 6], dtype=torch.float64)
    indices = torch.tensor(list(itertools.product(range(3), repeat=5)))
    normal = nodes[indices]
    probabilities = weights[indices].prod(dim=1)
    X = normal[:, :2] @ torch.linalg.cholesky(problem["Sigma_x"]).T
    noise = normal[:, 2:] @ torch.linalg.cholesky(problem["Sigma_epsilon"]).T
    Y = X @ problem["V"].T + noise
    error = X @ W1.T @ W2.T - Y
    G1 = 2 * torch.einsum("nh,nd->nhd", error @ W2, X)
    G2 = 2 * torch.einsum("nm,nh->nmh", error, X @ W1.T)
    if batch_size == 2:
        # Enumerate two independent samples and form their actual batch mean.
        G1 = (G1[:, None] + G1[None, :]) / 2
        G2 = (G2[:, None] + G2[None, :]) / 2
        probabilities = probabilities[:, None] * probabilities[None, :]
    penalties = (
        torch.einsum("...ij,ik,...kj->...", G1, rule["Lambda1"], G1)
        + torch.einsum("...ij,ik,...kj->...", G2, rule["Lambda2"], G2)
    )
    integrated = float((probabilities * penalties).sum())
    actual = expected_gradient_penalty(problem, rule, W1, W2, batch_size)
    assert actual == pytest.approx(integrated, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_exact_fit_penalty_matches_constrained_formula(dense_task, batch_size):
    problem, rule, _, _ = dense_task
    target = compute_target(problem, rule)
    W1, W2 = construct_minimizer(problem, rule, target)
    actual = expected_gradient_penalty(problem, rule, W1, W2, batch_size)
    assert actual == pytest.approx(target["min_penalty"] / batch_size, rel=1e-12)
    for scale in (1.0, 1.7):
        # Reciprocal rescaling preserves exact fitting but changes the penalty.
        first, second = scale * W1, W2 / scale
        constrained = gradient_penalty(problem, rule, first, second)
        actual = expected_gradient_penalty(problem, rule, first, second, batch_size)
        assert actual == pytest.approx(constrained / batch_size, rel=1e-12)


def test_evaluation_entropy_scales_with_eta_and_preserves_rng(dense_task):
    problem, rule, W1, W2 = dense_task
    target = compute_target(problem, rule)
    before = torch.random.get_rng_state().clone()
    first = evaluate(W1, W2, problem, rule, target, batch_size=7, eta=0.001)
    second = evaluate(W1, W2, problem, rule, target, batch_size=7, eta=0.002)
    penalty = expected_gradient_penalty(problem, rule, W1, W2, 7)
    assert first["entropy"] == pytest.approx(0.001 * penalty, rel=1e-12)
    assert second["entropy"] == pytest.approx(2 * first["entropy"], rel=1e-12)
    for metric in ("fit_excess", "k_relative_error", "shape_error"):
        assert first[metric] == second[metric]
    assert torch.equal(torch.random.get_rng_state(), before)


def test_zero_weights_have_zero_penalty_despite_positive_prediction_error(dense_task):
    problem, rule, W1, W2 = dense_task
    W1, W2 = torch.zeros_like(W1), torch.zeros_like(W2)
    target = compute_target(problem, rule)
    actual = expected_gradient_penalty(problem, rule, W1, W2, 16)
    metrics = evaluate(W1, W2, problem, rule, target, batch_size=16, eta=0.001)
    assert actual == 0
    assert metrics["entropy"] == 0
    assert metrics["fit_excess"] > 0
