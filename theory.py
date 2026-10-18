"""The proposition's Gram matrix and one of its complete minimizer pairs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


def compute_target(problem: Mapping[str, Any], rule: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Compute the unique Gram target for distinct weighted singular values.

    This pilot deliberately rejects repeated or numerically unresolved positive
    singular values, including cases where extra analysis could show uniqueness.
    It never substitutes a single arbitrary member for a set-valued target.
    """
    rank = problem["rank"]
    mu, eigenvectors = torch.linalg.eigh(rule["Lambda1"])
    A = problem["Sigma_epsilon_sqrt"] @ problem["V"] @ problem["Sigma_x_sqrt"]
    U, singular_values, Rh = torch.linalg.svd(A, full_matrices=False)
    threshold = max(problem["numerics"]["atol"], problem["numerics"]["rank_rtol"] * float(singular_values[0]))
    weighted_rank = int((singular_values > threshold).sum())
    if weighted_rank != rank:
        raise ValueError(f"Weighted V has numerical rank {weighted_rank}; expected {rank}")
    sigma, U, R = singular_values[:rank], U[:, :rank], Rh[:rank].T
    if rank > 1:
        threshold = problem["numerics"]["atol"] + problem["numerics"]["gap_rtol"] * float(sigma[0])
        if bool(((sigma[:-1] - sigma[1:]) <= threshold).any()):
            raise ValueError(
                "The weighted positive singular values are repeated or too close "
                f"(gap tolerance {threshold:.6g}). This pilot requires distinct "
                "weighted singular values to compare against a single Gram target."
            )
    Q = eigenvectors[:, :rank]
    trace_x = float(torch.trace(problem["Sigma_x"]))
    trace_rule_noise = float(torch.trace(rule["Lambda2"] @ problem["Sigma_epsilon"]))
    factors = sigma * mu[:rank].sqrt()
    K = math.sqrt(trace_x / trace_rule_noise) * (
        problem["Sigma_x_inv_sqrt"] @ (R * factors.unsqueeze(0)) @ R.T @ problem["Sigma_x_inv_sqrt"]
    )
    return {
        "K_theory": K, "sigma": sigma, "mu": mu, "U": U, "R": R, "Q": Q,
        "min_penalty": 8 * math.sqrt(trace_x * trace_rule_noise) * float(factors.sum()),
    }


def construct_minimizer(
    problem: Mapping[str, Any], rule: Mapping[str, torch.Tensor], target: Mapping[str, Any] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct W1[h,d], W2[m,h] from the proposition, for algebra checks."""
    if target is None:
        target = compute_target(problem, rule)
    rank = problem["rank"]
    sigma, mu, Q, U, R = (target[key] for key in ("sigma", "mu", "Q", "U", "R"))
    scale = (float(torch.trace(problem["Sigma_x"])) / float(torch.trace(rule["Lambda2"] @ problem["Sigma_epsilon"]))) ** 0.25
    W1 = scale * (Q * (mu[:rank].pow(0.25) * sigma.sqrt()).unsqueeze(0)) @ R.T @ problem["Sigma_x_inv_sqrt"]
    W2 = (1 / scale) * problem["Sigma_epsilon_inv_sqrt"] @ (U * (mu[:rank].pow(-0.25) * sigma.sqrt()).unsqueeze(0)) @ Q.T
    return W1, W2


def gradient_penalty(
    problem: Mapping[str, Any], rule: Mapping[str, torch.Tensor], W1: torch.Tensor, W2: torch.Tensor
) -> float:
    """Expected single-sample gradient penalty at an exact-fitting pair.

    The formula applies on W2 @ W1 = V. Reject off-constraint inputs so this
    helper cannot accidentally be interpreted as the penalty along training.
    """
    with torch.no_grad():
        def local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device=W1.device, dtype=W1.dtype)

        if not torch.allclose(W2 @ W1, local(problem["V"]), rtol=problem["numerics"]["rtol"], atol=problem["numerics"]["atol"]):
            raise ValueError("gradient_penalty requires W2 @ W1 = V within the recorded tolerances")
        Sigma_x, Sigma_epsilon = local(problem["Sigma_x"]), local(problem["Sigma_epsilon"])
        Lambda1, Lambda2 = local(rule["Lambda1"]), local(rule["Lambda2"])
        value = 4 * (
            torch.trace(Sigma_x) * torch.trace(Sigma_epsilon @ W2 @ Lambda1 @ W2.T)
            + torch.trace(Lambda2 @ Sigma_epsilon) * torch.trace(W1 @ Sigma_x @ W1.T)
        )
        return float(value)
