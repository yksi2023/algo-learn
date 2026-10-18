"""Deterministic matrix diagnostics; evaluation consumes no random samples."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch


def expected_gradient_penalty(
    problem: Mapping[str, Any], rule: Mapping[str, torch.Tensor],
    W1: torch.Tensor, W2: torch.Tensor, batch_size: int,
) -> float:
    """Expected weighted squared batch gradient, including fitting residuals.

    Uses Gaussian input fourth moments and independent zero-mean label noise.
    The batch loss averages independent samples. At W2 @ W1 = V, this equals
    the single-sample penalty in theory.py divided by batch_size.
    """
    with torch.no_grad():
        def local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device=W1.device, dtype=W1.dtype)

        Sigma_x, Sigma_epsilon = local(problem["Sigma_x"]), local(problem["Sigma_epsilon"])
        Lambda1, Lambda2 = local(rule["Lambda1"]), local(rule["Lambda2"])
        residual = W2 @ W1 - local(problem["V"])
        covariance = residual @ Sigma_x @ residual.T + Sigma_epsilon
        weighted_output = W2 @ Lambda1 @ W2.T
        gram = W1.T @ W1
        cross = residual @ Sigma_x
        base = (
            torch.trace(Sigma_x) * torch.trace(weighted_output @ covariance)
            + torch.trace(gram @ Sigma_x) * torch.trace(Lambda2 @ covariance)
        )
        mean_term = (
            torch.trace(cross.T @ weighted_output @ cross)
            + torch.trace(gram @ cross.T @ Lambda2 @ cross)
        )
        return float(4 * (base / batch_size + (1 + 1 / batch_size) * mean_term))


def evaluate(
    W1: torch.Tensor, W2: torch.Tensor, problem: Mapping[str, Any],
    rule: Mapping[str, torch.Tensor], target: Mapping[str, Any], *, batch_size: int, eta: float,
) -> dict[str, float]:
    """Return fitting error, eta-scaled gradient penalty, and Gram errors.

    ``shape_error`` is NaN when W1 is exactly zero: trace normalization is
    undefined there. The other diagnostics remain defined. No inverse of
    the potentially rank-deficient Gram matrix is taken.
    """
    with torch.no_grad():
        def local(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(device=W1.device, dtype=W1.dtype)

        V = local(problem["V"])
        Sigma_x_sqrt = local(problem["Sigma_x_sqrt"])
        K_theory = local(target["K_theory"])
        target_norm = torch.linalg.vector_norm(K_theory)
        target_trace = torch.trace(K_theory)
        K = W1.T @ W1
        fit_excess = ((W2 @ W1 - V) @ Sigma_x_sqrt).square().sum()
        k_relative_error = torch.linalg.vector_norm(K - K_theory) / target_norm
        trace = float(torch.trace(K))
        shape_error = (
            float(torch.linalg.vector_norm(K / trace - K_theory / target_trace))
            if trace > 0 else math.nan
        )
        return {
            "fit_excess": float(fit_excess),
            "entropy": eta * expected_gradient_penalty(problem, rule, W1, W2, batch_size),
            "k_relative_error": float(k_relative_error),
            "shape_error": shape_error,
        }
