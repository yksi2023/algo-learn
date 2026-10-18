"""Generate fixed tasks and sample independent Gaussian row batches.

All task and rule matrices are CPU float64 tensors. Matrix file paths are
relative to ``base_dir`` (normally the configuration file's directory).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


DTYPE = torch.float64
DEFAULT_NUMERICS = {"rtol": 1e-10, "atol": 1e-12, "gap_rtol": 1e-8}


def _child_seed(seed: int, name: str) -> int:
    """Derive stable streams without Python hash() or global RNG state."""
    payload = f"algorithm-dependent-representation-learning:{seed}:{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _validate_spec(spec: Mapping[str, Any], name: str, *, is_v: bool = False) -> str:
    """Catch unsupported kinds and fields that would otherwise be ignored."""
    kind = spec.get("kind", "svd" if is_v else "identity")
    allowed_by_kind = {
        "identity": {"kind"},
        "scaled_identity": {"kind", "scale"},
        "diagonal": {"kind", "values", "spectrum", "min", "max", "seed"},
        "spd": {"kind", "values", "spectrum", "min", "max", "seed"},
        "svd": {"kind", "values", "spectrum", "min", "max", "seed"},
        "active_linear": {"kind", "min", "max", "rest"},
        "values": {"kind", "values"},
        "file": {"kind", "path", "key"},
    }
    permitted_kinds = {"svd", "diagonal", "values", "file"} if is_v else set(allowed_by_kind) - {"svd"}
    if kind not in permitted_kinds:
        raise ValueError(f"{name}: unsupported matrix kind {kind!r}")
    unknown = set(spec) - allowed_by_kind[kind]
    if unknown:
        raise ValueError(f"{name}: unknown or inapplicable fields for {kind}: {sorted(map(str, unknown))}")
    if kind in {"svd", "spd", "diagonal"} and "values" in spec:
        if any(key in spec for key in ("min", "max", "spectrum")):
            raise ValueError(f"{name}: explicit values cannot be combined with min, max, or spectrum")
    return kind


def _tensor(value: Any, name: str) -> torch.Tensor:
    if torch.as_tensor(value).is_complex():
        raise ValueError(f"{name} must be real")
    # Convert the original values directly, avoiding a float32 intermediate.
    return torch.as_tensor(value, device="cpu", dtype=DTYPE).detach().clone()


def _read_matrix(spec: Mapping[str, Any], base_dir: str | Path | None, name: str) -> torch.Tensor:
    path = Path(spec["path"])
    if not path.is_absolute():
        path = Path(base_dir or ".") / path
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    elif suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            data = [[float(item) for item in row] for row in csv.reader(stream) if row]
    elif suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"{name}: supported matrix files are .json, .csv, and .pt")
    if "key" in spec:
        data = data[spec["key"]]
    return _tensor(data, name)


def _spectrum(
    spec: Mapping[str, Any], size: int, name: str, *, seed: int, descending: bool,
) -> torch.Tensor:
    """Read positive values, or generate a linear or uniform random spectrum."""
    if "values" in spec:
        values = _tensor(spec["values"], f"{name}.values")
        if values.shape != (size,):
            raise ValueError(f"{name}.values must have length {size}")
        if bool((values <= 0).any()):
            raise ValueError(f"{name}.values must be strictly positive")
        return values
    spectrum = spec.get("spectrum", "linear")
    if spectrum not in {"linear", "uniform"}:
        raise ValueError(f"{name}: spectrum must be linear or uniform, or supply explicit values")
    low = float(spec.get("min", 1.0))
    high = float(spec.get("max", low))
    if not 0 < low <= high:
        raise ValueError(f"{name} requires 0 < min <= max")
    if spectrum == "uniform":
        rng = _generator(_child_seed(seed, "spectrum"))
        return low + (high - low) * torch.rand(size, generator=rng, dtype=DTYPE, device="cpu")
    start, end = (high, low) if descending else (low, high)
    return torch.linspace(start, end, size, dtype=DTYPE, device="cpu")


def _orthogonal(size: int, rng: torch.Generator) -> torch.Tensor:
    basis, triangular = torch.linalg.qr(torch.randn(size, size, generator=rng, dtype=DTYPE, device="cpu"))
    # Fix the QR sign convention while preserving the Gaussian orthogonal basis.
    signs = torch.where(torch.diagonal(triangular) < 0, -1.0, 1.0)
    return basis * signs.unsqueeze(0)


def _check_matrix(matrix: torch.Tensor, shape: tuple[int, int], name: str) -> None:
    if tuple(matrix.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(matrix.shape)}")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} contains a non-finite entry")


def validate_spd_matrix(
    matrix: torch.Tensor, size: int, numerics: Mapping[str, float], name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate SPD at the recorded tolerances; return ascending eigenpairs.

    Small positive eigenvalues below atol + rtol * spectral_radius are rejected
    as numerically unresolved. No eigenvalue is clipped or replaced.
    """
    _check_matrix(matrix, (size, size), name)
    if not torch.allclose(matrix, matrix.T, rtol=numerics["rtol"], atol=numerics["atol"]):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    threshold = numerics["atol"] + numerics["rtol"] * float(eigenvalues.abs().max())
    if float(eigenvalues[0]) <= threshold:
        raise ValueError(
            f"{name} must be numerically positive definite: smallest eigenvalue "
            f"{float(eigenvalues[0]):.6g} <= tolerance {threshold:.6g}"
        )
    return eigenvalues, eigenvectors


def _spd(
    spec: Mapping[str, Any], size: int, seed: int, base_dir: str | Path | None,
    numerics: Mapping[str, float], name: str, *, rank: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kind = _validate_spec(spec, name)
    if kind == "identity":
        matrix = torch.eye(size, dtype=DTYPE, device="cpu")
    elif kind == "scaled_identity":
        matrix = float(spec.get("scale", 1.0)) * torch.eye(size, dtype=DTYPE, device="cpu")
    elif kind in {"diagonal", "spd"}:
        matrix_seed = spec.get("seed", _child_seed(seed, name))
        values = _spectrum(spec, size, name, seed=matrix_seed, descending=False)
        if kind == "diagonal":
            matrix = torch.diag(values)
        else:
            basis = _orthogonal(size, _generator(matrix_seed))
            matrix = (basis * values.unsqueeze(0)) @ basis.T
    elif kind == "active_linear":
        if rank is None:
            raise ValueError(f"{name}: active_linear is only supported for Lambda1")
        low, high = float(spec.get("min", 1)), float(spec.get("max", 4))
        if not 0 < low <= high:
            raise ValueError(f"{name} requires 0 < min <= max")
        rest = float(spec.get("rest", high + 1))
        if size > rank and rest <= high:
            raise ValueError(f"{name}.rest must be strictly greater than max")
        values = torch.full((size,), rest, dtype=DTYPE, device="cpu")
        values[:rank] = torch.linspace(low, high, rank, dtype=DTYPE, device="cpu")
        matrix = torch.diag(values)
    elif kind == "values":
        matrix = _tensor(spec["values"], name)
    else:  # file
        matrix = _read_matrix(spec, base_dir, name)
    eigenvalues, eigenvectors = validate_spd_matrix(matrix, size, numerics, name)
    root = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T
    inverse_root = (eigenvectors * eigenvalues.rsqrt().unsqueeze(0)) @ eigenvectors.T
    return matrix, root, inverse_root


def numerical_rank(matrix: torch.Tensor, numerics: Mapping[str, float]) -> int:
    singular_values = torch.linalg.svdvals(matrix)
    threshold = max(numerics["atol"], numerics.get("rank_rtol", numerics["rtol"]) * float(singular_values[0]))
    return int((singular_values > threshold).sum())


def make_problem(
    config: Mapping[str, Any], seed: int | None = None, base_dir: str | Path | None = None
) -> dict[str, Any]:
    """Build one task from a full experiment configuration.

    Independent named streams ensure covariance changes leave V unchanged.
    Explicit ``values`` for generated V specify its positive singular values.
    """
    task = config["problem"]
    dimensions = task["dimensions"]
    d, m, h, rank = (dimensions[key] for key in ("d", "m", "h", "rank"))
    if not 1 <= rank <= min(d, m, h):
        raise ValueError("dimensions.rank must be in [1, min(d, m, h)]")
    seed = config.get("seeds", {}).get("problem", 0) if seed is None else seed
    numerics = DEFAULT_NUMERICS | config.get("numerics", {})
    numerics.setdefault("rank_rtol", numerics["rtol"])
    spec = task["V"]
    kind = _validate_spec(spec, "V", is_v=True)
    if kind in {"svd", "diagonal"}:
        matrix_seed = spec.get("seed", _child_seed(seed, "V"))
        values = _spectrum(spec, rank, "V", seed=matrix_seed, descending=True)
        if kind == "diagonal":
            V = torch.zeros(m, d, dtype=DTYPE, device="cpu")
            V[torch.arange(rank, device="cpu"), torch.arange(rank, device="cpu")] = values
        else:
            rng = _generator(matrix_seed)
            U = _orthogonal(m, rng)[:, :rank]
            R = _orthogonal(d, rng)[:, :rank]
            V = (U * values.unsqueeze(0)) @ R.T
    elif kind == "values":
        V = _tensor(spec["values"], "V")
    else:  # file
        V = _read_matrix(spec, base_dir, "V")
    _check_matrix(V, (m, d), "V")
    actual_rank = numerical_rank(V, numerics)
    if actual_rank != rank:
        raise ValueError(f"V has numerical rank {actual_rank}; configured rank is {rank}")
    result = {"d": d, "m": m, "h": h, "rank": rank, "V": V, "numerics": numerics, "seed": seed}
    for name, size in (("Sigma_x", d), ("Sigma_epsilon", m)):
        matrix, root, inverse_root = _spd(task.get(name, {"kind": "identity"}), size, seed, base_dir, numerics, name)
        result[name] = matrix
        result[f"{name}_sqrt"] = root
        result[f"{name}_inv_sqrt"] = inverse_root
    return result


def make_rule(
    rule_config: Mapping[str, Any], problem: Mapping[str, Any], seed: int | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    seed = problem["seed"] if seed is None else seed
    rule = {}
    for name, size in (("Lambda1", problem["h"]), ("Lambda2", problem["m"])):
        matrix, _, _ = _spd(
            rule_config.get(name, {"kind": "identity"}), size, seed, base_dir,
            problem["numerics"], name, rank=problem["rank"] if name == "Lambda1" else None,
        )
        rule[name] = matrix
    return rule


def sample_batch(problem: Mapping[str, Any], n: int, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw X[n,d] and Y[n,m] using only the supplied CPU generator."""
    X = torch.randn(n, problem["d"], dtype=DTYPE, generator=rng, device="cpu") @ problem["Sigma_x_sqrt"]
    epsilon = torch.randn(n, problem["m"], dtype=DTYPE, generator=rng, device="cpu") @ problem["Sigma_epsilon_sqrt"]
    return X, X @ problem["V"].T + epsilon
