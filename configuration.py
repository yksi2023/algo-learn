"""Read experiment settings and expand paired seeds into individual runs."""

from copy import deepcopy
import math
from pathlib import Path
import re

import yaml


DEFAULTS = {
    "train": {
        "eta": 0.001, "batch_size": 16, "steps": 100000, "init_scale": 0.1,
        "logging": {"eval_every": 100, "checkpoint_every": 1000, "tail_fraction": 0.2},
    },
    "seeds": {"problem": 11, "init": [101], "data": [201]},
    "numerics": {"rtol": 1e-10, "atol": 1e-12, "gap_rtol": 1e-8},
    "runtime": {"device": "cuda:0", "dtype": "float64", "threads": 1},
}


def validate_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Run and rule names must contain only letters, digits, dots, underscores or hyphens")


def _merge(defaults: dict, supplied: dict) -> dict:
    result = deepcopy(defaults)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def validate_config(config: dict) -> None:
    """Check user-supplied training controls once, when loading the YAML file."""
    train, seeds, runtime = config["train"], config["seeds"], config["runtime"]
    for key in ("eta", "init_scale"):
        if not math.isfinite(train[key]) or train[key] <= 0:
            raise ValueError(f"train.{key} must be finite and positive")
    counts = {key: train[key] for key in ("batch_size", "steps")}
    counts.update({key: train["logging"][key] for key in ("eval_every", "checkpoint_every")})
    counts["threads"] = runtime["threads"]
    for key, value in counts.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 < train["logging"]["tail_fraction"] <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")
    if not seeds["init"] or len(seeds["init"]) != len(seeds["data"]):
        raise ValueError("Initialization and data seeds must be nonempty paired lists")
    for key in ("init", "data"):
        if len(set(seeds[key])) != len(seeds[key]):
            raise ValueError(f"seeds.{key} must be distinct across repeats")
    for seed in [seeds["problem"], *seeds["init"], *seeds["data"]]:
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Seeds must be integers in [0, 2**63)")
    if runtime["dtype"] != "float64":
        raise ValueError("This implementation uses float64")
    for key, value in config["numerics"].items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"numerics.{key} must be finite and nonnegative")
    if not config["rules"]:
        raise ValueError("At least one learning rule is required")
    for name in config["rules"]:
        validate_name(name)


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = _merge(DEFAULTS, yaml.safe_load(path.read_text(encoding="utf-8")))
    validate_config(config)
    specs = [config["problem"][key] for key in ("V", "Sigma_x", "Sigma_epsilon")]
    specs += [rule[key] for rule in config["rules"].values() for key in ("Lambda1", "Lambda2")]
    for spec in specs:
        if spec.get("kind") == "file":
            spec["path"] = str((path.parent / spec["path"]).resolve())
    return config


def expand_runs(
    config: dict, suite_id: str, records_dir: str | Path, *,
    rule_name: str | None = None, repeat: int | None = None,
    device: str | None = None, steps: int | None = None,
) -> list[dict]:
    """Each repeat uses the same initialization/data seed pair across rules."""
    validate_name(suite_id)
    names = [rule_name] if rule_name is not None else list(config["rules"])
    count = len(config["seeds"]["init"])
    if rule_name is not None and rule_name not in config["rules"]:
        raise ValueError(f"Unknown rule: {rule_name}")
    if repeat is not None and not 0 <= repeat < count:
        raise ValueError(f"repeat must be in 0..{count - 1}")
    if steps is not None and steps <= 0:
        raise ValueError("--steps must be positive")
    runs = []
    for name in names:
        for index in ([repeat] if repeat is not None else range(count)):
            single = deepcopy(config)
            single.update(rule_name=name, repeat=index)
            for key in ("init", "data"):
                single["seeds"][key] = config["seeds"][key][index]
            single["runtime"].update(
                run_id=f"{suite_id}-{name}-{index}", suite_id=suite_id,
                records_dir=str((Path(records_dir) / suite_id).resolve()),
            )
            if device is not None:
                single["runtime"]["device"] = device
            if steps is not None:
                single["train"]["steps"] = steps
            runs.append(single)
    return runs
