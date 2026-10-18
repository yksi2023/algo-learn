"""Run paired experiments or continue a saved training trajectory."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import torch

from configuration import expand_runs, load_config
from problem import make_problem, make_rule
from theory import compute_target
from train import train


ROOT = Path(__file__).resolve().parent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, metadata: dict) -> None:
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def environment_info(device: str) -> dict:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("torch", "matplotlib", "PyYAML")},
        "device": device, "threads": torch.get_num_threads(), "torch_cuda_build": torch.version.cuda,
    }


def require_cuda(name: str) -> torch.device:
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("Training requires a CUDA GPU; CPU is not supported. Use --device cuda:0.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the project's GPU dependencies and run on a GPU server.")
    torch.cuda.get_device_properties(device)
    return device


def _execute(problem: dict, rule: dict, target: dict, config: dict, metadata: dict,
             *, state: dict | None = None) -> None:
    """Keep the metadata status in sync with the training process."""
    runtime = config["runtime"]
    path = Path(runtime["records_dir"]) / f"{runtime['run_id']}.json"
    metadata.update(config=config, status="running", updated_at=_now())
    metadata.pop("error", None)
    _write_json(path, metadata)
    start = 0 if state is None else state["step"]
    print(f"{runtime['run_id']}: steps {start} -> {config['train']['steps']} on {runtime['device']}", flush=True)
    try:
        final = train(problem, rule, target, config, state=state)
    except (Exception, KeyboardInterrupt) as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        updated_at=_now(), error=f"{type(exc).__name__}: {exc}")
        _write_json(path, metadata)
        raise
    metadata.update(status="completed", updated_at=_now(), last_step=final["step"])
    _write_json(path, metadata)
    print(f"{runtime['run_id']}: completed", flush=True)


def run_new(configs: list[dict]) -> None:
    require_cuda(configs[0]["runtime"]["device"])
    records = Path(configs[0]["runtime"]["records_dir"])
    records.mkdir(parents=True, exist_ok=True)
    for config in configs:
        run_id = config["runtime"]["run_id"]
        if any((records / f"{run_id}{suffix}").exists() for suffix in (".json", ".csv", ".pt")):
            raise FileExistsError(f"{run_id} already exists; choose a new ID or use --resume")

    torch.set_num_threads(configs[0]["runtime"]["threads"])
    problem = make_problem(configs[0])
    rules, targets = {}, {}
    for config in configs:
        name = config["rule_name"]
        if name not in rules:
            rules[name] = make_rule(config["rules"][name], problem)
            targets[name] = compute_target(problem, rules[name])
    environment = environment_info(configs[0]["runtime"]["device"])
    for config in configs:
        runtime, name = config["runtime"], config["rule_name"]
        metadata = {
            "format_version": 1, "run_id": runtime["run_id"], "suite_id": runtime["suite_id"],
            "rule_name": name, "repeat": config["repeat"], "environment": environment,
            "created_at": _now(), "resume_history": [],
        }
        _execute(problem, rules[name], targets[name], config, metadata)


def resume(path: Path, *, until_step: int | None = None, device: str | None = None) -> None:
    path = path.resolve()
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = deepcopy(state["config"])
    if device is not None:
        config["runtime"]["device"] = device
    require_cuda(config["runtime"]["device"])
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    # Metadata retains the intended endpoint if a prior continuation was interrupted.
    config["train"]["steps"] = metadata["config"]["train"]["steps"] if until_step is None else until_step
    if config["train"]["steps"] < state["step"]:
        raise ValueError("--until-step cannot precede the saved checkpoint step")
    config["runtime"].update(records_dir=str(path.parent), run_id=path.stem)
    torch.set_num_threads(config["runtime"]["threads"])
    metadata["resume_history"].append({
        "at": _now(), "from_step": state["step"], "until_step": config["train"]["steps"],
        "environment": environment_info(config["runtime"]["device"]),
    })
    _execute(state["problem"], state["rule"], state["target"], config, metadata, state=state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="YAML configuration (default: config.yaml beside this script)")
    source.add_argument("--resume", type=Path, help="Continue a saved .pt checkpoint and matching .json")
    parser.add_argument("--run-id", help="Suite ID; actual IDs append the rule and zero-based repeat")
    parser.add_argument("--records", type=Path, help="Result directory (default: ./records)")
    parser.add_argument("--rule", help="Run one configured rule")
    parser.add_argument("--repeat", type=int, help="Run one zero-based paired seed index")
    parser.add_argument("--device", help="CUDA GPU: cuda or cuda:N (default: configured cuda:0)")
    parser.add_argument("--steps", type=int, help="Total steps for a new run")
    parser.add_argument("--until-step", type=int, help="New total step count when resuming")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume:
        if any(value is not None for value in (args.run_id, args.records, args.rule, args.repeat, args.steps)):
            parser.error("--resume accepts only --until-step and --device overrides")
        resume(args.resume, until_step=args.until_step, device=args.device)
    else:
        if not args.run_id or args.until_step is not None:
            parser.error("New experiments require --run-id; --until-step is only for --resume")
        config = load_config(args.config or ROOT / "config.yaml")
        configs = expand_runs(config, args.run_id, args.records or Path("records"),
                              rule_name=args.rule, repeat=args.repeat, device=args.device, steps=args.steps)
        run_new(configs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
