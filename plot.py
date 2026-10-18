"""Plot saved representation metrics without loading checkpoints or importing training code.

Tail statistics use recorded points whose *step* is in the requested final
training interval. They are descriptive statistics along one trajectory.
Condition summaries give equal weight to each run's tail mean. Standard
deviations are sample standard deviations, with zero for a single observation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METRICS = ("fit_excess", "entropy", "k_relative_error", "shape_error")
CSV_COLUMNS = ("step", *METRICS, "elapsed_seconds")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    suite_id: str
    rule_name: str
    repeat: int
    steps: int
    tail_fraction: float
    rows: tuple[dict[str, float | int], ...]


def _read_rows(path: Path) -> tuple[dict[str, float | int], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            # Runs recorded before Entropy was added have no value for it.
            row.setdefault("entropy", "nan")
            rows.append({"step": int(row["step"]), **{name: float(row[name]) for name in CSV_COLUMNS[1:]}})
        return tuple(rows)


def read_runs(records: Path | str, suite_id: str) -> tuple[list[RunRecord], list[str]]:
    """Read completed runs in one suite and report unfinished runs."""
    records = Path(records)
    runs: list[RunRecord] = []
    skipped: list[str] = []
    for path in sorted(records.glob("*.json")):
        if not path.stem.startswith(f"{suite_id}-"):
            continue
        with path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata["suite_id"] != suite_id:
            continue
        if metadata["status"] != "completed":
            skipped.append(f"Skipped {path.stem}: status={metadata['status']!r}")
            continue
        training = metadata["config"]["train"]
        runs.append(
            RunRecord(
                run_id=metadata["run_id"],
                suite_id=suite_id,
                rule_name=metadata["rule_name"],
                repeat=metadata["repeat"],
                steps=training["steps"],
                tail_fraction=training["logging"]["tail_fraction"],
                rows=_read_rows(path.with_suffix(".csv")),
            )
        )
    return runs, skipped


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    # Undefined or unrecorded metrics remain undefined in summaries.
    if math.isnan(mean):
        return mean, mean
    return mean, statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_run(run: RunRecord) -> dict[str, Any]:
    """Summarize the final interval of training steps, including its boundary.

    Logged points within that interval have equal weight. This is not a
    continuous-time average or an estimate of independent replicate variation.
    """
    start = run.steps * (1 - run.tail_fraction)
    # Include a mathematically integral boundary despite floating-point rounding.
    threshold = math.nextafter(start, -math.inf)
    rows = [row for row in run.rows if row["step"] >= threshold]
    result: dict[str, Any] = {
        "run_id": run.run_id,
        "rule_name": run.rule_name,
        "repeat": run.repeat,
        "steps": run.steps,
        "tail_fraction": run.tail_fraction,
        "tail_start_step": start,
        "tail_record_count": len(rows),
    }
    for name in METRICS:
        values = [float(row[name]) for row in rows]
        result[f"{name}_mean"], result[f"{name}_std"] = _mean_std(values)
    return result


def summarize_conditions(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give each run equal weight, regardless of its number of logged points."""
    result: list[dict[str, Any]] = []
    for rule in sorted({item["rule_name"] for item in summaries}):
        group = [item for item in summaries if item["rule_name"] == rule]
        row: dict[str, Any] = {"rule_name": rule, "run_count": len(group)}
        for name in METRICS:
            values = [item[f"{name}_mean"] for item in group]
            row[f"{name}_mean_of_run_means"], row[f"{name}_std_of_run_means"] = _mean_std(values)
        result.append(row)
    return result


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_runs(runs: list[RunRecord], records: Path, suite_id: str) -> list[Path]:
    """Show each metric's definition and compare runs with the same update rule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 10), sharex=True)
    axes = axes.ravel()
    titles = (
        r"$\mathbb{E}_x\|W_2W_1x-Vx\|_2^2$",
        "Entropy\n" + r"$\eta\sum_{i=1}^2\mathbb{E}\operatorname{tr}(\bar{G}_i^{\mathsf{T}}\Lambda_i\bar{G}_i)$",
        r"$\frac{\|K-K_{\mathrm{theory}}\|_F}{\|K_{\mathrm{theory}}\|_F}$",
        r"$\left\|\frac{K}{\operatorname{tr}(K)}-\frac{K_{\mathrm{theory}}}{\operatorname{tr}(K_{\mathrm{theory}})}\right\|_F$",
    )
    rule_labels = {
        "sgd": "SGD",
        "first_layer": r"$\Lambda_1$",
        "second_layer": r"$\Lambda_2$",
        "random_diagonal": r"$(\Lambda_1,\Lambda_2)$: diagonal",
        "spd_1": r"$(\Lambda_1,\Lambda_2)$: SPD 1",
        "spd_2": r"$(\Lambda_1,\Lambda_2)$: SPD 2",
    }
    rules = sorted({run.rule_name for run in runs}, key=lambda rule: (rule != "sgd", rule))
    colors = plt.get_cmap("tab10")
    for index, rule in enumerate(rules):
        group = sorted((run for run in runs if run.rule_name == rule), key=lambda run: run.repeat)
        for repeat_index, run in enumerate(group):
            steps = [row["step"] for row in run.rows]
            for axis, name in zip(axes, METRICS, strict=True):
                axis.plot(
                    steps,
                    [row[name] for row in run.rows],
                    color=colors(index % 10),
                    alpha=0.75,
                    linewidth=1.1,
                    marker="o",
                    markersize=3,
                    markevery=max(1, len(steps) // 20),
                    label=rule_labels.get(rule, rule) if repeat_index == 0 else None,
                )
    for axis, title, name in zip(axes, titles, METRICS, strict=True):
        axis.set_title(title, fontsize=15, pad=16)
        axis.set_xlabel("Update steps")
        axis.tick_params(axis="x", labelbottom=True)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        # Undefined metrics remain gaps; they do not determine the axis scale.
        values = [row[name] for run in runs for row in run.rows if math.isfinite(row[name])]
        if not values and name == "entropy":
            axis.text(0.5, 0.5, "Not recorded", transform=axis.transAxes, ha="center", va="center")
        lower, upper = min(values, default=0), max(values, default=0)
        if upper > 100 * max(lower, 1e-12):
            axis.set_yscale("symlog", linthresh=1e-12)
        else:
            # Keep small changes in proportion to the magnitude of the error.
            axis.set_ylim(0, 1.1 * upper if upper > 0 else 1)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        axis.grid(True, which="both", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False)
    figure.suptitle(suite_id, fontsize=17, y=0.98)
    figure.text(
        0.5, 0.025,
        r"$K=W_1^{\mathsf{T}}W_1$; $K_{\mathrm{theory}}$: theoretical prediction for the corresponding update rule."
        "\n" + r"$\bar{G}_i$: minibatch-average gradient of the sample loss with respect to $W_i$.",
        ha="center", va="bottom",
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.91), w_pad=2.5, h_pad=3)
    paths = [records / f"{suite_id}-curves.pdf", records / f"{suite_id}-curves.png"]
    try:
        for path in paths:
            figure.savefig(path, dpi=180)
    finally:
        plt.close(figure)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Exact suite_id from run metadata")
    parser.add_argument("--records", type=Path, default=Path("records"))
    args = parser.parse_args(argv)
    runs, skipped = read_runs(args.records, args.run_id)
    for message in skipped:
        print(message, file=sys.stderr)
    if not runs:
        parser.error(f"No completed runs have suite_id={args.run_id!r}")
    summaries = [summarize_run(run) for run in runs]
    paths = plot_runs(runs, args.records, args.run_id)
    summary_path = args.records / f"{args.run_id}-summary.csv"
    condition_path = args.records / f"{args.run_id}-conditions.csv"
    _write_summary(summary_path, summaries)
    _write_summary(condition_path, summarize_conditions(summaries))
    for path in [*paths, summary_path, condition_path]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
