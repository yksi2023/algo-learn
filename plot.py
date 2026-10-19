"""Plot seed means and sample standard deviations from saved representation metrics.

Curves use the steps recorded in every run of a condition, without interpolation.
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
    config: dict[str, Any]


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
    records = Path(records) / suite_id
    runs: list[RunRecord] = []
    skipped: list[str] = []
    for path in sorted(records.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
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
                config=metadata["config"],
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


def summarize_curves(runs: list[RunRecord]) -> list[dict[str, Any]]:
    """Average seeds at common recorded steps; do not drop missing metric values."""
    result: list[dict[str, Any]] = []
    for rule in sorted({run.rule_name for run in runs}):
        group = [run for run in runs if run.rule_name == rule]
        histories = [{row["step"]: row for row in run.rows} for run in group]
        steps = sorted(set.intersection(*(set(history) for history in histories)))
        for step in steps:
            row: dict[str, Any] = {"rule_name": rule, "step": step, "run_count": len(group)}
            for name in METRICS:
                values = [float(history[step][name]) for history in histories]
                row[f"{name}_mean"], row[f"{name}_std"] = _mean_std(values)
            result.append(row)
    return result


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix_description(spec: dict[str, Any]) -> str:
    """Describe the saved construction, including how the spectrum was chosen."""
    kind = spec["kind"]
    if kind == "identity":
        return r"$I$"
    if kind == "scaled_identity":
        return "$" + f"{spec['scale']:g}I$"
    if kind == "active_linear":
        return (
            rf"diagonal: first $r$ entries {spec['min']:g} to {spec['max']:g}, "
            rf"remaining entries {spec['rest']:g}"
        )
    if kind in ("diagonal", "spd"):
        structure = "diagonal" if kind == "diagonal" else "dense symmetric positive definite"
        if kind == "diagonal" and spec.get("spectrum") == "uniform" and "values" not in spec:
            structure = "random diagonal"
        if "values" in spec:
            spectrum = f"specified eigenvalues in [{min(spec['values']):g}, {max(spec['values']):g}]"
        elif spec.get("spectrum", "linear") == "uniform":
            spectrum = f"eigenvalues sampled uniformly in [{spec['min']:g}, {spec['max']:g}]"
        else:
            spectrum = f"linearly spaced eigenvalues in [{spec['min']:g}, {spec['max']:g}]"
        return f"{structure}\n{spectrum}"
    if kind == "file":
        return f"matrix from {Path(spec['path']).name}"
    return "specified matrix"


def rule_label(rule: str, config: dict[str, Any], compared_rules: list[str]) -> str:
    settings = config["rules"][rule]
    first, second = settings["Lambda1"], settings["Lambda2"]
    if first["kind"] == second["kind"] == "identity":
        dims = config["problem"]["dimensions"]
        return "Stochastic gradient descent\n" + rf"$\Lambda_1=I_{{{dims['h']}}},\ \Lambda_2=I_{{{dims['m']}}}$"
    if first["kind"] == second["kind"] == "spd" and "random_diagonal" in compared_rules:
        diagonal = config["rules"]["random_diagonal"]
        if all(
            {key: value for key, value in settings[layer].items() if key != "kind"}
            == {key: value for key, value in diagonal[layer].items() if key != "kind"}
            for layer in ("Lambda1", "Lambda2")
        ):
            return r"$(\Lambda_1,\Lambda_2)$: dense symmetric positive definite" + "\nsame eigenvalues as random diagonal"
    descriptions = [matrix_description(first), matrix_description(second)]
    if descriptions[0] == descriptions[1]:
        return r"$(\Lambda_1,\Lambda_2)$: " + descriptions[0]
    return r"$\Lambda_1$: " + descriptions[0] + "\n" + r"$\Lambda_2$: " + descriptions[1]


def plot_curves(
    curves: list[dict[str, Any]], records: Path, suite_id: str, runs: list[RunRecord],
) -> list[Path]:
    """Show one mean curve per condition, with seed variation at selected steps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 13
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.6), sharex=True)
    axes = axes.ravel()
    titles = (
        r"$\mathbb{E}_x\|W_2W_1x-Vx\|_2^2$",
        "Entropy: " + r"$\eta\sum_{i=1}^2\mathbb{E}\operatorname{tr}(\bar{G}_i^{\mathsf{T}}\Lambda_i\bar{G}_i)$",
        r"$\frac{\|K-K_{\mathrm{theory}}\|_F}{\|K_{\mathrm{theory}}\|_F}$",
        r"$\left\|\frac{K}{\operatorname{tr}(K)}-\frac{K_{\mathrm{theory}}}{\operatorname{tr}(K_{\mathrm{theory}})}\right\|_F$",
    )
    rules = sorted({row["rule_name"] for row in curves}, key=lambda rule: (rule != "sgd", rule))
    configs = {run.rule_name: run.config for run in runs}
    colors = plt.get_cmap("tab10")
    for index, rule in enumerate(rules):
        group = [row for row in curves if row["rule_name"] == rule]
        steps = [row["step"] for row in group]
        count = group[0]["run_count"]
        for axis, name in zip(axes, METRICS, strict=True):
            axis.errorbar(
                steps,
                [row[f"{name}_mean"] for row in group],
                yerr=[row[f"{name}_std"] for row in group] if count > 1 else None,
                errorevery=max(1, math.ceil(len(steps) / 12)),
                color=colors(index % 10),
                linewidth=1.8,
                elinewidth=1.2,
                capsize=3,
                label=f"{rule_label(rule, configs[rule], rules)} ({count} seeds)",
            )
    for axis, title, name in zip(axes, titles, METRICS, strict=True):
        axis.set_title(title, fontsize=19, pad=18)
        axis.set_xlabel("Update steps")
        axis.tick_params(axis="x", labelbottom=True)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        # Undefined metrics remain gaps; they do not determine the axis scale.
        values = [
            row[f"{name}_mean"] + direction * row[f"{name}_std"]
            for row in curves for direction in (-1, 0, 1)
            if math.isfinite(row[f"{name}_mean"] + direction * row[f"{name}_std"])
        ]
        if not values and name == "entropy":
            axis.text(0.5, 0.5, "Not recorded", transform=axis.transAxes, ha="center", va="center")
        lower, upper = min(values, default=0), max(values, default=0)
        if upper > 100 * max(lower, 1e-12):
            axis.set_yscale("symlog", linthresh=1e-12)
        else:
            # Keep small changes in proportion to the magnitude of the error.
            axis.set_ylim(min(0, 1.1 * lower), 1.1 * upper if upper > 0 else 1)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        axis.grid(True, which="both", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    legend = figure.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
        ncol=2, frameon=False, fontsize=12, labelspacing=0.6, columnspacing=1.5,
    )
    heading = figure.suptitle(suite_id, fontsize=20, y=0.99)
    heading.set_in_layout(False)
    footer = figure.text(
        0.5, 0.012,
        "Mean ± standard deviation across seeds;  " + r"$K=W_1^{\mathsf{T}}W_1$",
        ha="center", va="bottom", fontsize=12,
    )
    # Reserve the space actually used by the legend and footer.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_bottom = legend.get_window_extent(renderer).transformed(figure.transFigure.inverted()).y0
    footer_top = footer.get_window_extent(renderer).transformed(figure.transFigure.inverted()).y1
    figure.tight_layout(
        rect=(0, footer_top + 0.01, 1, legend_bottom - 0.01),
        pad=0.8, w_pad=1.5, h_pad=0.6,
    )
    paths = [records / suite_id / "curves.pdf", records / suite_id / "curves.png"]
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
    curves = summarize_curves(runs)
    paths = plot_curves(curves, args.records, args.run_id, runs)
    suite_dir = args.records / args.run_id
    summary_path = suite_dir / "summary.csv"
    condition_path = suite_dir / "conditions.csv"
    curve_path = suite_dir / "curves.csv"
    _write_summary(summary_path, summaries)
    _write_summary(condition_path, summarize_conditions(summaries))
    _write_summary(curve_path, curves)
    for path in [*paths, summary_path, condition_path, curve_path]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
