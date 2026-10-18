"""Checks for saved-record plotting; run with pytest on the target server."""

import csv
import json
import math
from pathlib import Path

import pytest

from plot import CSV_COLUMNS, METRICS, read_runs, summarize_conditions, summarize_run


def write_run(
    records: Path,
    *,
    suite: str = "pilot01",
    rule: str = "sgd",
    repeat: int = 0,
    status: str = "completed",
    steps: int = 100,
    tail_fraction: float = 0.2,
    observations: list[tuple[int, float]] | None = None,
) -> str:
    run_id = f"{suite}-{rule}-{repeat}"
    metadata = {
        "format_version": 1,
        "run_id": run_id,
        "suite_id": suite,
        "rule_name": rule,
        "repeat": repeat,
        "status": status,
        "config": {
            "train": {"steps": steps, "logging": {"tail_fraction": tail_fraction}},
        },
    }
    (records / f"{run_id}.json").write_text(json.dumps(metadata), encoding="utf-8")
    if status == "completed":
        observations = observations if observations is not None else [(0, 10.0), (steps, 1.0)]
        with (records / f"{run_id}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for step, value in observations:
                writer.writerow({"step": step, **dict.fromkeys(METRICS, value), "elapsed_seconds": step})
    return run_id


def test_exact_suite_matching_ignores_similar_prefix(tmp_path):
    expected = write_run(tmp_path)
    write_run(tmp_path, suite="pilot010")
    write_run(tmp_path, suite="pilot01-other")
    runs, skipped = read_runs(tmp_path, "pilot01")
    assert [run.run_id for run in runs] == [expected]
    assert skipped == []


def test_failed_and_running_runs_are_reported_without_reading_csv(tmp_path):
    write_run(tmp_path, repeat=0, status="failed")
    write_run(tmp_path, repeat=1, status="running")
    runs, skipped = read_runs(tmp_path, "pilot01")
    assert runs == []
    assert len(skipped) == 2
    assert "failed" in skipped[0]
    assert "running" in skipped[1]


def test_tail_uses_step_interval_including_boundary(tmp_path):
    # Three of seven records lie in the last 20% of steps.
    write_run(tmp_path, observations=[(0, 50), (10, 40), (20, 30), (30, 20), (80, 2), (90, 4), (100, 6)])
    runs, _ = read_runs(tmp_path, "pilot01")
    result = summarize_run(runs[0])
    assert result["tail_start_step"] == pytest.approx(80)
    assert result["tail_record_count"] == 3
    for name in METRICS:
        assert result[f"{name}_mean"] == pytest.approx(4)
        assert result[f"{name}_std"] == pytest.approx(2)


def test_resumed_run_uses_its_own_final_step(tmp_path):
    write_run(tmp_path, repeat=0, observations=[(0, 30), (80, 1), (100, 1)])
    write_run(tmp_path, repeat=1, steps=200, observations=[(0, 30), (100, 50), (160, 3), (200, 3)])
    runs, _ = read_runs(tmp_path, "pilot01")
    summaries = [summarize_run(run) for run in runs]
    assert [row["tail_start_step"] for row in summaries] == pytest.approx([80, 160])
    assert [row["fit_excess_mean"] for row in summaries] == pytest.approx([1, 3])


def test_condition_summary_weights_runs_equally(tmp_path):
    # Frequent logging in the second run must not give it more weight.
    write_run(tmp_path, repeat=0, observations=[(0, 50), (100, 1)])
    write_run(tmp_path, repeat=1, observations=[(0, 50), (80, 3), (85, 3), (90, 3), (95, 3), (100, 3)])
    runs, _ = read_runs(tmp_path, "pilot01")
    summaries = [summarize_run(run) for run in runs]
    assert [item["tail_record_count"] for item in summaries] == [1, 5]
    aggregate = summarize_conditions(summaries)[0]
    assert aggregate["run_count"] == 2
    for name in METRICS:
        assert aggregate[f"{name}_mean_of_run_means"] == pytest.approx(2)
        assert aggregate[f"{name}_std_of_run_means"] == pytest.approx(2**0.5)


def test_zero_error_and_single_tail_record_are_preserved(tmp_path):
    write_run(tmp_path, observations=[(0, 1), (100, 0)])
    runs, _ = read_runs(tmp_path, "pilot01")
    summary = summarize_run(runs[0])
    for name in METRICS:
        assert summary[f"{name}_mean"] == 0
        assert summary[f"{name}_std"] == 0


def test_undefined_shape_error_is_preserved_in_summaries(tmp_path):
    run_id = write_run(tmp_path)
    path = tmp_path / f"{run_id}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[-1]["shape_error"] = "nan"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    runs, _ = read_runs(tmp_path, "pilot01")
    summary = summarize_run(runs[0])
    assert summary["fit_excess_mean"] == 1
    assert math.isnan(summary["shape_error_mean"])
    assert math.isnan(summary["shape_error_std"])
    aggregate = summarize_conditions([summary])[0]
    assert math.isnan(aggregate["shape_error_mean_of_run_means"])


def test_old_records_have_unknown_entropy_without_changing_other_metrics(tmp_path):
    run_id = write_run(tmp_path)
    path = tmp_path / f"{run_id}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    columns = [name for name in CSV_COLUMNS if name != "entropy"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({name: row[name] for name in columns} for row in rows)
    runs, _ = read_runs(tmp_path, "pilot01")
    assert all(math.isnan(row["entropy"]) for row in runs[0].rows)
    summary = summarize_run(runs[0])
    assert summary["fit_excess_mean"] == 1
    assert summary["k_relative_error_mean"] == 1
    assert summary["shape_error_mean"] == 1
    assert math.isnan(summary["entropy_mean"])
    assert math.isnan(summary["entropy_std"])
    aggregate = summarize_conditions([summary])[0]
    assert math.isnan(aggregate["entropy_mean_of_run_means"])
    assert math.isnan(aggregate["entropy_std_of_run_means"])
