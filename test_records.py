"""Move existing results without changing data or breaking continuation paths."""

import json

import pytest
import torch

from configuration import expand_runs, load_config
from organize_records import organize
import run


def test_organize_preserves_runs_and_groups_exact_suite_names(tmp_path):
    original = {}
    suites = ("pilot", "pilot-extra", "pilot-sgd-0.extra")
    for suite in suites:
        stem = f"{suite}-sgd-0"
        original[f"{stem}.json"] = json.dumps({"suite_id": suite}).encode()
        original[f"{stem}.csv"] = b"step,fit_excess\n0,3\n"
        original[f"{stem}.pt"] = b"checkpoint contents"
        original[f"{suite}-curves.png"] = b"figure contents"
    for name, content in original.items():
        (tmp_path / name).write_bytes(content)

    assert len(organize(tmp_path)) == 12
    for suite in suites:
        stem = f"{suite}-sgd-0"
        for suffix in (".json", ".csv", ".pt"):
            name = stem + suffix
            assert (tmp_path / suite / name).read_bytes() == original[name]
        assert (tmp_path / suite / "curves.png").read_bytes() == original[f"{suite}-curves.png"]
    assert not any(path.is_file() for path in tmp_path.iterdir())
    assert organize(tmp_path) == []


def test_organize_does_not_overwrite_or_partially_move_runs(tmp_path):
    metadata = tmp_path / "pilot-sgd-0.json"
    metadata.write_text(json.dumps({"suite_id": "pilot"}))
    (tmp_path / "pilot").mkdir()
    existing = tmp_path / "pilot" / metadata.name
    existing.write_text("existing run")
    with pytest.raises(FileExistsError):
        organize(tmp_path)
    assert metadata.exists()
    assert existing.read_text() == "existing run"


def test_resume_uses_moved_checkpoint_location(tmp_path, monkeypatch):
    config = expand_runs(load_config("config_sgd.yaml"), "pilot", tmp_path)[0]
    config["runtime"]["records_dir"] = str(tmp_path)
    path = tmp_path / "pilot-sgd-0.pt"
    torch.save({"config": config, "step": 50, "problem": {}, "rule": {}, "target": {}}, path)
    path.with_suffix(".json").write_text(json.dumps({
        "suite_id": "pilot", "config": config, "resume_history": [],
    }))
    organize(tmp_path)
    captured = {}

    def capture(problem, rule, target, config, metadata, *, state):
        captured.update(config=config, metadata=metadata, state=state)

    monkeypatch.setattr(run, "require_cuda", lambda device: None)
    monkeypatch.setattr(run, "environment_info", lambda device: {"device": device})
    monkeypatch.setattr(run, "_execute", capture)
    run.resume(tmp_path / "pilot" / path.name, until_step=100)
    assert captured["config"]["runtime"]["records_dir"] == str(tmp_path / "pilot")
    assert captured["config"]["train"]["steps"] == 100
    assert captured["metadata"]["resume_history"][0]["from_step"] == 50
    assert captured["state"]["step"] == 50
