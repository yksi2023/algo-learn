"""Integration checks for training, saved records and continuation."""

import json

import pytest
import torch
import yaml

import run
from plot import read_runs


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "environment_info", lambda device: {"device": device})
    config = {
        "problem": {
            "dimensions": {"d": 3, "m": 2, "h": 4, "rank": 2},
            "V": {"kind": "svd", "values": [2.0, 1.0]},
            "Sigma_x": {"kind": "identity"}, "Sigma_epsilon": {"kind": "identity"},
        },
        "rules": {
            "sgd": {"Lambda1": {"kind": "identity"}, "Lambda2": {"kind": "identity"}},
            "second_layer": {"Lambda1": {"kind": "identity"},
                             "Lambda2": {"kind": "scaled_identity", "scale": 2.0}},
        },
        "train": {"steps": 3, "logging": {"eval_every": 1, "checkpoint_every": 2}},
        "seeds": {"problem": 11, "init": [101, 102], "data": [201, 202]},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _args(config_path, records, *extra):
    return ["--config", str(config_path), "--run-id", "pilot", "--records", str(records), *extra]


def test_run_and_resume(config_path, tmp_path):
    records = tmp_path / "records"
    assert run.main(_args(config_path, records, "--repeat", "0")) == 0
    # Both rules start from identical weights and consume identical random draws.
    runs, skipped = read_runs(records, "pilot")
    assert len(runs) == 2 and not skipped
    first = torch.load(records / "pilot-sgd-0.pt", weights_only=True)
    second = torch.load(records / "pilot-second_layer-0.pt", weights_only=True)
    for key in ("initial_W1", "initial_W2", "data_rng_state"):
        assert torch.equal(first[key], second[key])
    assert not torch.equal(first["W2"], second["W2"])
    metadata_before = (records / "pilot-sgd-0.json").read_bytes()
    with pytest.raises(ValueError, match="CPU"):
        run.main(["--resume", str(records / "pilot-sgd-0.pt"), "--device", "cpu"])
    assert (records / "pilot-sgd-0.json").read_bytes() == metadata_before
    assert run.main(["--resume", str(records / "pilot-sgd-0.pt"), "--until-step", "5"]) == 0
    meta = json.loads((records / "pilot-sgd-0.json").read_text())
    assert meta["status"] == "completed" and meta["last_step"] == 5
    assert meta["resume_history"][0]["from_step"] == 3
    assert sorted(item.steps for item in read_runs(records, "pilot")[0]) == [3, 5]
    before = (records / "pilot-sgd-0.pt").read_bytes()
    with pytest.raises(FileExistsError):
        run.main(_args(config_path, records, "--rule", "sgd", "--repeat", "0"))
    assert (records / "pilot-sgd-0.pt").read_bytes() == before


def test_failure_is_recorded(config_path, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise FloatingPointError("deliberate non-finite update")

    monkeypatch.setattr(run, "train", fail)
    records = tmp_path / "records"
    with pytest.raises(FloatingPointError):
        run.main(_args(config_path, records, "--rule", "sgd", "--repeat", "0"))
    meta = json.loads((records / "pilot-sgd-0.json").read_text())
    assert meta["status"] == "failed"
    assert read_runs(records, "pilot")[0] == []


def test_cpu_is_rejected_before_creating_records(config_path, tmp_path):
    records = tmp_path / "records"
    with pytest.raises(ValueError, match="CPU"):
        run.main(_args(config_path, records, "--device", "cpu"))
    assert not records.exists()


def test_unavailable_cuda_is_rejected_before_creating_records(config_path, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    records = tmp_path / "records"
    with pytest.raises(RuntimeError, match="CUDA"):
        run.main(_args(config_path, records))
    assert not records.exists()


def test_resume_arguments_are_unambiguous():
    with pytest.raises(SystemExit) as error:
        run.main(["--resume", "saved.pt", "--steps", "10"])
    assert error.value.code == 2
