"""Configuration defaults, paired runs and file paths."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from configuration import DEFAULTS, expand_runs, load_config, validate_config


def minimal_config():
    return {
        "problem": {
            "dimensions": {"d": 5, "m": 3, "h": 4, "rank": 2},
            "V": {"kind": "svd", "values": [3.0, 1.0]},
            "Sigma_x": {"kind": "identity"}, "Sigma_epsilon": {"kind": "identity"},
        },
        "rules": {
            "sgd": {"Lambda1": {"kind": "identity"}, "Lambda2": {"kind": "identity"}},
            "second_layer": {
                "Lambda1": {"kind": "identity"},
                "Lambda2": {"kind": "scaled_identity", "scale": 4.0},
            },
        },
        "seeds": {"problem": 11, "init": [101, 102], "data": [201, 202]},
    }


def write_config(tmp_path, raw):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.fixture
def config(tmp_path):
    return load_config(write_config(tmp_path, minimal_config()))


def test_defaults_merge_without_sharing_mutable_state(tmp_path):
    before = deepcopy(DEFAULTS)
    raw = minimal_config()
    raw["train"] = {"eta": 0.002, "logging": {"eval_every": 7}}
    path = write_config(tmp_path, raw)
    first = load_config(path)
    assert first["train"]["eta"] == 0.002
    assert first["train"]["logging"]["eval_every"] == 7
    assert first["train"]["logging"]["checkpoint_every"] == 1000
    assert first["runtime"]["device"] == "cuda:0"
    first["train"]["logging"]["tail_fraction"] = 0.5
    assert DEFAULTS == before
    assert load_config(path)["train"]["logging"]["tail_fraction"] == 0.2


def test_expansion_pairs_seeds_and_copies_config(config, tmp_path):
    before = deepcopy(config)
    runs = expand_runs(config, "pilot", tmp_path)
    assert len(runs) == 4
    for item in runs:
        index = item["repeat"]
        assert item["seeds"] == {"problem": 11, "init": 101 + index, "data": 201 + index}
        assert item["runtime"]["run_id"] == f"pilot-{item['rule_name']}-{index}"
    runs[0]["problem"]["V"]["values"][0] = 99.0
    assert runs[1]["problem"]["V"]["values"][0] == 3.0
    assert config == before


def test_selection_and_overrides(config, tmp_path):
    runs = expand_runs(config, "pilot", tmp_path, rule_name="second_layer", repeat=1,
                       device="cuda:2", steps=25)
    assert len(runs) == 1
    assert runs[0]["seeds"]["init"] == 102
    assert runs[0]["runtime"]["device"] == "cuda:2"
    assert runs[0]["train"]["steps"] == 25
    assert config["runtime"]["device"] == "cuda:0"


def test_file_paths_are_relative_to_config(tmp_path):
    folder = tmp_path / "configurations"
    folder.mkdir()
    raw = minimal_config()
    raw["problem"]["V"] = {"kind": "file", "path": "task.json", "key": "V"}
    raw["problem"]["Sigma_x"] = {"kind": "file", "path": "../covariance.csv"}
    config = load_config(write_config(folder, raw))
    assert config["problem"]["V"]["path"] == str(folder / "task.json")
    assert config["problem"]["Sigma_x"]["path"] == str(tmp_path / "covariance.csv")


@pytest.mark.parametrize("filename", ["config.yaml", "config_anisotropic.yaml"])
def test_supplied_configs(filename, tmp_path):
    config = load_config(Path(__file__).with_name(filename))
    assert len(expand_runs(config, "sample", tmp_path)) == 9
    assert config["problem"]["dimensions"] == {"d": 32, "m": 32, "h": 64, "rank": 32}
    assert config["runtime"]["device"] == "cuda:0"


@pytest.mark.parametrize("key,value", [("eta", 0), ("batch_size", 0), ("steps", -1)])
def test_invalid_training_controls(config, key, value):
    config["train"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)


def test_seed_lists_must_pair(config):
    config["seeds"]["data"] = [201]
    with pytest.raises(ValueError, match="paired"):
        validate_config(config)


@pytest.mark.parametrize("options", [{"rule_name": "missing"}, {"repeat": -1}, {"repeat": 2}, {"steps": 0}])
def test_invalid_selection(config, tmp_path, options):
    with pytest.raises(ValueError):
        expand_runs(config, "pilot", tmp_path, **options)
