"""Unit tests for inference/utils/ (hashing, io, timers, environment)."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest
import torch

from inference.utils.environment import get_environment_info, set_seed
from inference.utils.hashing import sha256_of_file
from inference.utils.io import load_json, save_json
from inference.utils.timers import Timer


def test_sha256_of_file_is_deterministic(tmp_path: Path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"hello world")
    assert sha256_of_file(path) == sha256_of_file(path)


def test_sha256_of_file_differs_for_different_content(tmp_path: Path):
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"content A")
    path_b.write_bytes(b"content B")
    assert sha256_of_file(path_a) != sha256_of_file(path_b)


def test_sha256_of_file_matches_known_digest(tmp_path: Path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    # SHA-256 of the empty string is a well-known constant.
    assert sha256_of_file(path) == hashlib.sha256(b"").hexdigest()


def test_save_and_load_json_round_trip(tmp_path: Path):
    path = tmp_path / "data.json"
    data = {"a": 1, "b": [1, 2, 3], "c": None}
    save_json(path, data)
    assert load_json(path) == data


def test_save_json_ensure_parent_creates_directory(tmp_path: Path):
    path = tmp_path / "nested" / "dir" / "data.json"
    save_json(path, {"x": 1}, ensure_parent=True)
    assert path.exists()
    assert load_json(path) == {"x": 1}


def test_save_json_without_ensure_parent_raises_if_missing(tmp_path: Path):
    path = tmp_path / "nonexistent_dir" / "data.json"
    with pytest.raises(FileNotFoundError):
        save_json(path, {"x": 1}, ensure_parent=False)


def test_timer_measures_positive_elapsed_time():
    with Timer() as t:
        time.sleep(0.01)
    assert t.elapsed_s > 0
    assert t.elapsed_ms == pytest.approx(t.elapsed_s * 1000.0)


def test_timer_raises_if_accessed_before_completion():
    t = Timer()
    with pytest.raises(RuntimeError):
        _ = t.elapsed_s


def test_set_seed_makes_torch_rng_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_get_environment_info_has_expected_keys():
    info = get_environment_info()
    required_keys = {
        "python_version", "torch_version", "torchvision_version", "device",
        "gpu_available", "cpu_cores", "ram_total_gb", "os", "random_seed",
        "onnx_available", "torchscript_available",
    }
    assert required_keys.issubset(info.keys())
    assert info["device"] in ("cpu", "cuda")


def test_get_environment_info_defaults_seed_to_configs_default():
    from configs.defaults import SEED

    info = get_environment_info()
    assert info["random_seed"] == SEED


def test_get_environment_info_respects_explicit_seed():
    info = get_environment_info(seed=777)
    assert info["random_seed"] == 777
