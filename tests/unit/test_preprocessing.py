"""Unit tests for inference/preprocessing.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from inference.preprocessing import (
    FALLBACK_IMAGENET_MEAN,
    FALLBACK_IMAGENET_STD,
    build_batch_tensor,
    preprocess_image,
    resolve_preprocessing_config,
    validate_and_decode_image,
)


def test_resolve_preprocessing_config_falls_back_to_imagenet_defaults(logger):
    config = resolve_preprocessing_config((1, 3, 224, 224), training_metadata={}, logger=logger)
    assert config.mean == FALLBACK_IMAGENET_MEAN
    assert config.std == FALLBACK_IMAGENET_STD
    assert "fallback" in config.mean_source
    assert (config.resize_height, config.resize_width, config.channels) == (224, 224, 3)


def test_resolve_preprocessing_config_uses_recorded_normalization(logger):
    metadata = {"normalization_mean": [0.1, 0.2, 0.3], "normalization_std": [0.4, 0.5, 0.6]}
    config = resolve_preprocessing_config((1, 3, 128, 96), training_metadata=metadata, logger=logger)
    assert config.mean == [0.1, 0.2, 0.3]
    assert config.std == [0.4, 0.5, 0.6]
    assert "normalization_mean" in config.mean_source
    assert (config.resize_height, config.resize_width) == (128, 96)


def test_resolve_preprocessing_config_searches_one_level_of_nesting(logger):
    metadata = {"stats": {"mean": [0.1, 0.1, 0.1], "std": [0.2, 0.2, 0.2]}}
    config = resolve_preprocessing_config((1, 3, 64, 64), training_metadata=metadata, logger=logger)
    assert config.mean == [0.1, 0.1, 0.1]
    assert config.std == [0.2, 0.2, 0.2]


def test_validate_and_decode_image_missing_file():
    img, error = validate_and_decode_image("/nonexistent/path/image.png", expected_channels=3)
    assert img is None
    assert error is not None
    assert error.error_type == "missing_file"


def test_validate_and_decode_image_unsupported_extension(tmp_path: Path):
    bad_file = tmp_path / "not_an_image.txt"
    bad_file.write_text("hello")
    img, error = validate_and_decode_image(str(bad_file), expected_channels=3)
    assert img is None
    assert error.error_type == "unsupported_extension"


def test_validate_and_decode_image_zero_byte(tmp_path: Path):
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")
    img, error = validate_and_decode_image(str(empty_file), expected_channels=3)
    assert img is None
    assert error.error_type == "zero_byte_image"


def test_validate_and_decode_image_corrupted(tmp_path: Path):
    corrupted = tmp_path / "corrupted.png"
    corrupted.write_bytes(b"not actually a png file")
    img, error = validate_and_decode_image(str(corrupted), expected_channels=3)
    assert img is None
    assert error.error_type == "corrupted_image"


def test_validate_and_decode_image_success_converts_to_rgb(tmp_path: Path):
    path = tmp_path / "valid.png"
    array = np.zeros((32, 32), dtype=np.uint8)  # grayscale
    Image.fromarray(array, mode="L").save(path)
    img, error = validate_and_decode_image(str(path), expected_channels=3)
    assert error is None
    assert img is not None
    assert img.mode == "RGB"


def test_preprocess_image_produces_expected_shape(sample_image_path, logger):
    config = resolve_preprocessing_config((1, 3, 224, 224), training_metadata={}, logger=logger)
    img, decode_error = validate_and_decode_image(sample_image_path, expected_channels=3)
    assert decode_error is None
    tensor, error = preprocess_image(img, config)
    assert error is None
    assert tuple(tensor.shape) == (3, 224, 224)


def test_preprocess_image_is_deterministic(sample_image_path, logger):
    config = resolve_preprocessing_config((1, 3, 224, 224), training_metadata={}, logger=logger)
    img, _ = validate_and_decode_image(sample_image_path, expected_channels=3)
    tensor_a, _ = preprocess_image(img, config)
    img2, _ = validate_and_decode_image(sample_image_path, expected_channels=3)
    tensor_b, _ = preprocess_image(img2, config)
    assert torch_equal(tensor_a, tensor_b)


def torch_equal(a, b) -> bool:
    return bool((a == b).all())


def test_build_batch_tensor_stacks_correctly(sample_image_path, logger):
    import torch

    config = resolve_preprocessing_config((1, 3, 224, 224), training_metadata={}, logger=logger)
    img, _ = validate_and_decode_image(sample_image_path, expected_channels=3)
    tensor, _ = preprocess_image(img, config)
    batch = build_batch_tensor([tensor, tensor], device=torch.device("cpu"))
    assert tuple(batch.shape) == (2, 3, 224, 224)
