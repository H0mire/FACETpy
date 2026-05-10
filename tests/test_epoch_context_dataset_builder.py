"""Tests for the generic epoch-context dataset builder."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_builder_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "build_epoch_context_dataset.py"
    spec = importlib.util.spec_from_file_location("build_epoch_context_dataset", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_parse_clean_source_requires_path_and_regex():
    builder = _load_builder_module()

    parsed = builder._parse_clean_source("/tmp/clean.fif::spike|sharp")
    assert parsed.path == Path("/tmp/clean.fif")
    assert parsed.annotation_regex == "spike|sharp"

    with pytest.raises(argparse.ArgumentTypeError):
        builder._parse_clean_source("/tmp/clean.fif")


@pytest.mark.unit
def test_merge_dataset_chunks_preserves_clean_source_metadata():
    builder = _load_builder_module()

    first = {
        "noisy_context": np.zeros((2, 3, 1, 8), dtype=np.float32),
        "clean_context": np.zeros((2, 3, 1, 8), dtype=np.float32),
        "artifact_context": np.ones((2, 3, 1, 8), dtype=np.float32),
        "artifact_center": np.ones((2, 1, 8), dtype=np.float32),
        "artifact_epoch_lengths_samples": np.full((2, 3), 10, dtype=np.int64),
        "clean_source_indices": np.zeros(2, dtype=np.int64),
        "clean_source_paths": np.asarray(["a.fif", "a.fif"], dtype=object),
        "clean_annotation_regexes": np.asarray(["spike", "spike"], dtype=object),
    }
    second = {
        "noisy_context": np.zeros((1, 3, 1, 8), dtype=np.float32),
        "clean_context": np.zeros((1, 3, 1, 8), dtype=np.float32),
        "artifact_context": np.ones((1, 3, 1, 8), dtype=np.float32) * 2,
        "artifact_center": np.ones((1, 1, 8), dtype=np.float32) * 2,
        "artifact_epoch_lengths_samples": np.full((1, 3), 20, dtype=np.int64),
        "clean_source_indices": np.ones(1, dtype=np.int64),
        "clean_source_paths": np.asarray(["b.fif"], dtype=object),
        "clean_annotation_regexes": np.asarray(["blink"], dtype=object),
    }

    merged = builder._merge_dataset_chunks([first, second])
    stats = builder._aggregate_stats(
        merged,
        context_epochs=3,
        target_epoch_samples=8,
        n_clean_sources=2,
        n_artifact_sources=1,
    )

    assert merged["noisy_context"].shape == (3, 3, 1, 8)
    assert merged["clean_source_indices"].tolist() == [0, 0, 1]
    assert merged["clean_annotation_regexes"].tolist() == ["spike", "spike", "blink"]
    assert stats["n_examples"] == 3
    assert stats["n_clean_sources"] == 2
    assert stats["epoch_samples"] == 8
    assert stats["min_artifact_epoch_length_samples"] == 10
    assert stats["max_artifact_epoch_length_samples"] == 20


@pytest.mark.unit
def test_write_dataset_records_multiple_clean_sources(tmp_path):
    builder = _load_builder_module()
    dataset = {
        "noisy_context": np.zeros((1, 3, 1, 8), dtype=np.float32),
        "clean_context": np.zeros((1, 3, 1, 8), dtype=np.float32),
        "artifact_context": np.ones((1, 3, 1, 8), dtype=np.float32),
        "artifact_center": np.ones((1, 1, 8), dtype=np.float32),
        "artifact_epoch_lengths_samples": np.full((1, 3), 10, dtype=np.int64),
        "clean_source_indices": np.zeros(1, dtype=np.int64),
        "clean_source_paths": np.asarray(["a.fif"], dtype=object),
        "clean_annotation_regexes": np.asarray(["spike"], dtype=object),
    }
    stats = builder._aggregate_stats(
        dataset,
        context_epochs=3,
        target_epoch_samples=8,
        n_clean_sources=1,
        n_artifact_sources=1,
    )
    artifact_source = SimpleNamespace(
        source_id="artifact_a",
        source_path=Path("artifact_a.npz"),
        n_contexts=2,
        n_channels=1,
        sfreq=100.0,
    )

    dataset_path, metadata_path = builder._write_dataset(
        output_dir=tmp_path,
        clean_sources=[builder.CleanSourceSpec(Path("a.fif"), "spike")],
        artifact_sources=[artifact_source],
        dataset=dataset,
        target_sfreq=100.0,
        stats=stats,
    )

    assert dataset_path.name == "synthetic_spike_artifact_context_dataset.npz"
    assert metadata_path.name == "synthetic_spike_artifact_context_dataset_metadata.json"
    assert dataset_path.exists()
    metadata = metadata_path.read_text(encoding="utf-8")
    assert '"annotation_regex": "spike"' in metadata
    assert '"mode": "multi_clean_source_single_channel_context"' in metadata
