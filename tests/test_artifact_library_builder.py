"""Tests for multi-source artifact library helpers used by the example builder."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_builder_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "build_synthetic_spike_artifact_context_dataset.py"
    spec = importlib.util.spec_from_file_location("build_synthetic_spike_artifact_context_dataset", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_artifact_bundle(path: Path, *, scale: float) -> None:
    artifact = np.vstack(
        [
            np.linspace(0.0, scale, 50, dtype=np.float32),
            np.linspace(scale, 0.0, 50, dtype=np.float32),
        ]
    )
    np.savez_compressed(
        path,
        artifact=artifact,
        ch_names=np.asarray(["EEG001", "EEG002"], dtype=object),
        sfreq=np.asarray([100.0], dtype=np.float64),
        triggers=np.asarray([0, 10, 20, 30, 40, 50], dtype=np.int64),
        artifact_length=np.asarray([-1], dtype=np.int64),
        artifact_to_trigger_offset=np.asarray([0.0], dtype=np.float64),
    )


@pytest.mark.unit
def test_artifact_library_sources_preserve_source_ids(tmp_path):
    builder = _load_builder_module()
    first = tmp_path / "source_a.npz"
    second = tmp_path / "source_b.npz"
    _write_artifact_bundle(first, scale=1.0)
    _write_artifact_bundle(second, scale=2.0)

    paths = builder._discover_artifact_bundle_paths([tmp_path], fallback_bundle=first)
    sources = builder._build_artifact_library_sources(
        paths,
        target_epoch_samples=10,
        context_epochs=3,
        max_jitter_samples=0,
    )
    (
        contexts,
        lengths,
        phase_linear,
        phase_sincos,
        source_ids,
        source_paths,
        source_channel_indices,
    ) = builder._concatenate_artifact_sources(sources)

    assert [source.source_id for source in sources] == ["source_a", "source_b"]
    assert contexts.shape == (12, 3, 1, 10)
    assert lengths.shape == (12, 3)
    assert phase_linear.shape == (12, 3, 1, 10)
    assert phase_sincos.shape == (12, 3, 2, 10)
    assert set(source_ids.tolist()) == {"source_a", "source_b"}
    assert all(str(path).endswith(".npz") for path in source_paths)
    assert set(source_channel_indices.tolist()) == {0, 1}


@pytest.mark.unit
def test_single_channel_dataset_records_selected_artifact_sources(tmp_path):
    builder = _load_builder_module()
    first = tmp_path / "source_a.npz"
    second = tmp_path / "source_b.npz"
    _write_artifact_bundle(first, scale=1.0)
    _write_artifact_bundle(second, scale=2.0)
    sources = builder._build_artifact_library_sources(
        [first, second],
        target_epoch_samples=10,
        context_epochs=3,
        max_jitter_samples=0,
    )
    (
        contexts,
        lengths,
        phase_linear,
        phase_sincos,
        source_ids,
        source_paths,
        source_channel_indices,
    ) = builder._concatenate_artifact_sources(sources)
    clean_contexts = np.zeros((2, 3, 2, 10), dtype=np.float32)

    dataset, stats = builder._build_single_channel_context_dataset(
        clean_contexts=clean_contexts,
        clean_channel_names=["EEG001", "EEG002"],
        onset_offsets=np.asarray([2, 3], dtype=np.int64),
        spike_descriptions=np.asarray(["spike", "spike"], dtype=object),
        artifact_contexts=contexts,
        artifact_lengths=lengths,
        phase_linear=phase_linear,
        phase_sincos=phase_sincos,
        artifact_source_ids_by_context=source_ids,
        artifact_source_paths_by_context=source_paths,
        artifact_source_channel_indices_by_context=source_channel_indices,
        artifact_scale_range=(1.0, 1.0),
        max_jitter_samples=0,
        rng=np.random.default_rng(0),
    )

    assert dataset["noisy_context"].shape == (4, 3, 1, 10)
    assert dataset["artifact_source_ids"].shape == (4,)
    assert set(dataset["artifact_source_ids"].tolist()).issubset({"source_a", "source_b"})
    assert dataset["artifact_source_paths"].shape == (4,)
    assert stats["n_artifact_sources"] == 2
