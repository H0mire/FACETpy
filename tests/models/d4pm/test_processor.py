"""Tests for the D4PM correction processor and adapter."""

from __future__ import annotations

import mne
import numpy as np
import pytest
import torch

from facet.core import ProcessingContext
from facet.core.context import ProcessingMetadata
from facet.models.d4pm import D4PMArtifactCorrection, D4PMArtifactDiffusionAdapter
from facet.models.d4pm.training import D4PMTrainingModule


def _make_synthetic_context(
    n_channels: int = 3, sfreq: float = 4096.0, n_samples: int = 4096
) -> ProcessingContext:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_channels, n_samples)).astype(np.float32) * 1e-5
    info = mne.create_info(
        [f"EEG{i+1}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    triggers = np.arange(0, n_samples - 512, 512, dtype=int)
    metadata = ProcessingMetadata(triggers=triggers, artifact_to_trigger_offset=0.0)
    return ProcessingContext(raw=raw, metadata=metadata)


def _save_random_state_dict(tmp_path, *, epoch_samples=128, num_steps=20):
    module = D4PMTrainingModule(
        epoch_samples=epoch_samples,
        num_steps=num_steps,
        feats=16,
        d_model=32,
        d_ff=64,
        n_heads=2,
        n_layers=1,
        embed_dim=32,
    )
    ckpt_path = tmp_path / "last.pt"
    torch.save({"model_state_dict": module.state_dict()}, ckpt_path)
    return ckpt_path


@pytest.mark.unit
def test_processor_registers_with_correct_name():
    proc = D4PMArtifactCorrection(checkpoint_path="/dev/null", epoch_samples=128, num_steps=20)
    assert proc.name == "d4pm_correction"
    assert proc.requires_triggers is True
    assert proc.modifies_raw is True
    assert proc.channel_wise is True


@pytest.mark.unit
def test_adapter_predicts_artifact_shape(tmp_path):
    ckpt_path = _save_random_state_dict(tmp_path, epoch_samples=128, num_steps=20)

    n_channels = 3
    sfreq = 1024.0
    n_samples = 4 * 128
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((n_channels, n_samples)).astype(np.float32) * 1e-5)
    info = mne.create_info(
        [f"EEG{i+1}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    triggers = np.array([0, 128, 256, 384], dtype=int)
    metadata = ProcessingMetadata(triggers=triggers, artifact_to_trigger_offset=0.0)
    context = ProcessingContext(raw=raw, metadata=metadata)

    adapter = D4PMArtifactDiffusionAdapter(
        checkpoint_path=str(ckpt_path),
        epoch_samples=128,
        num_steps=20,
        feats=16,
        d_model=32,
        d_ff=64,
        n_heads=2,
        n_layers=1,
        embed_dim=32,
        sample_steps=3,
        data_consistency_weight=0.5,
        device="cpu",
    )
    adapter.validate_context(context)
    prediction = adapter.predict(context)

    assert prediction.artifact_data is not None
    assert prediction.artifact_data.shape == raw._data.shape
    assert prediction.metadata["sample_steps"] == 3


@pytest.mark.unit
def test_processor_runs_end_to_end(tmp_path):
    ckpt_path = _save_random_state_dict(tmp_path, epoch_samples=128, num_steps=20)

    n_channels = 2
    sfreq = 1024.0
    n_samples = 3 * 128
    rng = np.random.default_rng(1)
    data = rng.standard_normal((n_channels, n_samples)).astype(np.float32) * 1e-5
    info = mne.create_info(
        [f"EEG{i+1}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    triggers = np.array([0, 128, 256], dtype=int)
    metadata = ProcessingMetadata(triggers=triggers, artifact_to_trigger_offset=0.0)
    context = ProcessingContext(raw=raw, metadata=metadata)

    proc = D4PMArtifactCorrection(
        checkpoint_path=str(ckpt_path),
        epoch_samples=128,
        num_steps=20,
        feats=16,
        d_model=32,
        d_ff=64,
        n_heads=2,
        n_layers=1,
        embed_dim=32,
        sample_steps=2,
        data_consistency_weight=0.5,
        device="cpu",
    )
    result = proc.execute(context)
    assert result.get_raw().n_times == n_samples
    assert result.get_raw().get_data().shape == raw._data.shape
