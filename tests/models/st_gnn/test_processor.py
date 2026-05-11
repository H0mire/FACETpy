"""End-to-end processor tests for the ST-GNN inference adapter."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessingMetadata, ProcessorValidationError

torch = pytest.importorskip("torch")

# torch.jit.trace emits TracerWarnings that the project-level
# `filterwarnings = error` would otherwise promote to test failures.
pytestmark = pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")

from facet.models.st_gnn import (  # noqa: E402  (after importorskip)
    NIAZY_PROOF_FIT_CHANNELS,
    SpatiotemporalGNNCorrection,
    build_model,
)


def _make_niazy_like_context(n_triggers: int = 12, sfreq: float = 4096.0) -> ProcessingContext:
    samples_per_epoch = 600
    n_samples = (n_triggers + 2) * samples_per_epoch
    rng = np.random.default_rng(0)
    data = rng.standard_normal((30, n_samples)).astype(np.float64) * 1e-6

    info = mne.create_info(
        ch_names=list(NIAZY_PROOF_FIT_CHANNELS),
        sfreq=sfreq,
        ch_types=["eeg"] * 30,
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    triggers = (np.arange(n_triggers) * samples_per_epoch + samples_per_epoch).astype(int)

    metadata = ProcessingMetadata()
    metadata.triggers = triggers
    metadata.artifact_length = samples_per_epoch
    metadata.artifact_to_trigger_offset = 0.0

    return ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=metadata)


def _export_test_checkpoint(tmp_path: Path) -> Path:
    model = build_model(input_shape=(7, 30, 512)).eval()
    example = torch.randn(1, 7, 30, 512)
    scripted = torch.jit.trace(model, example)
    ckpt = tmp_path / "st_gnn.ts"
    scripted.save(str(ckpt))
    return ckpt


def test_processor_runs_end_to_end(tmp_path: Path) -> None:
    checkpoint = _export_test_checkpoint(tmp_path)
    context = _make_niazy_like_context()

    processor = SpatiotemporalGNNCorrection(
        checkpoint_path=str(checkpoint),
        context_epochs=7,
        epoch_samples=512,
        artifact_to_trigger_offset=0.0,
        device="cpu",
    )

    result = processor.execute(context)
    out_raw = result.get_raw()
    assert out_raw.get_data().shape == context.get_raw().get_data().shape
    assert result.has_estimated_noise()
    noise = result.get_estimated_noise()
    assert noise.shape == context.get_raw().get_data().shape


def test_processor_rejects_missing_channel(tmp_path: Path) -> None:
    checkpoint = _export_test_checkpoint(tmp_path)
    context = _make_niazy_like_context()

    raw = context.get_raw()
    rename_map = {raw.ch_names[0]: "Custom1"}
    raw_renamed = raw.copy().rename_channels(rename_map)
    bad_context = ProcessingContext(
        raw=raw_renamed,
        raw_original=raw_renamed.copy(),
        metadata=context.metadata.copy(),
    )

    processor = SpatiotemporalGNNCorrection(
        checkpoint_path=str(checkpoint),
        context_epochs=7,
        epoch_samples=512,
        artifact_to_trigger_offset=0.0,
        device="cpu",
    )

    with pytest.raises(ProcessorValidationError):
        processor.execute(bad_context)


def test_processor_rejects_too_few_triggers(tmp_path: Path) -> None:
    checkpoint = _export_test_checkpoint(tmp_path)
    context = _make_niazy_like_context(n_triggers=4)

    processor = SpatiotemporalGNNCorrection(
        checkpoint_path=str(checkpoint),
        context_epochs=7,
        epoch_samples=512,
        artifact_to_trigger_offset=0.0,
        device="cpu",
    )

    with pytest.raises(ProcessorValidationError):
        processor.execute(context)


def test_processor_history_records_execution(tmp_path: Path) -> None:
    checkpoint = _export_test_checkpoint(tmp_path)
    context = _make_niazy_like_context()
    processor = SpatiotemporalGNNCorrection(
        checkpoint_path=str(checkpoint),
        context_epochs=7,
        epoch_samples=512,
        artifact_to_trigger_offset=0.0,
        device="cpu",
    )

    result = processor.execute(context)
    history = result.get_history()
    assert any(entry.name == "st_gnn_correction" for entry in history)
