"""Tests for trigger-context deep learning correction."""

import mne
import numpy as np
import pytest

from facet.core import Pipeline, ProcessingContext, ProcessingMetadata
from facet.correction import EpochContextDeepLearningCorrection


def _make_context(data: np.ndarray, triggers: list[int]) -> ProcessingContext:
    info = mne.create_info(
        ch_names=[f"EEG{i + 1:03d}" for i in range(data.shape[0])],
        sfreq=100.0,
        ch_types=["eeg"] * data.shape[0],
    )
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    metadata = ProcessingMetadata(
        triggers=np.asarray(triggers, dtype=int),
        artifact_to_trigger_offset=0.0,
    )
    return ProcessingContext(raw=raw, raw_original=raw.copy(), metadata=metadata)


def _write_center_epoch_model(tmp_path, context_epochs: int, epoch_samples: int):
    torch = pytest.importorskip("torch")

    class CenterEpochModel(torch.nn.Module):
        def forward(self, x):
            return x[:, x.shape[1] // 2, :, :]

    model = CenterEpochModel().eval()
    traced = torch.jit.trace(model, torch.zeros(1, context_epochs, 1, epoch_samples))
    path = tmp_path / "center_epoch_model.ts"
    traced.save(str(path))
    return path


@pytest.mark.unit
def test_epoch_context_correction_subtracts_center_epoch(tmp_path):
    model_path = _write_center_epoch_model(tmp_path, context_epochs=3, epoch_samples=10)
    data = np.vstack(
        [
            np.arange(50, dtype=float),
            np.arange(50, dtype=float) * 2.0,
        ]
    )
    context = _make_context(data, triggers=[0, 10, 20, 30, 40])

    result = EpochContextDeepLearningCorrection(
        model_path,
        context_epochs=3,
        epoch_samples=10,
        demean_input=False,
        remove_prediction_mean=False,
    ).execute(context)

    corrected = result.get_data(copy=False)
    expected = data.copy()
    expected[:, 10:30] = 0.0
    np.testing.assert_allclose(corrected, expected)
    np.testing.assert_allclose(result.get_estimated_noise()[:, 10:30], data[:, 10:30])
    np.testing.assert_allclose(result.get_estimated_noise()[:, :10], 0.0)

    run = result.metadata.custom["epoch_context_deep_learning_runs"][0]
    assert run["corrected_epochs"] == 2
    assert run["epoch_length_min"] == 10
    assert run["epoch_length_max"] == 10


@pytest.mark.unit
def test_epoch_context_correction_supports_channel_sequential(tmp_path):
    model_path = _write_center_epoch_model(tmp_path, context_epochs=3, epoch_samples=10)
    data = np.vstack(
        [
            np.arange(50, dtype=float),
            np.arange(50, dtype=float) * 3.0,
        ]
    )
    context = _make_context(data, triggers=[0, 10, 20, 30, 40])
    processor = EpochContextDeepLearningCorrection(
        model_path,
        context_epochs=3,
        epoch_samples=10,
        demean_input=False,
        remove_prediction_mean=False,
    )
    pipeline = Pipeline([processor])

    serial = pipeline.run(initial_context=context, channel_sequential=False, show_progress=False)
    channel_seq = pipeline.run(initial_context=context, channel_sequential=True, show_progress=False)

    np.testing.assert_allclose(channel_seq.context.get_data(copy=False), serial.context.get_data(copy=False))
    np.testing.assert_allclose(channel_seq.context.get_estimated_noise(), serial.context.get_estimated_noise())


@pytest.mark.unit
def test_epoch_context_correction_accepts_variable_trigger_deltas(tmp_path):
    model_path = _write_center_epoch_model(tmp_path, context_epochs=3, epoch_samples=11)
    data = np.vstack([np.linspace(0.0, 1.0, 60)])
    context = _make_context(data, triggers=[0, 8, 20, 33, 43, 57])

    result = EpochContextDeepLearningCorrection(
        model_path,
        context_epochs=3,
        epoch_samples=None,
        demean_input=False,
        remove_prediction_mean=False,
    ).execute(context)

    run = result.metadata.custom["epoch_context_deep_learning_runs"][0]
    assert run["corrected_epochs"] == 3
    assert run["epoch_length_min"] == 8
    assert run["epoch_length_max"] == 14
    assert result.get_estimated_noise().shape == data.shape


@pytest.mark.unit
def test_epoch_context_correction_removes_prediction_mean_by_default(tmp_path):
    model_path = _write_center_epoch_model(tmp_path, context_epochs=3, epoch_samples=10)
    data = np.vstack([np.arange(50, dtype=float)])
    context = _make_context(data, triggers=[0, 10, 20, 30, 40])

    result = EpochContextDeepLearningCorrection(
        model_path,
        context_epochs=3,
        epoch_samples=10,
    ).execute(context)

    estimated = result.get_estimated_noise()
    np.testing.assert_allclose(np.mean(estimated[:, 10:20], axis=-1), 0.0)
    assert result.metadata.custom["epoch_context_deep_learning_runs"][0]["remove_prediction_mean"] is True
