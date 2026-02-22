"""Regression tests for channel-sequential execution."""

import mne
import numpy as np
import pytest

from facet.core import Pipeline, ProcessingContext, Processor


class _SubtractEstimatedNoise(Processor):
    """Subtract the current estimated noise from raw data."""

    name = "subtract_estimated_noise"
    channel_wise = True
    parallel_safe = True
    modifies_raw = True

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        noise = context.get_estimated_noise()
        for ch_idx in range(raw._data.shape[0]):
            raw._data[ch_idx] -= noise[ch_idx]
        return context.with_raw(raw)


class _RunOnceSetScale(Processor):
    """Set a metadata scale exactly once."""

    name = "run_once_set_scale"
    channel_wise = True
    run_once = True
    modifies_raw = False

    def process(self, context: ProcessingContext) -> ProcessingContext:
        metadata = context.metadata.copy()
        metadata.custom["scale"] = 2.0
        return context.with_metadata(metadata)


class _ApplyScaleFromMetadata(Processor):
    """Scale raw data by metadata value."""

    name = "apply_scale_from_metadata"
    channel_wise = True
    modifies_raw = True

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        scale = float(context.metadata.custom.get("scale", 1.0))
        raw._data *= scale
        return context.with_raw(raw)


class _SetBiasFromFirstChannel(Processor):
    """Set metadata based on channel index 0 in the current context."""

    name = "set_bias_from_first_channel"
    channel_wise = True
    modifies_raw = False

    def process(self, context: ProcessingContext) -> ProcessingContext:
        metadata = context.metadata.copy()
        metadata.custom["bias"] = float(np.mean(context.get_raw()._data[0]))
        return context.with_metadata(metadata)


class _SubtractMetadataBias(Processor):
    """Subtract metadata bias from every channel in the current context."""

    name = "subtract_metadata_bias"
    channel_wise = True
    modifies_raw = True

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        bias = float(context.metadata.custom.get("bias", 0.0))
        raw._data -= bias
        return context.with_raw(raw)


@pytest.mark.unit
def test_channel_sequential_uses_per_channel_noise():
    """Channel-sequential mode must map noise rows to the current channel."""
    n_times = 64
    data = np.vstack(
        [
            np.full(n_times, 1.0),
            np.full(n_times, 2.0),
            np.full(n_times, 3.0),
        ]
    )
    info = mne.create_info(ch_names=["EEG001", "EEG002", "EEG003"], sfreq=250.0, ch_types=["eeg", "eeg", "eeg"])
    raw = mne.io.RawArray(data.copy(), info, verbose=False)

    context = ProcessingContext(raw=raw, raw_original=raw.copy())
    context.set_estimated_noise(data.copy())

    pipeline = Pipeline([_SubtractEstimatedNoise()])

    serial = pipeline.run(initial_context=context, channel_sequential=False, show_progress=False)
    channel_seq = pipeline.run(initial_context=context, channel_sequential=True, show_progress=False)

    np.testing.assert_allclose(serial.context.get_data(copy=False), np.zeros_like(data))
    np.testing.assert_allclose(channel_seq.context.get_data(copy=False), np.zeros_like(data))


@pytest.mark.unit
def test_channel_sequential_propagates_run_once_metadata():
    """Metadata produced by run_once processors must be visible on all channels."""
    n_times = 32
    data = np.vstack(
        [
            np.full(n_times, 1.0),
            np.full(n_times, 3.0),
            np.full(n_times, 5.0),
        ]
    )
    info = mne.create_info(ch_names=["EEG001", "EEG002", "EEG003"], sfreq=250.0, ch_types=["eeg", "eeg", "eeg"])
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    context = ProcessingContext(raw=raw, raw_original=raw.copy())

    pipeline = Pipeline([_RunOnceSetScale(), _ApplyScaleFromMetadata()])

    serial = pipeline.run(initial_context=context, channel_sequential=False, show_progress=False)
    channel_seq = pipeline.run(initial_context=context, channel_sequential=True, show_progress=False)

    np.testing.assert_allclose(serial.context.get_data(copy=False), data * 2.0)
    np.testing.assert_allclose(channel_seq.context.get_data(copy=False), data * 2.0)


@pytest.mark.unit
def test_channel_sequential_freezes_metadata_to_first_channel_trajectory():
    """Non-run_once metadata drift must not vary across channels."""
    n_times = 40
    data = np.vstack(
        [
            np.full(n_times, 1.0),
            np.full(n_times, 2.0),
            np.full(n_times, 3.0),
        ]
    )
    info = mne.create_info(ch_names=["EEG001", "EEG002", "EEG003"], sfreq=250.0, ch_types=["eeg", "eeg", "eeg"])
    raw = mne.io.RawArray(data.copy(), info, verbose=False)
    context = ProcessingContext(raw=raw, raw_original=raw.copy())

    pipeline = Pipeline([_SetBiasFromFirstChannel(), _SubtractMetadataBias()])

    serial = pipeline.run(initial_context=context, channel_sequential=False, show_progress=False)
    channel_seq = pipeline.run(initial_context=context, channel_sequential=True, show_progress=False)

    expected = np.vstack(
        [
            np.zeros(n_times),
            np.ones(n_times),
            np.full(n_times, 2.0),
        ]
    )
    np.testing.assert_allclose(serial.context.get_data(copy=False), expected)
    np.testing.assert_allclose(channel_seq.context.get_data(copy=False), expected)
