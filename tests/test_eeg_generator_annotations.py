from __future__ import annotations

import numpy as np

from facet.misc import ChannelSchema, SpikeParams, generate_synthetic_eeg


def test_generate_synthetic_eeg_annotates_spike_onsets() -> None:
    raw, spike_events = generate_synthetic_eeg(
        duration=20.0,
        sfreq=512.0,
        channel_schema=ChannelSchema(eeg_channels=4, eog_channels=0, ecg_channels=0),
        spike_params=SpikeParams(
            enabled=True,
            spike_rate=60.0,
            spike_amplitude=120.0,
            annotate=True,
            annotation_description="spike_onset",
        ),
        random_seed=7,
    )

    assert spike_events
    assert len(raw.annotations) == len(spike_events)
    assert set(raw.annotations.description) == {"spike_onset"}
    np.testing.assert_allclose(
        raw.annotations.onset,
        [event["onset_time"] for event in spike_events],
        atol=1.0 / raw.info["sfreq"],
    )


def test_generate_synthetic_eeg_can_disable_spike_annotations() -> None:
    raw, spike_events = generate_synthetic_eeg(
        duration=20.0,
        sfreq=512.0,
        channel_schema=ChannelSchema(eeg_channels=4, eog_channels=0, ecg_channels=0),
        spike_params=SpikeParams(
            enabled=True,
            spike_rate=60.0,
            spike_amplitude=120.0,
            annotate=False,
        ),
        random_seed=7,
    )

    assert spike_events
    assert len(raw.annotations) == 0
