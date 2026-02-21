"""
Pytest configuration and fixtures for FACETpy tests.

This module provides common fixtures and utilities for testing.
"""

import numpy as np
import mne
import pytest
from pathlib import Path
import tempfile
import shutil

from facet.core import ProcessingContext, ProcessingMetadata


# Test data parameters
TEST_SFREQ = 250  # Hz
TEST_N_CHANNELS = 4
TEST_DURATION = 10  # seconds
TEST_N_TRIGGERS = 10
TEST_ARTIFACT_LENGTH = 50  # samples


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_raw():
    """Create a sample MNE Raw object for testing."""
    n_samples = int(TEST_SFREQ * TEST_DURATION)

    # Create random data
    data = np.random.randn(TEST_N_CHANNELS, n_samples) * 1e-6

    # Create info structure
    ch_names = [f'EEG{i+1:03d}' for i in range(TEST_N_CHANNELS)]
    ch_types = ['eeg'] * TEST_N_CHANNELS
    info = mne.create_info(ch_names=ch_names, sfreq=TEST_SFREQ, ch_types=ch_types)

    # Create Raw object
    raw = mne.io.RawArray(data, info, verbose=False)

    return raw


@pytest.fixture
def sample_raw_with_artifacts():
    """Create a sample Raw object with simulated fMRI artifacts."""
    n_samples = int(TEST_SFREQ * TEST_DURATION)

    # Base EEG signal
    data = np.random.randn(TEST_N_CHANNELS, n_samples) * 1e-6

    # Add simulated artifacts at regular intervals
    artifact_interval = n_samples // TEST_N_TRIGGERS
    artifact_amplitude = 50e-6  # 50 µV

    for i in range(TEST_N_TRIGGERS):
        start = i * artifact_interval
        end = start + TEST_ARTIFACT_LENGTH
        if end < n_samples:
            # Simple square wave artifact
            data[:, start:end] += artifact_amplitude

    # Create info
    ch_names = [f'EEG{i+1:03d}' for i in range(TEST_N_CHANNELS)]
    ch_types = ['eeg'] * TEST_N_CHANNELS
    info = mne.create_info(ch_names=ch_names, sfreq=TEST_SFREQ, ch_types=ch_types)

    raw = mne.io.RawArray(data, info, verbose=False)

    return raw


@pytest.fixture
def sample_triggers():
    """Generate sample trigger positions."""
    # Evenly spaced triggers
    n_samples = int(TEST_SFREQ * TEST_DURATION)
    interval = n_samples // TEST_N_TRIGGERS
    triggers = np.arange(0, n_samples, interval)[:TEST_N_TRIGGERS]
    return triggers.astype(int)


@pytest.fixture
def sample_context(sample_raw, sample_triggers):
    """Create a sample ProcessingContext."""
    metadata = ProcessingMetadata()
    metadata.triggers = sample_triggers
    metadata.artifact_length = TEST_ARTIFACT_LENGTH
    metadata.upsampling_factor = 10
    metadata.artifact_to_trigger_offset = 0.0

    context = ProcessingContext(
        raw=sample_raw,
        raw_original=sample_raw.copy(),
        metadata=metadata
    )

    return context


@pytest.fixture
def sample_edf_file(temp_dir):
    """Create a sample EDF file with triggers and pre/post-acquisition padding."""
    rng = np.random.RandomState(0)

    # Add 2 s of pre-acquisition and 2 s of post-acquisition padding so that
    # SNR / RMS calculators can find clean reference data outside the
    # acquisition window.
    pre_acq_samples = 2 * TEST_SFREQ   # 500 samples @ 250 Hz
    post_acq_samples = 2 * TEST_SFREQ  # 500 samples @ 250 Hz
    acq_n_samples = TEST_N_TRIGGERS * (TEST_SFREQ // TEST_N_TRIGGERS) * TEST_N_TRIGGERS
    total_samples = pre_acq_samples + acq_n_samples + post_acq_samples

    data = rng.randn(TEST_N_CHANNELS, total_samples) * 1e-6

    # Add simulated artifacts at regular intervals starting after pre-acq padding
    interval = acq_n_samples // TEST_N_TRIGGERS
    artifact_amplitude = 50e-6
    trigger_positions = []
    for i in range(TEST_N_TRIGGERS):
        start = pre_acq_samples + i * interval
        end = start + TEST_ARTIFACT_LENGTH
        if end <= total_samples:
            data[:, start:end] += artifact_amplitude
            trigger_positions.append(start)

    ch_names = [f'EEG{i+1:03d}' for i in range(TEST_N_CHANNELS)]
    info = mne.create_info(ch_names=ch_names, sfreq=TEST_SFREQ, ch_types=['eeg'] * TEST_N_CHANNELS)
    raw = mne.io.RawArray(data, info, verbose=False)

    trigger_times = np.array(trigger_positions) / TEST_SFREQ
    annotations = mne.Annotations(
        onset=trigger_times,
        duration=np.zeros(len(trigger_times)),
        description=['1'] * len(trigger_times)
    )
    raw.set_annotations(annotations)

    edf_path = temp_dir / "test_data.edf"
    raw.export(str(edf_path), verbose=False)
    return edf_path


@pytest.fixture
def sample_context_with_noise(sample_context):
    """Create a context with estimated noise."""
    noise = np.random.randn(*sample_context.get_raw()._data.shape) * 1e-7
    sample_context.set_estimated_noise(noise)
    return sample_context


# Helper functions

def assert_raw_equal(raw1, raw2, rtol=1e-5):
    """Assert that two Raw objects are approximately equal."""
    assert raw1.info['sfreq'] == raw2.info['sfreq']
    assert raw1.ch_names == raw2.ch_names
    assert raw1._data.shape == raw2._data.shape
    np.testing.assert_allclose(raw1._data, raw2._data, rtol=rtol)


def assert_context_valid(context):
    """Assert that a ProcessingContext is valid."""
    assert isinstance(context, ProcessingContext)
    assert context.get_raw() is not None
    assert context.metadata is not None


def create_mock_processor(name="mock", process_fn=None):
    """Create a mock processor for testing.

    Args:
        name: Processor name
        process_fn: Optional function to use for processing

    Returns:
        Mock processor class instance
    """
    from facet.core import Processor

    class MockProcessor(Processor):
        def __init__(self):
            self.name = name
            self.description = f"Mock processor: {name}"
            self.call_count = 0
            self._process_fn = process_fn
            super().__init__()

        def process(self, context):
            self.call_count += 1
            if self._process_fn:
                return self._process_fn(context)
            return context

    return MockProcessor()


# Markers and test organization

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for workflows"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time"
    )
    config.addinivalue_line(
        "markers", "requires_data: Tests that require data files"
    )
    config.addinivalue_line(
        "markers", "requires_c_extension: Tests that need C extension"
    )
