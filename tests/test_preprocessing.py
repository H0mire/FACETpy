"""
Tests for preprocessing processors.
"""

import pytest
import numpy as np
import mne

from facet.preprocessing import (
    TriggerDetector,
    UpSample,
    DownSample,
    TriggerAligner,
    HighPassFilter,
    LowPassFilter,
    BandPassFilter
)
from facet.core import ProcessingContext, ProcessorValidationError


@pytest.mark.unit
class TestTriggerDetector:
    """Tests for TriggerDetector processor."""

    def test_trigger_detection_from_annotations(self, sample_edf_file):
        """Test detecting triggers from annotations."""
        from facet.io import EDFLoader

        # Load file with triggers
        loader = EDFLoader(path=str(sample_edf_file), preload=True)
        context = loader.execute(ProcessingContext())

        # Detect triggers
        detector = TriggerDetector(regex=r"\b1\b")
        result = detector.execute(context)

        # Check triggers were found
        assert result.has_triggers()
        triggers = result.get_triggers()
        assert len(triggers) > 0

    def test_trigger_detection_sets_artifact_length(self, sample_edf_file):
        """Test that trigger detection calculates artifact length."""
        from facet.io import EDFLoader

        loader = EDFLoader(path=str(sample_edf_file), preload=True)
        context = loader.execute(ProcessingContext())

        detector = TriggerDetector(regex=r"\b1\b")
        result = detector.execute(context)

        # Check artifact length was calculated
        assert result.get_artifact_length() is not None
        assert result.get_artifact_length() > 0

    def test_no_triggers_found(self, sample_raw):
        """Test behavior when no triggers are found."""
        context = ProcessingContext(raw=sample_raw)

        # Try to detect with pattern that won't match
        detector = TriggerDetector(regex=r"NONEXISTENT")
        result = detector.execute(context)

        # Should return unchanged context
        assert not result.has_triggers()


@pytest.mark.unit
class TestResampling:
    """Tests for resampling processors."""

    def test_upsample(self, sample_context):
        """Test upsampling."""
        original_sfreq = sample_context.get_raw().info['sfreq']

        upsampler = UpSample(factor=2)
        result = upsampler.execute(sample_context)

        new_sfreq = result.get_raw().info['sfreq']

        # Check frequency doubled
        assert new_sfreq == original_sfreq * 2

    def test_downsample(self, sample_context):
        """Test downsampling."""
        # First upsample
        upsampler = UpSample(factor=4)
        context_upsampled = upsampler.execute(sample_context)

        original_sfreq = context_upsampled.get_raw().info['sfreq']

        # Then downsample
        downsampler = DownSample(factor=2)
        result = downsampler.execute(context_upsampled)

        new_sfreq = result.get_raw().info['sfreq']

        # Check frequency halved
        assert new_sfreq == original_sfreq / 2

    def test_resampling_updates_triggers(self, sample_context):
        """Test that resampling updates trigger positions."""
        original_triggers = sample_context.get_triggers().copy()
        original_sfreq = sample_context.get_raw().info['sfreq']

        # Upsample by 2
        upsampler = UpSample(factor=2)
        result = upsampler.execute(sample_context)

        new_triggers = result.get_triggers()
        new_sfreq = result.get_raw().info['sfreq']

        # Triggers should be scaled
        scaling_factor = new_sfreq / original_sfreq
        expected_triggers = (original_triggers * scaling_factor).astype(int)

        np.testing.assert_array_equal(new_triggers, expected_triggers)

    def test_upsample_downsample_roundtrip(self, sample_context):
        """Test upsampling then downsampling returns to original."""
        original_sfreq = sample_context.get_raw().info['sfreq']
        original_shape = sample_context.get_raw()._data.shape

        # Upsample then downsample
        upsampler = UpSample(factor=4)
        downsampler = DownSample(factor=4)

        context_up = upsampler.execute(sample_context)
        context_down = downsampler.execute(context_up)

        # Check we're back to original frequency
        final_sfreq = context_down.get_raw().info['sfreq']
        assert final_sfreq == original_sfreq

        # Shape should be similar (may differ slightly due to resampling)
        final_shape = context_down.get_raw()._data.shape
        assert final_shape[0] == original_shape[0]  # Same channels
        assert abs(final_shape[1] - original_shape[1]) < 10  # Similar samples


@pytest.mark.unit
class TestTriggerAligner:
    """Tests for TriggerAligner processor."""

    def test_trigger_alignment(self, sample_context):
        """Test trigger alignment."""
        aligner = TriggerAligner(ref_trigger_index=0)
        result = aligner.execute(sample_context)

        # Check triggers were aligned
        assert result.has_triggers()

        # Triggers may have changed slightly
        original_triggers = sample_context.get_triggers()
        aligned_triggers = result.get_triggers()

        assert len(aligned_triggers) == len(original_triggers)

    def test_alignment_requires_triggers(self, sample_raw):
        """Test that alignment requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        aligner = TriggerAligner(ref_trigger_index=0)

        with pytest.raises(ProcessorValidationError):
            aligner.execute(context)

    def test_alignment_requires_artifact_length(self, sample_context):
        """Test that alignment requires artifact length."""
        # Remove artifact length
        metadata = sample_context.metadata.copy()
        metadata.artifact_length = None
        context = sample_context.with_metadata(metadata)

        aligner = TriggerAligner(ref_trigger_index=0)

        with pytest.raises(ProcessorValidationError):
            aligner.execute(context)


@pytest.mark.unit
class TestFiltering:
    """Tests for filtering processors."""

    def test_highpass_filter(self, sample_context):
        """Test highpass filtering."""
        original_data = sample_context.get_raw()._data.copy()

        hpf = HighPassFilter(freq=1.0)
        result = hpf.execute(sample_context)

        filtered_data = result.get_raw()._data

        # Data should have changed
        assert not np.array_equal(original_data, filtered_data)

        # Check it's still reasonable
        assert not np.any(np.isnan(filtered_data))
        assert not np.any(np.isinf(filtered_data))

    def test_lowpass_filter(self, sample_context):
        """Test lowpass filtering."""
        original_data = sample_context.get_raw()._data.copy()

        lpf = LowPassFilter(freq=40.0)
        result = lpf.execute(sample_context)

        filtered_data = result.get_raw()._data

        # Data should have changed
        assert not np.array_equal(original_data, filtered_data)

    def test_bandpass_filter(self, sample_context):
        """Test bandpass filtering."""
        original_data = sample_context.get_raw()._data.copy()

        bpf = BandPassFilter(l_freq=1.0, h_freq=40.0)
        result = bpf.execute(sample_context)

        filtered_data = result.get_raw()._data

        # Data should have changed
        assert not np.array_equal(original_data, filtered_data)

    def test_filter_also_filters_noise(self, sample_context_with_noise):
        """Test that filtering also filters estimated noise."""
        original_noise = sample_context_with_noise.get_estimated_noise().copy()

        hpf = HighPassFilter(freq=1.0)
        result = hpf.execute(sample_context_with_noise)

        filtered_noise = result.get_estimated_noise()

        # Noise should have been filtered too
        assert not np.array_equal(original_noise, filtered_noise)

    def test_filter_chain(self, sample_context):
        """Test chaining multiple filters."""
        from facet.core import Pipeline

        pipeline = Pipeline([
            HighPassFilter(freq=0.5),
            LowPassFilter(freq=50.0)
        ])

        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
        filtered_data = result.context.get_raw()._data

        # Should be finite
        assert np.all(np.isfinite(filtered_data))


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for preprocessing workflow."""

    def test_full_preprocessing_workflow(self, sample_edf_file):
        """Test complete preprocessing workflow."""
        from facet.io import EDFLoader
        from facet.core import Pipeline

        pipeline = Pipeline([
            EDFLoader(path=str(sample_edf_file), preload=True),
            TriggerDetector(regex=r"\b1\b"),
            UpSample(factor=2),
            TriggerAligner(ref_trigger_index=0),
            HighPassFilter(freq=0.5),
            DownSample(factor=2)
        ])

        result = pipeline.run()

        assert result.success is True
        assert result.context.has_triggers()
        assert result.context.get_raw() is not None
