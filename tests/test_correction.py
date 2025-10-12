"""
Tests for correction processors.
"""

import pytest
import numpy as np

from facet.correction import AASCorrection, ANCCorrection, PCACorrection
from facet.core import ProcessingContext, ProcessorValidationError


@pytest.mark.unit
class TestAASCorrection:
    """Tests for AASCorrection processor."""

    def test_aas_initialization(self):
        """Test AAS processor initialization."""
        aas = AASCorrection(
            window_size=30,
            correlation_threshold=0.975
        )

        assert aas.window_size == 30
        assert aas.correlation_threshold == 0.975

    def test_aas_requires_triggers(self, sample_raw):
        """Test that AAS requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        aas = AASCorrection(window_size=5)

        with pytest.raises(ProcessorValidationError):
            aas.execute(context)

    def test_aas_requires_artifact_length(self, sample_context):
        """Test that AAS requires artifact length."""
        # Remove artifact length
        metadata = sample_context.metadata.copy()
        metadata.artifact_length = None
        context = sample_context.with_metadata(metadata)

        aas = AASCorrection(window_size=5)

        with pytest.raises(ProcessorValidationError):
            aas.execute(context)

    def test_aas_execution(self, sample_context):
        """Test AAS execution."""
        original_data = sample_context.get_raw()._data.copy()

        aas = AASCorrection(window_size=5)
        result = aas.execute(sample_context)

        # Check result is valid
        assert result.get_raw() is not None
        corrected_data = result.get_raw()._data

        # Data should have changed
        assert not np.array_equal(original_data, corrected_data)

        # Check estimated noise was created
        assert result.has_estimated_noise()
        noise = result.get_estimated_noise()
        assert noise.shape == corrected_data.shape

    def test_aas_reduces_artifact_amplitude(self, sample_raw_with_artifacts, sample_triggers):
        """Test that AAS actually reduces artifact amplitude."""
        from facet.core import ProcessingMetadata

        # Create context with artifacts
        metadata = ProcessingMetadata()
        metadata.triggers = sample_triggers
        metadata.artifact_length = 50
        context = ProcessingContext(
            raw=sample_raw_with_artifacts,
            raw_original=sample_raw_with_artifacts.copy(),
            metadata=metadata
        )

        # Calculate RMS before
        rms_before = np.sqrt(np.mean(sample_raw_with_artifacts._data ** 2))

        # Apply AAS
        aas = AASCorrection(window_size=5)
        result = aas.execute(context)

        # Calculate RMS after
        rms_after = np.sqrt(np.mean(result.get_raw()._data ** 2))

        # RMS should be reduced (artifacts removed)
        assert rms_after < rms_before

    def test_aas_with_small_window(self, sample_context):
        """Test AAS with small window size."""
        aas = AASCorrection(window_size=3)
        result = aas.execute(sample_context)

        assert result.get_raw() is not None

    def test_aas_with_realignment(self, sample_context):
        """Test AAS with trigger realignment."""
        aas = AASCorrection(
            window_size=5,
            realign_after_averaging=True
        )
        result = aas.execute(sample_context)

        # Triggers may have been adjusted
        assert result.has_triggers()


@pytest.mark.unit
class TestANCCorrection:
    """Tests for ANCCorrection processor."""

    def test_anc_initialization(self):
        """Test ANC processor initialization."""
        anc = ANCCorrection(filter_order=5, hp_freq=1.0)

        assert anc.filter_order == 5
        assert anc.hp_freq == 1.0

    def test_anc_requires_estimated_noise(self, sample_context):
        """Test that ANC requires estimated noise."""
        anc = ANCCorrection(filter_order=5)

        with pytest.raises(ProcessorValidationError):
            anc.execute(sample_context)

    def test_anc_execution(self, sample_context_with_noise):
        """Test ANC execution."""
        original_data = sample_context_with_noise.get_raw()._data.copy()

        anc = ANCCorrection(filter_order=5, use_c_extension=False)
        result = anc.execute(sample_context_with_noise)

        corrected_data = result.get_raw()._data

        # Data may have changed (depends on noise)
        assert corrected_data.shape == original_data.shape
        assert not np.any(np.isnan(corrected_data))
        assert not np.any(np.isinf(corrected_data))

    def test_anc_python_fallback(self, sample_context_with_noise):
        """Test ANC Python fallback when C extension unavailable."""
        anc = ANCCorrection(filter_order=3, use_c_extension=False)
        result = anc.execute(sample_context_with_noise)

        # Should complete successfully
        assert result.get_raw() is not None

    @pytest.mark.requires_c_extension
    def test_anc_c_extension(self, sample_context_with_noise):
        """Test ANC with C extension."""
        anc = ANCCorrection(filter_order=5, use_c_extension=True)

        # May skip if C extension not available
        try:
            result = anc.execute(sample_context_with_noise)
            assert result.get_raw() is not None
        except ImportError:
            pytest.skip("C extension not available")


@pytest.mark.unit
class TestPCACorrection:
    """Tests for PCACorrection processor."""

    def test_pca_initialization(self):
        """Test PCA processor initialization."""
        pca = PCACorrection(n_components=0.95)

        assert pca.n_components == 0.95

    def test_pca_requires_triggers(self, sample_raw):
        """Test that PCA requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        pca = PCACorrection(n_components=0.95)

        with pytest.raises(ProcessorValidationError):
            pca.execute(context)

    def test_pca_execution(self, sample_context):
        """Test PCA execution."""
        original_data = sample_context.get_raw()._data.copy()

        pca = PCACorrection(n_components=2)  # Use integer for predictable behavior
        result = pca.execute(sample_context)

        corrected_data = result.get_raw()._data

        # Data should have changed
        assert not np.array_equal(original_data, corrected_data)

        # Check estimated noise was updated
        assert result.has_estimated_noise()

    def test_pca_with_variance_threshold(self, sample_context):
        """Test PCA with variance threshold."""
        pca = PCACorrection(n_components=0.95)  # Keep 95% variance
        result = pca.execute(sample_context)

        assert result.get_raw() is not None

    def test_pca_with_zero_components_skips(self, sample_context):
        """Test that PCA with n_components=0 skips processing."""
        original_data = sample_context.get_raw()._data.copy()

        pca = PCACorrection(n_components=0)
        result = pca.execute(sample_context)

        # Data should be unchanged
        np.testing.assert_array_equal(result.get_raw()._data, original_data)


@pytest.mark.integration
class TestCorrectionPipeline:
    """Integration tests for correction pipeline."""

    def test_aas_anc_pipeline(self, sample_context):
        """Test AAS followed by ANC."""
        from facet.core import Pipeline

        pipeline = Pipeline([
            AASCorrection(window_size=5),
            ANCCorrection(filter_order=3, use_c_extension=False)
        ])

        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
        assert result.context.get_raw() is not None

    def test_full_correction_pipeline(self, sample_edf_file):
        """Test full correction pipeline."""
        from facet.io import EDFLoader
        from facet.preprocessing import TriggerDetector, UpSample, DownSample
        from facet.core import Pipeline

        pipeline = Pipeline([
            EDFLoader(path=str(sample_edf_file), preload=True),
            TriggerDetector(regex=r"\b1\b"),
            UpSample(factor=2),
            AASCorrection(window_size=5),
            DownSample(factor=2)
        ])

        result = pipeline.run()

        assert result.success is True

    def test_aas_pca_pipeline(self, sample_context):
        """Test AAS followed by PCA."""
        from facet.core import Pipeline

        pipeline = Pipeline([
            AASCorrection(window_size=5),
            PCACorrection(n_components=2)
        ])

        result = pipeline.run(initial_context=sample_context)

        assert result.success is True
