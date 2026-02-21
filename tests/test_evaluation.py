"""
Tests for evaluation processors.
"""

import pytest
import numpy as np

from facet.evaluation import (
    SNRCalculator,
    RMSCalculator,
    MedianArtifactCalculator,
    MetricsReport,
    RMSResidualCalculator,
    FFTAllenCalculator,
    FFTNiazyCalculator,
    LegacySNRCalculator
)
from facet.core import ProcessingContext, ProcessorValidationError


@pytest.mark.unit
class TestSNRCalculator:
    """Tests for SNRCalculator processor."""

    def test_snr_calculation(self, sample_context):
        """Test SNR calculation."""
        snr_calc = SNRCalculator()
        result = snr_calc.execute(sample_context)

        # Check SNR was calculated
        metrics = result.metadata.custom.get('metrics', {})
        assert 'snr' in metrics
        assert isinstance(metrics['snr'], float)
        assert metrics['snr'] > 0

    def test_snr_requires_triggers(self, sample_raw):
        """Test that SNR calculator requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        snr_calc = SNRCalculator()

        with pytest.raises(ProcessorValidationError):
            snr_calc.execute(context)

    def test_snr_requires_original_raw(self, sample_context):
        """Test that SNR requires original raw data."""
        # Create context without original
        context = ProcessingContext(raw=sample_context.get_raw())

        snr_calc = SNRCalculator()

        with pytest.raises(ProcessorValidationError):
            snr_calc.execute(context)

    def test_snr_per_channel(self, sample_context):
        """Test per-channel SNR calculation."""
        snr_calc = SNRCalculator()
        result = snr_calc.execute(sample_context)

        metrics = result.metadata.custom.get('metrics', {})
        assert 'snr_per_channel' in metrics
        assert len(metrics['snr_per_channel']) > 0


@pytest.mark.unit
class TestRMSCalculator:
    """Tests for RMSCalculator processor."""

    def test_rms_calculation(self, sample_context):
        """Test RMS calculation."""
        rms_calc = RMSCalculator()
        result = rms_calc.execute(sample_context)

        # Check RMS ratio was calculated
        metrics = result.metadata.custom.get('metrics', {})
        assert 'rms_ratio' in metrics
        assert isinstance(metrics['rms_ratio'], float)
        assert metrics['rms_ratio'] > 0

    def test_rms_requires_triggers(self, sample_raw):
        """Test that RMS calculator requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        rms_calc = RMSCalculator()

        with pytest.raises(ProcessorValidationError):
            rms_calc.execute(context)

    def test_rms_requires_original_raw(self, sample_context):
        """Test that RMS requires original raw data."""
        context = ProcessingContext(raw=sample_context.get_raw())

        rms_calc = RMSCalculator()

        with pytest.raises(ProcessorValidationError):
            rms_calc.execute(context)

    def test_rms_per_channel(self, sample_context):
        """Test per-channel RMS calculation."""
        rms_calc = RMSCalculator()
        result = rms_calc.execute(sample_context)

        metrics = result.metadata.custom.get('metrics', {})
        assert 'rms_ratio_per_channel' in metrics


@pytest.mark.unit
class TestRMSResidualCalculator:
    """Tests for RMSResidualCalculator processor."""

    def test_rms_residual_calculation(self, sample_context):
        """Test RMS residual calculation."""
        rms_calc = RMSResidualCalculator()
        result = rms_calc.execute(sample_context)

        metrics = result.metadata.custom.get('metrics', {})
        assert 'rms_residual' in metrics
        assert isinstance(metrics['rms_residual'], float)
        assert metrics['rms_residual'] > 0

    def test_rms_residual_requires_triggers(self, sample_raw):
        """Test that RMS residual calculator requires triggers."""
        context = ProcessingContext(raw=sample_raw)
        rms_calc = RMSResidualCalculator()
        with pytest.raises(ProcessorValidationError):
            rms_calc.execute(context)

    def test_rms_residual_per_channel(self, sample_context):
        """Test per-channel RMS residual calculation."""
        rms_calc = RMSResidualCalculator()
        result = rms_calc.execute(sample_context)

        metrics = result.metadata.custom.get('metrics', {})
        assert 'rms_residual_per_channel' in metrics
        assert len(metrics['rms_residual_per_channel']) > 0


@pytest.mark.unit
class TestLegacySNRCalculator:
    """Tests for LegacySNRCalculator processor."""

    def test_legacy_snr_calculation(self, sample_context):
        """Test legacy SNR calculation."""
        snr_calc = LegacySNRCalculator()
        result = snr_calc.execute(sample_context)

        metrics = result.metadata.custom.get('metrics', {})
        assert 'legacy_snr' in metrics
        assert isinstance(metrics['legacy_snr'], float)
        assert metrics['legacy_snr'] > 0

    def test_legacy_snr_requires_original_raw(self, sample_context):
        """Test that legacy SNR requires original raw data."""
        context = ProcessingContext(raw=sample_context.get_raw())
        snr_calc = LegacySNRCalculator()
        with pytest.raises(ProcessorValidationError):
            snr_calc.execute(context)


@pytest.mark.unit
class TestMedianArtifactCalculator:
    """Tests for MedianArtifactCalculator processor."""

    def test_median_artifact_calculation(self, sample_context):
        """Test median artifact calculation."""
        median_calc = MedianArtifactCalculator()
        result = median_calc.execute(sample_context)

        # Check median artifact was calculated
        metrics = result.metadata.custom.get('metrics', {})
        assert 'median_artifact' in metrics
        assert isinstance(metrics['median_artifact'], float)
        assert metrics['median_artifact'] >= 0
        
        # Check for reference ratio (might be missing if sample data is too short for reference extraction)
        # But the sample_context usually provides enough data
        if 'median_artifact_reference' in metrics:
            assert 'median_artifact_ratio' in metrics

    def test_median_requires_triggers(self, sample_raw):
        """Test that median calculator requires triggers."""
        context = ProcessingContext(raw=sample_raw)

        median_calc = MedianArtifactCalculator()

        with pytest.raises(ProcessorValidationError):
            median_calc.execute(context)

    def test_median_requires_artifact_length(self, sample_context):
        """Test that median calculator requires artifact length."""
        metadata = sample_context.metadata.copy()
        metadata.artifact_length = None
        context = sample_context.with_metadata(metadata)

        median_calc = MedianArtifactCalculator()

        with pytest.raises(ProcessorValidationError):
            median_calc.execute(context)


@pytest.mark.unit
class TestFFTAllenCalculator:
    """Tests for FFTAllenCalculator processor."""

    def test_fft_allen_calculation(self, sample_context):
        """Test FFT Allen calculation."""
        fft_calc = FFTAllenCalculator()
        result = fft_calc.execute(sample_context)

        metrics = result.metadata.custom.get('metrics', {})
        assert 'fft_allen' in metrics
        assert isinstance(metrics['fft_allen'], dict)
        
        # Check bands
        bands = ['Delta', 'Theta', 'Alpha', 'Beta']
        for band in bands:
            if band in metrics['fft_allen']:
                assert isinstance(metrics['fft_allen'][band], float)
                assert metrics['fft_allen'][band] >= 0

    def test_fft_allen_requires_triggers(self, sample_raw):
        context = ProcessingContext(raw=sample_raw)
        fft_calc = FFTAllenCalculator()
        with pytest.raises(ProcessorValidationError):
            fft_calc.execute(context)


@pytest.mark.unit
class TestFFTNiazyCalculator:
    """Tests for FFTNiazyCalculator processor."""

    def test_fft_niazy_calculation(self, sample_context):
        """Test FFT Niazy calculation."""
        fft_calc = FFTNiazyCalculator()
        result = fft_calc.execute(sample_context)

        metrics = result.metadata.custom.get('metrics', {})
        assert 'fft_niazy' in metrics
        assert 'slice' in metrics['fft_niazy']
        # volume might be missing if slices_per_volume is not set

    def test_fft_niazy_with_volume(self, sample_context):
        """Test FFT Niazy with volume information."""
        metadata = sample_context.metadata.copy()
        metadata.slices_per_volume = 10 # fake value
        context = sample_context.with_metadata(metadata)
        
        fft_calc = FFTNiazyCalculator()
        result = fft_calc.execute(context)
        
        metrics = result.metadata.custom.get('metrics', {})
        assert 'volume' in metrics['fft_niazy']


@pytest.mark.unit
class TestMetricsReport:
    """Tests for MetricsReport processor."""

    def test_metrics_report(self, sample_context, capsys):
        """Test metrics report generation."""
        # Add some metrics
        metadata = sample_context.metadata.copy()
        metadata.custom['metrics'] = {
            'snr': 15.5,
            'rms_ratio': 2.3,
            'median_artifact': 1.5e-5,
            'rms_residual': 1.1,
            'fft_allen': {'Alpha': 5.0},
            'fft_niazy': {'slice': {'h1': -10.0}}
        }
        context = sample_context.with_metadata(metadata)

        # Generate report
        report = MetricsReport()
        result = report.execute(context)

        # Check it doesn't fail
        assert result is not None

    def test_metrics_report_storage(self, sample_context):
        """Test metrics report storage."""
        metadata = sample_context.metadata.copy()
        metadata.custom['metrics'] = {
            'snr': 15.5,
            'fft_allen': {'Alpha': 5.0}
        }
        context = sample_context.with_metadata(metadata)
        
        results = {}
        report = MetricsReport(store=results)
        report.execute(context)
        
        assert len(results) == 1
        key = list(results.keys())[0]
        assert results[key]['snr'] == 15.5
        assert results[key]['fft_allen_Alpha'] == 5.0


@pytest.mark.integration
class TestEvaluationPipeline:
    """Integration tests for evaluation pipeline."""

    def test_full_evaluation_pipeline(self, sample_context):
        """Test full evaluation workflow."""
        from facet.core import Pipeline

        pipeline = Pipeline([
            SNRCalculator(),
            RMSCalculator(),
            RMSResidualCalculator(),
            MedianArtifactCalculator(),
            LegacySNRCalculator(),
            FFTAllenCalculator(),
            FFTNiazyCalculator(),
            MetricsReport()
        ])

        result = pipeline.run(initial_context=sample_context)

        assert result.success is True

        # Check all metrics were calculated
        metrics = result.context.metadata.custom.get('metrics', {})
        assert 'snr' in metrics
        assert 'rms_ratio' in metrics
        assert 'rms_residual' in metrics
        assert 'median_artifact' in metrics
        assert 'legacy_snr' in metrics
        assert 'fft_allen' in metrics
        assert 'fft_niazy' in metrics

    def test_evaluation_after_correction(self, sample_edf_file):
        """Test evaluation after correction."""
        from facet.io import Loader
        from facet.preprocessing import TriggerDetector, UpSample, DownSample
        from facet.correction import AASCorrection
        from facet.core import Pipeline

        pipeline = Pipeline([
            Loader(path=str(sample_edf_file), preload=True),
            TriggerDetector(regex=r"\b1\b"),
            UpSample(factor=2),
            AASCorrection(window_size=5),
            DownSample(factor=2),
            SNRCalculator(),
            RMSCalculator(),
            RMSResidualCalculator(),
            MedianArtifactCalculator(),
            FFTAllenCalculator(),
            MetricsReport()
        ])

        result = pipeline.run()

        assert result.success is True

        # All metrics should be present
        metrics = result.context.metadata.custom.get('metrics', {})
        assert 'snr' in metrics
