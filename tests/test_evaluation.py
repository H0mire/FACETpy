"""
Tests for evaluation processors.
"""

import mne
import numpy as np
import pytest

from facet.core import ProcessingContext, ProcessorValidationError
from facet.evaluation import (
    FFTAllenCalculator,
    FFTNiazyCalculator,
    LegacySNRCalculator,
    MedianArtifactCalculator,
    MetricsReport,
    ReferenceIntervalSelector,
    RMSCalculator,
    RMSResidualCalculator,
    SignalIntervalSelector,
    SNRCalculator,
    SpectralCoherenceCalculator,
    SpikeDetectionRateCalculator,
)


@pytest.mark.unit
class TestReferenceIntervalSelector:
    """Tests for ReferenceIntervalSelector processor."""

    def test_reference_interval_selector_stores_interval(self, sample_context, monkeypatch):
        """Selected interval should be written to metadata.custom."""
        selector = ReferenceIntervalSelector(channel=0, min_duration=0.2)
        monkeypatch.setattr(
            selector,
            "_show_selector",
            lambda **kwargs: (0.5, 1.5),
        )

        result = selector.execute(sample_context)
        interval = result.metadata.custom.get("reference_interval", {})

        assert interval.get("tmin") == pytest.approx(0.5)
        assert interval.get("tmax") == pytest.approx(1.5)
        assert interval.get("source") == "reference_interval_selector"

    def test_reference_interval_selector_cancel_keeps_context(self, sample_context, monkeypatch):
        """Cancelling selection should keep metadata unchanged."""
        selector = ReferenceIntervalSelector()
        monkeypatch.setattr(selector, "_show_selector", lambda **kwargs: None)

        result = selector.execute(sample_context)
        assert "reference_interval" not in result.metadata.custom

    def test_reference_interval_selector_uses_explicit_bounds(self, sample_context, monkeypatch):
        """Explicit tmin/tmax should skip GUI and store configured interval."""
        selector = ReferenceIntervalSelector(channel=0, tmin=0.25, tmax=1.25)

        def _must_not_be_called(**kwargs):
            raise AssertionError("_show_selector should not be called when explicit bounds are provided")

        monkeypatch.setattr(selector, "_show_selector", _must_not_be_called)

        result = selector.execute(sample_context)
        interval = result.metadata.custom.get("reference_interval", {})

        assert interval.get("tmin") == pytest.approx(0.25)
        assert interval.get("tmax") == pytest.approx(1.25)

    def test_reference_interval_selector_rejects_invalid_bounds(self, sample_context):
        """Invalid explicit interval should raise validation error."""
        selector = ReferenceIntervalSelector(tmin=2.0, tmax=1.0)

        with pytest.raises(ProcessorValidationError):
            selector.execute(sample_context)


@pytest.mark.unit
class TestSignalIntervalSelector:
    """Tests for SignalIntervalSelector processor."""

    def test_signal_interval_selector_stores_interval(self, sample_context, monkeypatch):
        """Selected interval should be written to metadata.custom."""
        selector = SignalIntervalSelector(channel=0, min_duration=0.2)
        monkeypatch.setattr(
            selector,
            "_show_selector",
            lambda **kwargs: (1.0, 2.0),
        )

        result = selector.execute(sample_context)
        interval = result.metadata.custom.get("evaluation_interval", {})

        assert interval.get("tmin") == pytest.approx(1.0)
        assert interval.get("tmax") == pytest.approx(2.0)
        assert interval.get("source") == "signal_interval_selector"

    def test_signal_interval_selector_cancel_keeps_context(self, sample_context, monkeypatch):
        """Cancelling selection should keep metadata unchanged."""
        selector = SignalIntervalSelector()
        monkeypatch.setattr(selector, "_show_selector", lambda **kwargs: None)

        result = selector.execute(sample_context)
        assert "evaluation_interval" not in result.metadata.custom

    def test_signal_interval_selector_uses_explicit_bounds(self, sample_context, monkeypatch):
        """Explicit tmin/tmax should skip GUI and store configured interval."""
        selector = SignalIntervalSelector(channel=0, tmin=1.0, tmax=2.5)

        def _must_not_be_called(**kwargs):
            raise AssertionError("_show_selector should not be called when explicit bounds are provided")

        monkeypatch.setattr(selector, "_show_selector", _must_not_be_called)

        result = selector.execute(sample_context)
        interval = result.metadata.custom.get("evaluation_interval", {})

        assert interval.get("tmin") == pytest.approx(1.0)
        assert interval.get("tmax") == pytest.approx(2.5)

    def test_signal_interval_selector_rejects_invalid_bounds(self, sample_context):
        """Invalid explicit interval should raise validation error."""
        selector = SignalIntervalSelector(tmin=3.0, tmax=2.0)

        with pytest.raises(ProcessorValidationError):
            selector.execute(sample_context)

    def test_signal_interval_selector_requires_triggers(self, sample_raw):
        """Signal interval selection requires triggers."""
        selector = SignalIntervalSelector()
        context = ProcessingContext(raw=sample_raw)

        with pytest.raises(ProcessorValidationError):
            selector.execute(context)


@pytest.mark.unit
class TestSNRCalculator:
    """Tests for SNRCalculator processor."""

    def test_snr_calculation(self, sample_context):
        """Test SNR calculation."""
        snr_calc = SNRCalculator()
        result = snr_calc.execute(sample_context)

        # Check SNR was calculated
        metrics = result.metadata.custom.get("metrics", {})
        assert "snr" in metrics
        assert isinstance(metrics["snr"], float)
        assert metrics["snr"] > 0

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

        metrics = result.metadata.custom.get("metrics", {})
        assert "snr_per_channel" in metrics
        assert len(metrics["snr_per_channel"]) > 0

    def test_snr_uses_selected_reference_interval(self, sample_context):
        """User-selected reference interval should override automatic extraction."""
        metadata = sample_context.metadata.copy()
        metadata.custom["reference_interval"] = {"tmin": 0.0, "tmax": 1.0}
        context = sample_context.with_metadata(metadata)

        snr_calc = SNRCalculator()
        result = snr_calc.execute(context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "snr" in metrics

    def test_snr_uses_selected_evaluation_interval(self, sample_context):
        """User-selected evaluation interval should override automatic extraction."""
        metadata = sample_context.metadata.copy()
        metadata.custom["evaluation_interval"] = {"tmin": 1.0, "tmax": 2.0}
        context = sample_context.with_metadata(metadata)

        snr_calc = SNRCalculator()
        raw = context.get_raw()
        acquisition_data = snr_calc.get_acquisition_data(
            raw=raw,
            triggers=context.get_triggers(),
            artifact_length=context.get_artifact_length(),
            context=context,
        )

        eeg_picks = snr_calc.get_eeg_channels(raw)
        expected = raw.get_data(picks=eeg_picks, tmin=1.0, tmax=2.0)

        assert acquisition_data.shape == expected.shape
        np.testing.assert_allclose(acquisition_data, expected)


@pytest.mark.unit
class TestRMSCalculator:
    """Tests for RMSCalculator processor."""

    def test_rms_calculation(self, sample_context):
        """Test RMS calculation."""
        rms_calc = RMSCalculator()
        result = rms_calc.execute(sample_context)

        # Check RMS ratio was calculated
        metrics = result.metadata.custom.get("metrics", {})
        assert "rms_ratio" in metrics
        assert isinstance(metrics["rms_ratio"], float)
        assert metrics["rms_ratio"] > 0

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

        metrics = result.metadata.custom.get("metrics", {})
        assert "rms_ratio_per_channel" in metrics


@pytest.mark.unit
class TestRMSResidualCalculator:
    """Tests for RMSResidualCalculator processor."""

    def test_rms_residual_calculation(self, sample_context):
        """Test RMS residual calculation."""
        rms_calc = RMSResidualCalculator()
        result = rms_calc.execute(sample_context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "rms_residual" in metrics
        assert isinstance(metrics["rms_residual"], float)
        assert metrics["rms_residual"] > 0

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

        metrics = result.metadata.custom.get("metrics", {})
        assert "rms_residual_per_channel" in metrics
        assert len(metrics["rms_residual_per_channel"]) > 0


@pytest.mark.unit
class TestLegacySNRCalculator:
    """Tests for LegacySNRCalculator processor."""

    def test_legacy_snr_calculation(self, sample_context):
        """Test legacy SNR calculation."""
        snr_calc = LegacySNRCalculator()
        result = snr_calc.execute(sample_context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "legacy_snr" in metrics
        assert isinstance(metrics["legacy_snr"], float)
        assert metrics["legacy_snr"] > 0

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
        metrics = result.metadata.custom.get("metrics", {})
        assert "median_artifact" in metrics
        assert isinstance(metrics["median_artifact"], float)
        assert metrics["median_artifact"] >= 0

        # Check for reference ratio (might be missing if sample data is too short for reference extraction)
        # But the sample_context usually provides enough data
        if "median_artifact_reference" in metrics:
            assert "median_artifact_ratio" in metrics

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

        metrics = result.metadata.custom.get("metrics", {})
        assert "fft_allen" in metrics
        assert isinstance(metrics["fft_allen"], dict)

        # Check bands
        bands = ["Delta", "Theta", "Alpha", "Beta"]
        for band in bands:
            if band in metrics["fft_allen"]:
                assert isinstance(metrics["fft_allen"][band], float)
                assert metrics["fft_allen"][band] >= 0

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

        metrics = result.metadata.custom.get("metrics", {})
        assert "fft_niazy" in metrics
        assert "slice" in metrics["fft_niazy"]
        # volume might be missing if slices_per_volume is not set

    def test_fft_niazy_with_volume(self, sample_context):
        """Test FFT Niazy with volume information."""
        metadata = sample_context.metadata.copy()
        metadata.slices_per_volume = 10  # fake value
        context = sample_context.with_metadata(metadata)

        fft_calc = FFTNiazyCalculator()
        result = fft_calc.execute(context)

        metrics = result.metadata.custom.get("metrics", {})
        assert "volume" in metrics["fft_niazy"]


@pytest.mark.unit
def test_evaluation_processors_accept_verbose_mode(sample_context):
    """All main evaluation processors should run with verbose diagnostics enabled."""
    processors = [
        (SNRCalculator(verbose=True), "snr"),
        (RMSCalculator(verbose=True), "rms_ratio"),
        (RMSResidualCalculator(verbose=True), "rms_residual"),
        (MedianArtifactCalculator(verbose=True), "median_artifact"),
        (LegacySNRCalculator(verbose=True), "legacy_snr"),
        (FFTAllenCalculator(verbose=True), "fft_allen"),
        (FFTNiazyCalculator(verbose=True), "fft_niazy"),
    ]

    context = sample_context
    for proc, metric_key in processors:
        context = proc.execute(context)
        metrics = context.metadata.custom.get("metrics", {})
        assert metric_key in metrics


@pytest.mark.unit
class TestMetricsReport:
    """Tests for MetricsReport processor."""

    def test_metrics_report(self, sample_context, capsys):
        """Test metrics report generation."""
        # Add some metrics
        metadata = sample_context.metadata.copy()
        metadata.custom["metrics"] = {
            "snr": 15.5,
            "rms_ratio": 2.3,
            "median_artifact": 1.5e-5,
            "rms_residual": 1.1,
            "fft_allen": {"Alpha": 5.0},
            "fft_niazy": {"slice": {"h1": -10.0}},
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
        metadata.custom["metrics"] = {"snr": 15.5, "fft_allen": {"Alpha": 5.0}}
        context = sample_context.with_metadata(metadata)

        results = {}
        report = MetricsReport(store=results)
        report.execute(context)

        assert len(results) == 1
        key = list(results.keys())[0]
        assert results[key]["snr"] == 15.5
        assert results[key]["fft_allen_Alpha"] == 5.0


@pytest.mark.integration
class TestEvaluationPipeline:
    """Integration tests for evaluation pipeline."""

    def test_full_evaluation_pipeline(self, sample_context):
        """Test full evaluation workflow."""
        from facet.core import Pipeline

        pipeline = Pipeline(
            [
                SNRCalculator(),
                RMSCalculator(),
                RMSResidualCalculator(),
                MedianArtifactCalculator(),
                LegacySNRCalculator(),
                FFTAllenCalculator(),
                FFTNiazyCalculator(),
                MetricsReport(),
            ]
        )

        result = pipeline.run(initial_context=sample_context)

        assert result.success is True

        # Check all metrics were calculated
        metrics = result.context.metadata.custom.get("metrics", {})
        assert "snr" in metrics
        assert "rms_ratio" in metrics
        assert "rms_residual" in metrics
        assert "median_artifact" in metrics
        assert "legacy_snr" in metrics
        assert "fft_allen" in metrics
        assert "fft_niazy" in metrics

    def test_evaluation_after_correction(self, sample_edf_file):
        """Test evaluation after correction."""
        from facet.core import Pipeline
        from facet.correction import AASCorrection
        from facet.io import Loader
        from facet.preprocessing import DownSample, TriggerDetector, UpSample

        pipeline = Pipeline(
            [
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
                MetricsReport(),
            ]
        )

        result = pipeline.run()

        assert result.success is True

        # All metrics should be present
        metrics = result.context.metadata.custom.get("metrics", {})
        assert "snr" in metrics


# ---------------------------------------------------------------------------
# SpectralCoherenceCalculator
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSpectralCoherenceCalculator:
    """Tests for SpectralCoherenceCalculator processor."""

    def test_coherence_stored_in_metrics(self, sample_context):
        """Result should be stored under metrics['spectral_coherence']."""
        calc = SpectralCoherenceCalculator()
        result = calc.execute(sample_context)
        metrics = result.metadata.custom.get("metrics", {})
        assert "spectral_coherence" in metrics

    def test_coherence_bands_present(self, sample_context):
        """Default bands 'alpha', 'beta', and 'mean' must be present."""
        calc = SpectralCoherenceCalculator()
        result = calc.execute(sample_context)
        sc = result.metadata.custom["metrics"]["spectral_coherence"]
        for key in ("alpha", "beta", "mean"):
            assert key in sc, f"Missing band '{key}'"

    def test_coherence_values_in_range(self, sample_context):
        """Coherence values must be in [0, 1]."""
        calc = SpectralCoherenceCalculator()
        result = calc.execute(sample_context)
        sc = result.metadata.custom["metrics"]["spectral_coherence"]
        for key, val in sc.items():
            assert 0.0 <= val <= 1.0, f"Band '{key}': coherence={val} out of [0,1]"

    def test_identical_signals_yield_coherence_one(self, sample_context):
        """Identical corrected and original signal should give coherence ≈ 1.0."""
        calc = SpectralCoherenceCalculator()
        result = calc.execute(sample_context)
        # sample_context has raw_original = raw.copy(), so data is identical
        sc = result.metadata.custom["metrics"]["spectral_coherence"]
        assert sc["mean"] == pytest.approx(1.0, abs=0.05)

    def test_custom_bands(self, sample_context):
        """Custom band definitions should produce results for those bands only."""
        calc = SpectralCoherenceCalculator(bands={"gamma": (30.0, 50.0)})
        result = calc.execute(sample_context)
        sc = result.metadata.custom["metrics"]["spectral_coherence"]
        assert "gamma" in sc
        assert "alpha" not in sc

    def test_requires_triggers(self, sample_raw):
        """Missing triggers must raise ProcessorValidationError."""
        ctx = ProcessingContext(raw=sample_raw, raw_original=sample_raw.copy())
        calc = SpectralCoherenceCalculator()
        with pytest.raises(ProcessorValidationError):
            calc.execute(ctx)

    def test_mean_is_average_of_bands(self, sample_context):
        """'mean' must equal the arithmetic mean of the individual band values."""
        calc = SpectralCoherenceCalculator()
        result = calc.execute(sample_context)
        sc = result.metadata.custom["metrics"]["spectral_coherence"]
        band_values = [v for k, v in sc.items() if k != "mean"]
        expected_mean = float(np.mean(band_values))
        assert sc["mean"] == pytest.approx(expected_mean, abs=1e-9)

    def test_context_unchanged_when_no_eeg_channels(self, sample_context, monkeypatch):
        """Processor should return context unchanged when there are no EEG channels."""
        calc = SpectralCoherenceCalculator()
        monkeypatch.setattr(calc, "get_eeg_channels", lambda raw: [])
        result = calc.execute(sample_context)
        assert "spectral_coherence" not in result.metadata.custom.get("metrics", {})


# ---------------------------------------------------------------------------
# SpikeDetectionRateCalculator
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSpikeDetectionRateCalculator:
    """Tests for SpikeDetectionRateCalculator processor."""

    def test_spike_detection_stored_in_metrics(self, sample_context):
        """Result should be stored under metrics['spike_detection']."""
        calc = SpikeDetectionRateCalculator()
        result = calc.execute(sample_context)
        metrics = result.metadata.custom.get("metrics", {})
        assert "spike_detection" in metrics

    def test_spike_detection_keys_present(self, sample_context):
        """Result dict must contain the expected keys."""
        calc = SpikeDetectionRateCalculator()
        result = calc.execute(sample_context)
        sd = result.metadata.custom["metrics"]["spike_detection"]
        for key in ("spike_rate_original", "spike_rate_corrected", "spike_suppression_ratio", "threshold_std"):
            assert key in sd, f"Missing key '{key}'"

    def test_spike_rates_non_negative(self, sample_context):
        """Spike rates must be non-negative."""
        calc = SpikeDetectionRateCalculator()
        result = calc.execute(sample_context)
        sd = result.metadata.custom["metrics"]["spike_detection"]
        assert sd["spike_rate_original"] >= 0.0
        assert sd["spike_rate_corrected"] >= 0.0

    def test_identical_signals_suppression_ratio_one(self, sample_context):
        """Identical signals should yield suppression ratio ≈ 1.0."""
        calc = SpikeDetectionRateCalculator()
        result = calc.execute(sample_context)
        sd = result.metadata.custom["metrics"]["spike_detection"]
        ratio = sd["spike_suppression_ratio"]
        if ratio is not None:
            assert ratio == pytest.approx(1.0, abs=0.1)

    def test_corrected_signal_with_fewer_spikes_raises_ratio(self, sample_raw, sample_triggers):
        """Injecting spikes into original but not corrected should produce ratio > 1."""
        from facet.core import ProcessingMetadata

        rng = np.random.default_rng(7)
        n_ch, n_samples = sample_raw._data.shape

        data_corr = rng.standard_normal((n_ch, n_samples)) * 1e-7
        data_orig = data_corr.copy()
        for t in sample_triggers:
            if t < n_samples:
                data_orig[:, t] += 500e-6   # large spike only in original

        ch_names = [f"EEG{i:02d}" for i in range(n_ch)]
        info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types=["eeg"] * n_ch)
        raw_corr = mne.io.RawArray(data_corr, info, verbose=False)
        raw_orig = mne.io.RawArray(data_orig, info, verbose=False)

        metadata = ProcessingMetadata()
        metadata.triggers = sample_triggers
        metadata.artifact_length = 50
        ctx = ProcessingContext(raw=raw_corr, raw_original=raw_orig, metadata=metadata)

        calc = SpikeDetectionRateCalculator(threshold_std=3.0)
        result = calc.execute(ctx)
        sd = result.metadata.custom["metrics"]["spike_detection"]

        assert sd["spike_rate_original"] > sd["spike_rate_corrected"]
        ratio = sd["spike_suppression_ratio"]
        assert ratio is not None and ratio > 1.0

    def test_requires_triggers(self, sample_raw):
        """Missing triggers must raise ProcessorValidationError."""
        ctx = ProcessingContext(raw=sample_raw, raw_original=sample_raw.copy())
        calc = SpikeDetectionRateCalculator()
        with pytest.raises(ProcessorValidationError):
            calc.execute(ctx)

    def test_threshold_std_stored(self, sample_context):
        """threshold_std parameter must be echoed into the result dict."""
        calc = SpikeDetectionRateCalculator(threshold_std=7.0)
        result = calc.execute(sample_context)
        sd = result.metadata.custom["metrics"]["spike_detection"]
        assert sd["threshold_std"] == pytest.approx(7.0)
