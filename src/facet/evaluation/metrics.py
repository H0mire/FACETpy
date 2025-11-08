"""
Evaluation Metrics Processors Module

This module contains processors for evaluating correction quality.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, Dict, Any
import mne
import numpy as np
from loguru import logger

from ..console import report_metric
from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError


@register_processor
class SNRCalculator(Processor):
    """
    Calculate Signal-to-Noise Ratio (SNR).

    Compares corrected data to a clean reference (data outside acquisition window).
    Higher SNR indicates better correction.

    SNR = variance(reference) / variance(residual)

    Example:
        snr = SNRCalculator()
        context = snr.execute(context)
        print(context.metadata.custom['snr'])
    """

    name = "snr_calculator"
    description = "Calculate Signal-to-Noise Ratio"
    requires_triggers = True
    parallel_safe = False

    def __init__(self, time_buffer: float = 0.1):
        """
        Initialize SNR calculator.

        Args:
            time_buffer: Time buffer around acquisition window (seconds)
        """
        self.time_buffer = time_buffer
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate prerequisites."""
        super().validate(context)
        if context.get_raw_original() is None:
            raise ProcessorValidationError(
                "Original raw data not available. Cannot calculate SNR."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Calculate SNR."""
        logger.info("Calculating Signal-to-Noise Ratio (SNR)")

        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info['sfreq']

        # Get EEG channels
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found")
            return context

        # Define acquisition and reference windows
        acq_start = max(0, triggers[0] - int(artifact_length * 0.5))
        acq_end = min(raw.n_times, triggers[-1] + int(artifact_length * 1.5))

        # Create time windows
        acq_tmin = acq_start / sfreq
        acq_tmax = acq_end / sfreq

        # Reference: data outside acquisition (with buffer)
        ref_tmax = acq_tmin - self.time_buffer
        ref2_tmin = acq_tmax + self.time_buffer

        # Get corrected and reference data
        if ref_tmax > 0:
            ref_data1 = raw.copy().crop(tmax=ref_tmax).get_data(picks=eeg_channels)
        else:
            ref_data1 = np.array([]).reshape(len(eeg_channels), 0)

        if ref2_tmin < raw.times[-1]:
            ref_data2 = raw.copy().crop(tmin=ref2_tmin).get_data(picks=eeg_channels)
        else:
            ref_data2 = np.array([]).reshape(len(eeg_channels), 0)

        # Combine reference data
        if ref_data1.size > 0 and ref_data2.size > 0:
            ref_data = np.concatenate([ref_data1, ref_data2], axis=1)
        elif ref_data1.size > 0:
            ref_data = ref_data1
        elif ref_data2.size > 0:
            ref_data = ref_data2
        else:
            logger.warning("No reference data available for SNR calculation")
            return context

        # Get corrected data in acquisition window
        corrected_data = raw.copy().crop(tmin=acq_tmin, tmax=acq_tmax).get_data(picks=eeg_channels)

        # Calculate variances
        var_corrected = np.var(corrected_data, axis=1)
        var_reference = np.var(ref_data, axis=1)

        # Calculate residual variance (assuming reference is clean)
        var_residual = var_corrected - var_reference

        # Prevent division by zero
        var_residual = np.maximum(var_residual, 1e-10)

        # Calculate SNR per channel
        snr_per_channel = np.abs(var_reference / var_residual)

        # Average SNR across channels
        snr_mean = np.mean(snr_per_channel)

        report_metric("snr", float(snr_mean), label="SNR", display=f"{snr_mean:.2f}")

        # Store in metadata
        new_metadata = context.metadata.copy()
        if 'metrics' not in new_metadata.custom:
            new_metadata.custom['metrics'] = {}
        new_metadata.custom['metrics']['snr'] = float(snr_mean)
        new_metadata.custom['metrics']['snr_per_channel'] = snr_per_channel.tolist()

        return context.with_metadata(new_metadata)


@register_processor
class LegacySNRCalculator(Processor):
    """
    Calculate legacy-style Signal-to-Noise Ratio (SNR).

    Mirrors the original FACET implementation by comparing the variance of the
    corrected data to the variance of the uncorrected reference recording.
    """

    name = "legacy_snr_calculator"
    description = "Legacy-style SNR using original raw as reference"
    requires_triggers = False
    parallel_safe = False

    def __init__(self):
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate prerequisites."""
        super().validate(context)
        if context.get_raw_original() is None:
            raise ProcessorValidationError(
                "Original raw data not available. Cannot calculate legacy SNR."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Calculate legacy SNR."""
        logger.info("Calculating legacy SNR (corrected vs original)")

        raw_corrected = context.get_raw()
        raw_original = context.get_raw_original()

        # Use EEG channels only
        picks = mne.pick_types(
            raw_corrected.info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            exclude='bads'
        )

        if len(picks) == 0:
            logger.warning("No EEG channels found for legacy SNR calculation")
            return context

        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        if triggers is None or len(triggers) == 0 or artifact_length is None:
            logger.warning("Legacy SNR requires triggers and artifact length; skipping")
            return context

        sfreq = raw_corrected.info['sfreq']
        acq_start = max(0, triggers[0] - int(artifact_length * 0.5))
        acq_end = min(raw_corrected.n_times, triggers[-1] + int(artifact_length * 1.5))

        acq_tmin = acq_start / sfreq
        acq_tmax = acq_end / sfreq

        corrected_data = raw_corrected.copy().crop(tmin=acq_tmin, tmax=acq_tmax).get_data(picks=picks)

        ref_segments = []
        if acq_tmin > 0:
            ref_segments.append(
                raw_original.copy().crop(tmax=acq_tmin).get_data(picks=picks)
            )
        if acq_tmax < raw_original.times[-1]:
            ref_segments.append(
                raw_original.copy().crop(tmin=acq_tmax).get_data(picks=picks)
            )

        if ref_segments:
            reference_data = np.concatenate(ref_segments, axis=1)
        else:
            reference_data = raw_original.get_data(picks=picks)

        var_corrected = np.var(corrected_data, axis=1)
        var_reference = np.var(reference_data, axis=1)

        var_residual = var_corrected - var_reference
        var_residual = np.maximum(var_residual, 1e-10)

        snr_per_channel = np.abs(var_reference / var_residual)
        snr_mean = float(np.mean(snr_per_channel))

        report_metric(
            "legacy_snr",
            snr_mean,
            label="Legacy SNR",
            display=f"{snr_mean:.2f}",
        )

        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault('metrics', {})
        metrics['legacy_snr'] = snr_mean
        metrics['legacy_snr_per_channel'] = snr_per_channel.tolist()

        return context.with_metadata(new_metadata)


@register_processor
class RMSCalculator(Processor):
    """
    Calculate Root Mean Square (RMS) improvement ratio.

    Compares RMS of corrected data to uncorrected data.
    Higher ratio indicates better correction (more artifact removed).

    RMS_ratio = RMS(uncorrected) / RMS(corrected)

    Example:
        rms = RMSCalculator()
        context = rms.execute(context)
        print(context.metadata.custom['rms_ratio'])
    """

    name = "rms_calculator"
    description = "Calculate RMS improvement ratio"
    requires_triggers = True
    parallel_safe = False

    def validate(self, context: ProcessingContext) -> None:
        """Validate prerequisites."""
        super().validate(context)
        if context.get_raw_original() is None:
            raise ProcessorValidationError(
                "Original raw data not available. Cannot calculate RMS ratio."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Calculate RMS ratio."""
        logger.info("Calculating RMS improvement ratio")

        raw = context.get_raw()
        raw_orig = context.get_raw_original()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info['sfreq']

        # Get EEG channels
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found")
            return context

        # Define acquisition window
        acq_start = max(0, triggers[0] - int(artifact_length * 0.5))
        acq_end = min(raw.n_times, triggers[-1] + int(artifact_length * 1.5))

        acq_tmin = acq_start / sfreq
        acq_tmax = acq_end / sfreq

        # Get data
        data_corrected = raw.copy().crop(tmin=acq_tmin, tmax=acq_tmax).get_data(picks=eeg_channels)
        data_uncorrected = raw_orig.copy().crop(tmin=acq_tmin, tmax=acq_tmax).get_data(picks=eeg_channels)

        # Handle different channel counts
        if data_corrected.shape[0] != data_uncorrected.shape[0]:
            min_channels = min(data_corrected.shape[0], data_uncorrected.shape[0])
            data_corrected = data_corrected[:min_channels]
            data_uncorrected = data_uncorrected[:min_channels]

        # Calculate RMS
        rms_corrected = np.sqrt(np.mean(data_corrected ** 2, axis=1))
        rms_uncorrected = np.sqrt(np.mean(data_uncorrected ** 2, axis=1))

        # Calculate ratio (prevent division by zero)
        rms_corrected = np.maximum(rms_corrected, 1e-10)
        rms_ratio_per_channel = rms_uncorrected / rms_corrected

        # Median across channels
        rms_ratio = np.median(rms_ratio_per_channel)

        report_metric("rms_ratio", float(rms_ratio), label="RMS Ratio", display=f"{rms_ratio:.2f}")

        # Store in metadata
        new_metadata = context.metadata.copy()
        if 'metrics' not in new_metadata.custom:
            new_metadata.custom['metrics'] = {}
        new_metadata.custom['metrics']['rms_ratio'] = float(rms_ratio)
        new_metadata.custom['metrics']['rms_ratio_per_channel'] = rms_ratio_per_channel.tolist()

        return context.with_metadata(new_metadata)


@register_processor
class MedianArtifactCalculator(Processor):
    """
    Calculate median peak-to-peak artifact amplitude.

    Measures the median artifact amplitude across all epochs.
    Lower values indicate smaller artifacts (better correction).

    Example:
        median = MedianArtifactCalculator()
        context = median.execute(context)
        print(context.metadata.custom['median_artifact'])
    """

    name = "median_artifact_calculator"
    description = "Calculate median artifact amplitude"
    requires_triggers = True
    parallel_safe = False

    def validate(self, context: ProcessingContext) -> None:
        """Validate prerequisites."""
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector first."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Calculate median artifact amplitude."""
        logger.info("Calculating median artifact amplitude")

        raw = context.get_raw()
        triggers = context.get_triggers()
        sfreq = raw.info['sfreq']

        # Get EEG channels
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found")
            return context

        # Create epochs
        tmin = context.metadata.artifact_to_trigger_offset
        tmax = tmin + (context.get_artifact_length() / sfreq)

        events = np.column_stack([
            triggers,
            np.zeros(len(triggers), dtype=int),
            np.ones(len(triggers), dtype=int)
        ])

        epochs = mne.Epochs(
            raw,
            events=events,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            reject=None,
            preload=True,
            picks=eeg_channels,
            verbose=False
        )

        # Calculate peak-to-peak per epoch and channel
        p2p_per_epoch = [np.ptp(epoch, axis=1) for epoch in epochs.get_data(copy=False)]

        # Mean across channels for each epoch
        mean_p2p_per_epoch = [np.mean(epoch_p2p) for epoch_p2p in p2p_per_epoch]

        # Median across epochs
        median_artifact = np.median(mean_p2p_per_epoch)

        report_metric(
            "median_artifact",
            float(median_artifact),
            label="Median Artifact",
            display=f"{median_artifact:.2e}",
        )

        # Store in metadata
        new_metadata = context.metadata.copy()
        if 'metrics' not in new_metadata.custom:
            new_metadata.custom['metrics'] = {}
        new_metadata.custom['metrics']['median_artifact'] = float(median_artifact)

        return context.with_metadata(new_metadata)


@register_processor
class MetricsReport(Processor):
    """
    Generate a summary report of all calculated metrics.

    Collects all metrics from context and logs/stores a summary.

    Example:
        report = MetricsReport()
        context = report.execute(context)
    """

    name = "metrics_report"
    description = "Generate metrics summary report"
    requires_triggers = False
    parallel_safe = False

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Generate metrics report."""
        logger.info("=" * 60)
        logger.info("EVALUATION METRICS REPORT")
        logger.info("=" * 60)

        metrics = context.metadata.custom.get('metrics', {})

        if not metrics:
            logger.warning("No metrics available")
            return context

        # Print each metric
        if 'snr' in metrics:
            logger.info(f"SNR (Signal-to-Noise Ratio):     {metrics['snr']:.2f}")

        if 'rms_ratio' in metrics:
            logger.info(f"RMS Ratio (improvement):         {metrics['rms_ratio']:.2f}")

        if 'median_artifact' in metrics:
            logger.info(f"Median Artifact Amplitude:       {metrics['median_artifact']:.2e}")

        logger.info("=" * 60)

        return context
