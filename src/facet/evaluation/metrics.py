"""
Evaluation Metrics Processors Module

This module contains processors for evaluating correction quality.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, Dict, Any, List, Tuple, Union
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from scipy import signal

from ..console import report_metric
from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError


class ReferenceDataMixin:
    """Mixin for extracting reference data (outside acquisition)."""

    def get_eeg_channels(self, raw: mne.io.BaseRaw) -> List[str]:
        """Get EEG channel names."""
        return mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )

    def get_reference_data(
        self, 
        raw: mne.io.BaseRaw, 
        triggers: np.ndarray, 
        artifact_length: int, 
        time_buffer: float = 0.1
    ) -> np.ndarray:
        """
        Extract reference data (outside acquisition window).
        
        Args:
            raw: Raw data object
            triggers: Trigger indices
            artifact_length: Length of one artifact in samples
            time_buffer: Buffer in seconds to stay away from acquisition
            
        Returns:
            numpy array of shape (n_channels, n_times) containing concatenated reference data
        """
        sfreq = raw.info['sfreq']
        eeg_picks = self.get_eeg_channels(raw)
        
        if len(eeg_picks) == 0:
            logger.warning("No EEG channels found")
            return np.array([])

        # Define acquisition start and end based on triggers
        # Triggers are typically slice triggers. 
        # Acquisition is considered to be from first trigger to last trigger + artifact
        acq_start_sample = triggers[0]
        acq_end_sample = triggers[-1] + artifact_length
        
        # Convert to time
        acq_start_time = acq_start_sample / sfreq
        acq_end_time = acq_end_sample / sfreq
        
        # Define reference windows (safely outside acquisition)
        ref_pre_tmax = acq_start_time - time_buffer
        ref_post_tmin = acq_end_time + time_buffer
        
        ref_segments = []
        
        # Pre-acquisition data
        if ref_pre_tmax > 0:
            # Ensure we don't go below 0
            try:
                data = raw.copy().crop(tmax=ref_pre_tmax).get_data(picks=eeg_picks)
                ref_segments.append(data)
            except Exception:
                pass

        # Post-acquisition data
        if ref_post_tmin < raw.times[-1]:
            try:
                data = raw.copy().crop(tmin=ref_post_tmin).get_data(picks=eeg_picks)
                ref_segments.append(data)
            except Exception:
                pass
                
        if not ref_segments:
            logger.warning("No reference data found outside acquisition window")
            return np.array([]).reshape(len(eeg_picks), 0)
            
        return np.concatenate(ref_segments, axis=1)

    def get_acquisition_data(
        self,
        raw: mne.io.BaseRaw,
        triggers: np.ndarray,
        artifact_length: int
    ) -> np.ndarray:
        """
        Extract data within the acquisition window.
        """
        sfreq = raw.info['sfreq']
        eeg_picks = self.get_eeg_channels(raw)
        
        if len(eeg_picks) == 0:
            return np.array([])
            
        # Define acquisition window with small margin to cover full artifacts
        acq_start = max(0, triggers[0] - int(artifact_length * 0.5))
        acq_end = min(raw.n_times, triggers[-1] + int(artifact_length * 1.5))
        
        tmin = acq_start / sfreq
        tmax = min(acq_end / sfreq, raw.times[-1])
        
        return raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=eeg_picks)


@register_processor
class SNRCalculator(Processor, ReferenceDataMixin):
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
        
        # Get data using Mixin
        ref_data = self.get_reference_data(raw, triggers, artifact_length, self.time_buffer)
        corrected_data = self.get_acquisition_data(raw, triggers, artifact_length)
        
        if ref_data.size == 0 or corrected_data.size == 0:
            logger.warning("Insufficient data for SNR calculation")
            return context

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
class RMSResidualCalculator(Processor, ReferenceDataMixin):
    """
    Calculate RMS Residual Ratio (Corrected vs. Reference).

    Compares the RMS of the corrected signal (during acquisition) to the RMS of the 
    clean reference signal (outside acquisition).

    Ratio = RMS(corrected) / RMS(reference)

    Target value is 1.0.
    < 1.0: Signal might be over-corrected (dampened).
    > 1.0: Signal might still contain artifacts.

    Corresponds to `rms_residual` in FACET MATLAB Edition.
    """

    name = "rms_residual_calculator"
    description = "Calculate RMS ratio (corrected vs reference)"
    requires_triggers = True
    parallel_safe = False

    def __init__(self, time_buffer: float = 0.1):
        self.time_buffer = time_buffer
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)

    def process(self, context: ProcessingContext) -> ProcessingContext:
        logger.info("Calculating RMS Residual Ratio (corrected vs reference)")

        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()

        ref_data = self.get_reference_data(raw, triggers, artifact_length, self.time_buffer)
        corrected_data = self.get_acquisition_data(raw, triggers, artifact_length)

        if ref_data.size == 0 or corrected_data.size == 0:
            logger.warning("Insufficient data for RMS Residual calculation")
            return context

        # Calculate RMS
        rms_corrected = np.std(corrected_data, axis=1)
        rms_reference = np.std(ref_data, axis=1)

        # Prevent division by zero
        rms_reference = np.maximum(rms_reference, 1e-10)
        
        # Ratio: corrected / reference
        ratio_per_channel = rms_corrected / rms_reference
        
        # Aggregate
        ratio_mean = np.mean(ratio_per_channel)

        report_metric("rms_residual", float(ratio_mean), label="RMS Residual", display=f"{ratio_mean:.2f}")

        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault('metrics', {})
        metrics['rms_residual'] = float(ratio_mean)
        metrics['rms_residual_per_channel'] = ratio_per_channel.tolist()

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
        if context.get_raw_original() is context.get_raw():
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
        acq_tmax = min(acq_end / sfreq, raw_corrected.times[-1])

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
        acq_tmax = min(acq_end / sfreq, raw.times[-1])

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
class MedianArtifactCalculator(Processor, ReferenceDataMixin):
    """
    Calculate median peak-to-peak artifact amplitude.

    Measures the median artifact amplitude across all epochs.
    Lower values indicate smaller artifacts (better correction).
    
    Also calculates the ratio to the median amplitude of the reference signal
    (outside acquisition), which should ideally be close to 1.0.

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
        artifact_len = context.get_artifact_length()

        # Get EEG channels
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found")
            return context

        # 1. Calculate for Corrected Data (Acquisition)
        # Create epochs
        tmin = context.metadata.artifact_to_trigger_offset
        tmax = tmin + (artifact_len / sfreq)

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

        # 2. Calculate for Reference Data (Outside Acquisition)
        # Simulate epochs on reference data to get comparable stats
        ref_data = self.get_reference_data(raw, triggers, artifact_len)
        
        median_ref = np.nan
        ratio = np.nan
        
        if ref_data.size > 0:
            n_samples_ref = ref_data.shape[1]
            # How many 'epochs' fit into the reference data?
            epoch_len = int(artifact_len)
            n_ref_epochs = n_samples_ref // epoch_len
            
            if n_ref_epochs > 0:
                # Reshape reference data into mock epochs
                # Take only full epochs
                ref_data_truncated = ref_data[:, :n_ref_epochs * epoch_len]
                # Shape: (channels, n_epochs, samples) -> (n_epochs, channels, samples)
                ref_epochs = ref_data_truncated.reshape(len(eeg_channels), n_ref_epochs, epoch_len)
                ref_epochs = np.moveaxis(ref_epochs, 1, 0)
                
                p2p_ref = [np.ptp(epoch, axis=1) for epoch in ref_epochs]
                mean_p2p_ref = [np.mean(epoch_p2p) for epoch_p2p in p2p_ref]
                median_ref = np.median(mean_p2p_ref)
                
                if median_ref > 0:
                    ratio = median_artifact / median_ref

        report_metric(
            "median_artifact",
            float(median_artifact),
            label="Median Artifact",
            display=f"{median_artifact:.2e}",
        )
        
        if not np.isnan(ratio):
             report_metric(
                "median_artifact_ratio",
                float(ratio),
                label="Median Ratio (Corr/Ref)",
                display=f"{ratio:.2f}",
            )

        # Store in metadata
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault('metrics', {})
        metrics['median_artifact'] = float(median_artifact)
        if not np.isnan(median_ref):
            metrics['median_artifact_reference'] = float(median_ref)
        if not np.isnan(ratio):
            metrics['median_artifact_ratio'] = float(ratio)

        return context.with_metadata(new_metadata)


@register_processor
class FFTAllenCalculator(Processor, ReferenceDataMixin):
    """
    Calculate FFT Allen metric.

    Compares spectral power in specific frequency bands between corrected data 
    and clean reference data.

    Bands: 0.8-4, 4-8, 8-12, 12-24 Hz.
    Metric: Median absolute percent difference.
    
    (abs(Power_corr - Power_ref) / Power_ref) * 100
    """
    
    name = "fft_allen_calculator"
    description = "Calculate spectral power difference (Allen)"
    requires_triggers = True
    parallel_safe = False
    
    BANDS = [
        (0.8, 4, "Delta"),
        (4, 8, "Theta"),
        (8, 12, "Alpha"),
        (12, 24, "Beta")
    ]

    def process(self, context: ProcessingContext) -> ProcessingContext:
        logger.info("Calculating FFT Allen metric")
        
        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_len = context.get_artifact_length()
        sfreq = raw.info['sfreq']
        
        # Use 3-second segments as in MATLAB
        n_fft = int(3.0 * sfreq)
        
        # Get data
        ref_data = self.get_reference_data(raw, triggers, artifact_len)
        corr_data = self.get_acquisition_data(raw, triggers, artifact_len)
        
        if ref_data.size == 0 or corr_data.size == 0:
            logger.warning("Insufficient data for FFT Allen")
            return context
            
        # Use the same nperseg for both to ensure matching frequency resolution
        nperseg = min(n_fft, ref_data.shape[1], corr_data.shape[1])
        
        def calc_psd(data):
            freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg, axis=1)
            return freqs, psd

        freqs_ref, psd_ref = calc_psd(ref_data)
        freqs_corr, psd_corr = calc_psd(corr_data)
        
        # Ensure freqs match (they should since we use the same nperseg)
        if not np.array_equal(freqs_ref, freqs_corr):
            logger.warning("Frequency mismatch in FFT Allen")
            return context
            
        results = {}
        
        for fmin, fmax, band_name in self.BANDS:
            idx = np.logical_and(freqs_ref >= fmin, freqs_ref <= fmax)
            
            # Mean power in band per channel
            power_ref = np.mean(psd_ref[:, idx], axis=1)
            power_corr = np.mean(psd_corr[:, idx], axis=1)
            
            # Percent difference
            diff_pct = np.abs(power_corr - power_ref) / (power_ref + 1e-10) * 100
            
            # Median across channels
            median_diff = np.median(diff_pct)
            
            results[band_name] = float(median_diff)
            
            logger.info(f"FFT Allen {band_name} ({fmin}-{fmax}Hz): {median_diff:.2f}%")
            
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault('metrics', {})
        metrics['fft_allen'] = results
        
        return context.with_metadata(new_metadata)


@register_processor
class FFTNiazyCalculator(Processor, ReferenceDataMixin):
    """
    Calculate FFT Niazy metric.
    
    Analyzes residual artifacts at slice and volume frequencies.
    Calculates ratio of spectral power (uncorrected / corrected) at these frequencies.
    Values are in dB.
    """
    
    name = "fft_niazy_calculator"
    description = "Calculate spectral power ratio at slice/volume frequencies"
    requires_triggers = True
    parallel_safe = False

    def process(self, context: ProcessingContext) -> ProcessingContext:
        logger.info("Calculating FFT Niazy metric")
        
        raw = context.get_raw()
        raw_orig = context.get_raw_original()
        triggers = context.get_triggers()
        artifact_len = context.get_artifact_length()
        sfreq = raw.info['sfreq']
        
        if raw_orig is None:
            logger.warning("Original raw data missing for FFT Niazy")
            return context

        # Calculate Slice Frequency
        # Mean distance between triggers
        trigger_diffs = np.diff(triggers)
        mean_diff = np.mean(trigger_diffs)
        slice_freq = sfreq / mean_diff
        
        # Calculate Volume Frequency
        # Try to find slices_per_volume in metadata
        slices_per_vol = getattr(context.metadata, 'slices_per_volume', None)
        if not slices_per_vol:
             # Try custom
             slices_per_vol = context.metadata.custom.get('slices_per_volume')
             
        if slices_per_vol:
            vol_freq = slice_freq / slices_per_vol
        else:
            vol_freq = None
            logger.warning("Slices per volume not found, skipping volume frequency analysis")
            
        # Frequencies to analyze (harmonics)
        harmonics = 5
        
        # Get data in acquisition window
        data_corr = self.get_acquisition_data(raw, triggers, artifact_len)
        data_orig = self.get_acquisition_data(raw_orig, triggers, artifact_len)
        
        # Handle channels
        min_ch = min(data_corr.shape[0], data_orig.shape[0])
        data_corr = data_corr[:min_ch]
        data_orig = data_orig[:min_ch]
        
        # PSD
        nperseg = min(int(4 * sfreq), data_corr.shape[1]) # 4 seconds window
        freqs, psd_corr = signal.welch(data_corr, fs=sfreq, nperseg=nperseg, axis=1)
        _, psd_orig = signal.welch(data_orig, fs=sfreq, nperseg=nperseg, axis=1)
        
        def get_power_at_freq(f, tolerance=0.5):
            idx = np.logical_and(freqs >= f - tolerance, freqs <= f + tolerance)
            if not np.any(idx):
                return np.nan, np.nan
            p_corr = np.mean(psd_corr[:, idx], axis=1)
            p_orig = np.mean(psd_orig[:, idx], axis=1)
            # Return median across channels
            return np.median(p_corr), np.median(p_orig)

        results = {'slice': {}, 'volume': {}}
        
        # Analyze Slice Frequencies
        for h in range(1, harmonics + 1):
            f = slice_freq * h
            pc, po = get_power_at_freq(f)
            if not np.isnan(pc) and pc > 0:
                ratio = po / pc
                ratio_db = 10 * np.log10(ratio)
                results['slice'][f"h{h}"] = float(ratio_db)
                
        # Analyze Volume Frequencies
        if vol_freq:
            for h in range(1, harmonics + 1):
                f = vol_freq * h
                pc, po = get_power_at_freq(f)
                if not np.isnan(pc) and pc > 0:
                    ratio = po / pc
                    ratio_db = 10 * np.log10(ratio)
                    results['volume'][f"h{h}"] = float(ratio_db)

        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault('metrics', {})
        metrics['fft_niazy'] = results
        
        logger.info(f"FFT Niazy Slice Fundamental: {results['slice'].get('h1', 'N/A'):.2f} dB")
        
        return context.with_metadata(new_metadata)


@register_processor
class MetricsReport(Processor):
    """
    Generate a summary report of all calculated metrics.

    Collects all metrics from context and logs/stores a summary.
    Can also store results in a shared dictionary for comparison and plotting.

    Example:
        # Basic usage
        report = MetricsReport()
        context = report.execute(context)

        # Advanced usage (collecting results for comparison)
        results = {}
        report = MetricsReport(name="Pipeline A", store=results)
        context = report.execute(context)
        
        # ... run another pipeline ...
        report2 = MetricsReport(name="Pipeline B", store=results)
        context2 = report2.execute(context2)
        
        # Plot comparison
        MetricsReport.plot(results)
        
        # Plot specific metrics
        MetricsReport.plot(results, metrics=['snr', 'rms_ratio'])
    """

    name = "metrics_report"
    description = "Generate metrics summary report"
    requires_triggers = False
    parallel_safe = False

    def __init__(self, name: Optional[str] = None, store: Optional[Dict] = None):
        """
        Initialize MetricsReport.

        Args:
            name: Name of the result set (e.g., "Pipeline A").
                  If None, a default name will be generated during processing.
            store: Dictionary to store results in. Structure: {name: {metric: value}}
        """
        self.report_name = name
        self.store = store
        super().__init__()

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
            
        if 'rms_residual' in metrics:
            logger.info(f"RMS Residual Ratio (ref match):  {metrics['rms_residual']:.2f} (target: 1.0)")

        if 'median_artifact' in metrics:
            logger.info(f"Median Artifact Amplitude:       {metrics['median_artifact']:.2e}")
            if 'median_artifact_ratio' in metrics:
                logger.info(f"Median Artifact Ratio (to ref):  {metrics['median_artifact_ratio']:.2f} (target: 1.0)")

        if 'legacy_snr' in metrics:
             logger.info(f"Legacy SNR:                      {metrics['legacy_snr']:.2f}")

        if 'fft_allen' in metrics:
            logger.info("FFT Allen (Diff to Ref):")
            for band, val in metrics['fft_allen'].items():
                logger.info(f"  - {band}: {val:.2f}%")

        if 'fft_niazy' in metrics:
            logger.info("FFT Niazy (Power Ratio Uncorr/Corr dB):")
            if 'slice' in metrics['fft_niazy']:
                logger.info("  Slice Harmonics:")
                for k, v in metrics['fft_niazy']['slice'].items():
                    logger.info(f"    - {k}: {v:.2f} dB")
            if 'volume' in metrics['fft_niazy']:
                logger.info("  Volume Harmonics:")
                for k, v in metrics['fft_niazy']['volume'].items():
                    logger.info(f"    - {k}: {v:.2f} dB")

        logger.info("=" * 60)

        # Store results if configured
        if self.store is not None:
            # Determine name
            name = self.report_name
            if name is None:
                # Use existing count to generate unique name
                name = f"Result_{len(self.store) + 1}"
            
            # Filter for scalar values only (flatten nested dicts if needed for plotting)
            scalar_metrics = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.number)):
                    scalar_metrics[k] = float(v)
                elif k == 'fft_allen' and isinstance(v, dict):
                    for band, val in v.items():
                        scalar_metrics[f"fft_allen_{band}"] = float(val)
                elif k == 'fft_niazy' and isinstance(v, dict):
                    if 'slice' in v:
                         scalar_metrics[f"fft_niazy_slice_h1"] = float(v['slice'].get('h1', 0))
            
            self.store[name] = scalar_metrics
            logger.info(f"Stored metrics for '{name}'")

        return context

    @staticmethod
    def plot(results: Dict[str, Dict[str, float]], 
             title: str = "Metrics Comparison", 
             save_path: Optional[str] = None, 
             show: bool = True,
             metrics: Optional[List[str]] = None):
        """
        Plot comparison of metrics using Matplotlib.

        Args:
            results: Dictionary of results {name: {metric: value}}
            title: Plot title
            save_path: Path to save figure
            show: Whether to show the plot
            metrics: Optional list of metric keys to plot. If None, plots all.
        """
        if not results:
            logger.warning("No results to plot")
            return

        # Convert to DataFrame for easier handling
        df = pd.DataFrame.from_dict(results, orient='index')
        
        if df.empty:
            logger.warning("Results DataFrame is empty")
            return
            
        if metrics:
            # Filter columns if metrics are specified
            existing_metrics = [m for m in metrics if m in df.columns]
            if not existing_metrics:
                logger.warning(f"None of the requested metrics {metrics} found in results.")
                return
            df = df[existing_metrics]

        metrics_list = df.columns.tolist()
        n_metrics = len(metrics_list)
        
        if n_metrics == 0:
            logger.warning("No metrics found in results")
            return

        # Setup plot
        # Dynamic layout
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
        axes = axes.flatten()
        
        # Color palette
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0.2, 0.8, len(df))]

        for i, metric in enumerate(metrics_list):
            ax = axes[i]
            # Plot bars for this metric
            values = df[metric]
            
            # Plot
            values.plot(kind='bar', ax=ax, color=colors, rot=45)
            
            ax.set_title(metric)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Annotate values
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f"{height:.2g}", 
                           (p.get_x() + p.get_width() / 2., height), 
                           ha='center', va='bottom', 
                           fontsize=8,
                           rotation=0)
        
        # Hide unused axes
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Saved metrics plot to {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
