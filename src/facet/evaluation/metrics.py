"""
Evaluation Metrics Processors Module

This module contains processors for evaluating correction quality.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, Dict, Any, List, Union
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

    def get_eeg_channels(self, raw: mne.io.BaseRaw) -> np.ndarray:
        """Get EEG channel indices.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data object.

        Returns
        -------
        np.ndarray
            Array of EEG channel indices.
        """
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
        """Extract reference data (outside acquisition window).

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data object.
        triggers : np.ndarray
            Trigger indices.
        artifact_length : int
            Length of one artifact in samples.
        time_buffer : float, optional
            Buffer in seconds to stay away from acquisition (default: 0.1).

        Returns
        -------
        np.ndarray
            Array of shape (n_channels, n_times) containing concatenated reference data.
        """
        sfreq = raw.info['sfreq']
        eeg_picks = self.get_eeg_channels(raw)

        if len(eeg_picks) == 0:
            logger.warning("No EEG channels found")
            return np.array([])

        # Acquisition is considered to span from first trigger to last trigger + artifact length.
        acq_start_sample = triggers[0]
        acq_end_sample = triggers[-1] + artifact_length

        acq_start_time = acq_start_sample / sfreq
        acq_end_time = acq_end_sample / sfreq

        ref_pre_tmax = acq_start_time - time_buffer
        ref_post_tmin = acq_end_time + time_buffer

        ref_segments = []

        if ref_pre_tmax > 0:
            try:
                data = raw.copy().crop(tmax=ref_pre_tmax).get_data(picks=eeg_picks)
                ref_segments.append(data)
            except Exception:
                pass

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
        """Extract data within the acquisition window.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw data object.
        triggers : np.ndarray
            Trigger indices.
        artifact_length : int
            Length of one artifact in samples.

        Returns
        -------
        np.ndarray
            Array of shape (n_channels, n_times) from the acquisition window.
        """
        sfreq = raw.info['sfreq']
        eeg_picks = self.get_eeg_channels(raw)

        if len(eeg_picks) == 0:
            return np.array([])

        acq_start = max(0, triggers[0] - int(artifact_length * 0.5))
        acq_end = min(raw.n_times, triggers[-1] + int(artifact_length * 1.5))

        tmin = acq_start / sfreq
        tmax = min(acq_end / sfreq, raw.times[-1])

        return raw.copy().crop(tmin=tmin, tmax=tmax).get_data(picks=eeg_picks)


@register_processor
class SNRCalculator(Processor, ReferenceDataMixin):
    """Calculate Signal-to-Noise Ratio (SNR).

    Compares corrected data to a clean reference (data outside acquisition
    window). Higher SNR indicates better correction.

    SNR = variance(reference) / variance(residual)

    Parameters
    ----------
    time_buffer : float, optional
        Time buffer around acquisition window in seconds (default: 0.1).

    Examples
    --------
    ::

        snr = SNRCalculator()
        context = snr.execute(context)
        print(context.metadata.custom['snr'])
    """

    name = "snr_calculator"
    description = "Calculate Signal-to-Noise Ratio"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, time_buffer: float = 0.1) -> None:
        self.time_buffer = time_buffer
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_raw_original() is None:
            raise ProcessorValidationError(
                "Original raw data not available. Cannot calculate SNR."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()

        # --- LOG ---
        logger.info("Calculating Signal-to-Noise Ratio (SNR)")

        # --- COMPUTE ---
        ref_data = self.get_reference_data(raw, triggers, artifact_length, self.time_buffer)
        corrected_data = self.get_acquisition_data(raw, triggers, artifact_length)

        if ref_data.size == 0 or corrected_data.size == 0:
            logger.warning("Insufficient data for SNR calculation")
            return context

        var_corrected = np.var(corrected_data, axis=1)
        var_reference = np.var(ref_data, axis=1)

        # Residual variance is the difference; clamped to avoid division by zero.
        var_residual = np.maximum(var_corrected - var_reference, 1e-10)

        snr_per_channel = np.abs(var_reference / var_residual)
        snr_mean = np.mean(snr_per_channel)

        report_metric("snr", float(snr_mean), label="SNR", display=f"{snr_mean:.2f}")

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault('metrics', {})
        metrics['snr'] = float(snr_mean)
        metrics['snr_per_channel'] = snr_per_channel.tolist()

        # --- RETURN ---
        return context.with_metadata(new_metadata)


@register_processor
class RMSResidualCalculator(Processor, ReferenceDataMixin):
    """Calculate RMS Residual Ratio (corrected vs. reference).

    Compares the RMS of the corrected signal (during acquisition) to the RMS
    of the clean reference signal (outside acquisition).

    Ratio = RMS(corrected) / RMS(reference)

    A ratio of 1.0 is the target. Values below 1.0 suggest over-correction;
    values above 1.0 indicate residual artifacts. Corresponds to
    ``rms_residual`` in FACET MATLAB Edition.

    Parameters
    ----------
    time_buffer : float, optional
        Time buffer around acquisition window in seconds (default: 0.1).
    """

    name = "rms_residual_calculator"
    description = "Calculate RMS ratio (corrected vs reference)"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, time_buffer: float = 0.1) -> None:
        self.time_buffer = time_buffer
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()

        # --- LOG ---
        logger.info("Calculating RMS Residual Ratio (corrected vs reference)")

        # --- COMPUTE ---
        ref_data = self.get_reference_data(raw, triggers, artifact_length, self.time_buffer)
        corrected_data = self.get_acquisition_data(raw, triggers, artifact_length)

        if ref_data.size == 0 or corrected_data.size == 0:
            logger.warning("Insufficient data for RMS Residual calculation")
            return context

        rms_corrected = np.std(corrected_data, axis=1)
        # Clamp to avoid division by zero.
        rms_reference = np.maximum(np.std(ref_data, axis=1), 1e-10)

        ratio_per_channel = rms_corrected / rms_reference
        ratio_mean = np.mean(ratio_per_channel)

        report_metric("rms_residual", float(ratio_mean), label="RMS Residual", display=f"{ratio_mean:.2f}")

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault('metrics', {})
        metrics['rms_residual'] = float(ratio_mean)
        metrics['rms_residual_per_channel'] = ratio_per_channel.tolist()

        # --- RETURN ---
        return context.with_metadata(new_metadata)


@register_processor
class LegacySNRCalculator(Processor):
    """Calculate legacy-style Signal-to-Noise Ratio (SNR).

    Mirrors the original FACET implementation by comparing the variance of the
    corrected data to the variance of the uncorrected reference recording.
    """

    name = "legacy_snr_calculator"
    description = "Legacy-style SNR using original raw as reference"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self) -> None:
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_raw_original() is context.get_raw():
            raise ProcessorValidationError(
                "Original raw data not available. Cannot calculate legacy SNR."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw_corrected = context.get_raw()
        raw_original = context.get_raw_original()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = raw_corrected.info['sfreq']

        # --- LOG ---
        logger.info("Calculating legacy SNR (corrected vs original)")

        # --- COMPUTE ---
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

        if triggers is None or len(triggers) == 0 or artifact_length is None:
            logger.warning("Legacy SNR requires triggers and artifact length; skipping")
            return context

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

        var_residual = np.maximum(var_corrected - var_reference, 1e-10)

        snr_per_channel = np.abs(var_reference / var_residual)
        snr_mean = float(np.mean(snr_per_channel))

        report_metric(
            "legacy_snr",
            snr_mean,
            label="Legacy SNR",
            display=f"{snr_mean:.2f}",
        )

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault('metrics', {})
        metrics['legacy_snr'] = snr_mean
        metrics['legacy_snr_per_channel'] = snr_per_channel.tolist()

        # --- RETURN ---
        return context.with_metadata(new_metadata)


@register_processor
class RMSCalculator(Processor):
    """Calculate Root Mean Square (RMS) improvement ratio.

    Compares RMS of corrected data to uncorrected data. A higher ratio
    indicates better correction (more artifact removed).

    RMS_ratio = RMS(uncorrected) / RMS(corrected)

    Examples
    --------
    ::

        rms = RMSCalculator()
        context = rms.execute(context)
        print(context.metadata.custom['rms_ratio'])
    """

    name = "rms_calculator"
    description = "Calculate RMS improvement ratio"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_raw_original() is None:
            raise ProcessorValidationError(
                "Original raw data not available. Cannot calculate RMS ratio."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        raw_orig = context.get_raw_original()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info['sfreq']

        # --- LOG ---
        logger.info("Calculating RMS improvement ratio")

        # --- COMPUTE ---
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found")
            return context

        acq_start = max(0, triggers[0] - int(artifact_length * 0.5))
        acq_end = min(raw.n_times, triggers[-1] + int(artifact_length * 1.5))

        acq_tmin = acq_start / sfreq
        acq_tmax = min(acq_end / sfreq, raw.times[-1])

        data_corrected = raw.copy().crop(tmin=acq_tmin, tmax=acq_tmax).get_data(picks=eeg_channels)
        data_uncorrected = raw_orig.copy().crop(tmin=acq_tmin, tmax=acq_tmax).get_data(picks=eeg_channels)

        if data_corrected.shape[0] != data_uncorrected.shape[0]:
            min_channels = min(data_corrected.shape[0], data_uncorrected.shape[0])
            data_corrected = data_corrected[:min_channels]
            data_uncorrected = data_uncorrected[:min_channels]

        rms_uncorrected = np.sqrt(np.mean(data_uncorrected ** 2, axis=1))
        # Clamp corrected RMS to avoid division by zero.
        rms_corrected = np.maximum(np.sqrt(np.mean(data_corrected ** 2, axis=1)), 1e-10)

        rms_ratio_per_channel = rms_uncorrected / rms_corrected
        rms_ratio = np.median(rms_ratio_per_channel)

        report_metric("rms_ratio", float(rms_ratio), label="RMS Ratio", display=f"{rms_ratio:.2f}")

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault('metrics', {})
        metrics['rms_ratio'] = float(rms_ratio)
        metrics['rms_ratio_per_channel'] = rms_ratio_per_channel.tolist()

        # --- RETURN ---
        return context.with_metadata(new_metadata)


@register_processor
class MedianArtifactCalculator(Processor, ReferenceDataMixin):
    """Calculate median peak-to-peak artifact amplitude.

    Measures the median artifact amplitude across all epochs. Lower values
    indicate smaller artifacts (better correction).

    Also calculates the ratio to the median amplitude of the reference signal
    (outside acquisition), which should ideally be close to 1.0.

    Examples
    --------
    ::

        median = MedianArtifactCalculator()
        context = median.execute(context)
        print(context.metadata.custom['median_artifact'])
    """

    name = "median_artifact_calculator"
    description = "Calculate median artifact amplitude"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector first."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        sfreq = raw.info['sfreq']
        artifact_len = context.get_artifact_length()

        # --- LOG ---
        logger.info("Calculating median artifact amplitude")

        # --- COMPUTE ---
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found")
            return context

        median_artifact, median_ref, ratio = self._compute_median_metrics(
            raw, triggers, sfreq, artifact_len, eeg_channels, context
        )

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

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        metrics = new_metadata.custom.setdefault('metrics', {})
        metrics['median_artifact'] = float(median_artifact)
        if not np.isnan(median_ref):
            metrics['median_artifact_reference'] = float(median_ref)
        if not np.isnan(ratio):
            metrics['median_artifact_ratio'] = float(ratio)

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    def _compute_median_metrics(
        self,
        raw: mne.io.BaseRaw,
        triggers: np.ndarray,
        sfreq: float,
        artifact_len: int,
        eeg_channels: np.ndarray,
        context: ProcessingContext,
    ):
        """Compute median artifact amplitude for corrected and reference data.

        Returns
        -------
        tuple
            (median_artifact, median_ref, ratio) where nan indicates unavailable.
        """
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

        p2p_per_epoch = [np.ptp(epoch, axis=1) for epoch in epochs.get_data(copy=False)]
        mean_p2p_per_epoch = [np.mean(epoch_p2p) for epoch_p2p in p2p_per_epoch]
        median_artifact = np.median(mean_p2p_per_epoch)

        ref_data = self.get_reference_data(raw, triggers, artifact_len)

        median_ref = np.nan
        ratio = np.nan

        if ref_data.size > 0:
            n_samples_ref = ref_data.shape[1]
            epoch_len = int(artifact_len)
            n_ref_epochs = n_samples_ref // epoch_len

            if n_ref_epochs > 0:
                ref_data_truncated = ref_data[:, :n_ref_epochs * epoch_len]
                # Reshape to (n_epochs, channels, samples) for per-epoch peak-to-peak.
                ref_epochs = ref_data_truncated.reshape(len(eeg_channels), n_ref_epochs, epoch_len)
                ref_epochs = np.moveaxis(ref_epochs, 1, 0)

                p2p_ref = [np.ptp(epoch, axis=1) for epoch in ref_epochs]
                mean_p2p_ref = [np.mean(epoch_p2p) for epoch_p2p in p2p_ref]
                median_ref = np.median(mean_p2p_ref)

                if median_ref > 0:
                    ratio = median_artifact / median_ref

        return median_artifact, median_ref, ratio


@register_processor
class FFTAllenCalculator(Processor, ReferenceDataMixin):
    """Calculate FFT Allen metric.

    Compares spectral power in specific frequency bands between corrected data
    and clean reference data. The metric is the median absolute percent
    difference per band.

    Bands: 0.8–4 Hz (delta), 4–8 Hz (theta), 8–12 Hz (alpha), 12–24 Hz (beta).

    Formula::

        metric = median(|Power_corr - Power_ref| / Power_ref) * 100
    """

    name = "fft_allen_calculator"
    description = "Calculate spectral power difference (Allen)"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    BANDS = [
        (0.8, 4, "Delta"),
        (4, 8, "Theta"),
        (8, 12, "Alpha"),
        (12, 24, "Beta")
    ]

    def __init__(self) -> None:
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_len = context.get_artifact_length()
        sfreq = raw.info['sfreq']

        # --- LOG ---
        logger.info("Calculating FFT Allen metric")

        # --- COMPUTE ---
        n_fft = int(3.0 * sfreq)  # 3-second segments as in MATLAB

        ref_data = self.get_reference_data(raw, triggers, artifact_len)
        corr_data = self.get_acquisition_data(raw, triggers, artifact_len)

        if ref_data.size == 0 or corr_data.size == 0:
            logger.warning("Insufficient data for FFT Allen")
            return context

        nperseg = min(n_fft, ref_data.shape[1], corr_data.shape[1])

        freqs_ref, psd_ref = signal.welch(ref_data, fs=sfreq, nperseg=nperseg, axis=1)
        freqs_corr, psd_corr = signal.welch(corr_data, fs=sfreq, nperseg=nperseg, axis=1)

        if not np.array_equal(freqs_ref, freqs_corr):
            logger.warning("Frequency mismatch in FFT Allen")
            return context

        results = self._compute_band_differences(freqs_ref, psd_ref, psd_corr)

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.custom.setdefault('metrics', {})['fft_allen'] = results

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    def _compute_band_differences(
        self,
        freqs: np.ndarray,
        psd_ref: np.ndarray,
        psd_corr: np.ndarray,
    ) -> Dict[str, float]:
        """Compute median absolute percent power difference per frequency band.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency array from Welch PSD.
        psd_ref : np.ndarray
            PSD of reference data, shape (n_channels, n_freqs).
        psd_corr : np.ndarray
            PSD of corrected data, shape (n_channels, n_freqs).

        Returns
        -------
        Dict[str, float]
            Band name → median percent difference.
        """
        results: Dict[str, float] = {}

        for fmin, fmax, band_name in self.BANDS:
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)

            power_ref = np.mean(psd_ref[:, idx], axis=1)
            power_corr = np.mean(psd_corr[:, idx], axis=1)

            diff_pct = np.abs(power_corr - power_ref) / (power_ref + 1e-10) * 100
            median_diff = np.median(diff_pct)

            results[band_name] = float(median_diff)

            logger.debug(
                "FFT Allen {} ({}-{}Hz): {:.2f}%", band_name, fmin, fmax, median_diff
            )

        return results


@register_processor
class FFTNiazyCalculator(Processor, ReferenceDataMixin):
    """Calculate FFT Niazy metric.

    Analyzes residual artifacts at slice and volume frequencies by computing
    the power ratio (uncorrected / corrected) at these frequencies and their
    harmonics. Values are reported in dB.
    """

    name = "fft_niazy_calculator"
    description = "Calculate spectral power ratio at slice/volume frequencies"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self) -> None:
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        raw_orig = context.get_raw_original()
        triggers = context.get_triggers()
        artifact_len = context.get_artifact_length()
        sfreq = raw.info['sfreq']

        # --- LOG ---
        logger.info("Calculating FFT Niazy metric")

        # --- COMPUTE ---
        if raw_orig is None:
            logger.warning("Original raw data missing for FFT Niazy")
            return context

        slice_freq, vol_freq = self._estimate_frequencies(triggers, sfreq, context)

        data_corr = self.get_acquisition_data(raw, triggers, artifact_len)
        data_orig = self.get_acquisition_data(raw_orig, triggers, artifact_len)

        min_ch = min(data_corr.shape[0], data_orig.shape[0])
        data_corr = data_corr[:min_ch]
        data_orig = data_orig[:min_ch]

        nperseg = min(int(4 * sfreq), data_corr.shape[1])
        freqs, psd_corr = signal.welch(data_corr, fs=sfreq, nperseg=nperseg, axis=1)
        _, psd_orig = signal.welch(data_orig, fs=sfreq, nperseg=nperseg, axis=1)

        results = self._compute_harmonic_ratios(
            freqs, psd_corr, psd_orig, slice_freq, vol_freq
        )

        slice_h1 = results['slice'].get('h1', float('nan'))
        if not np.isnan(slice_h1):
            logger.info("FFT Niazy Slice h1: {:.2f} dB", slice_h1)

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.custom.setdefault('metrics', {})['fft_niazy'] = results

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    def _estimate_frequencies(
        self,
        triggers: np.ndarray,
        sfreq: float,
        context: ProcessingContext,
    ):
        """Estimate slice and volume frequencies from triggers and context.

        Returns
        -------
        tuple
            (slice_freq, vol_freq) where vol_freq may be None.
        """
        mean_trigger_diff = np.mean(np.diff(triggers))
        slice_freq = sfreq / mean_trigger_diff

        slices_per_vol = getattr(context.metadata, 'slices_per_volume', None)
        if not slices_per_vol:
            slices_per_vol = context.metadata.custom.get('slices_per_volume')

        if slices_per_vol:
            vol_freq = slice_freq / slices_per_vol
        else:
            vol_freq = None
            logger.warning("Slices per volume not found, skipping volume frequency analysis")

        return slice_freq, vol_freq

    def _compute_harmonic_ratios(
        self,
        freqs: np.ndarray,
        psd_corr: np.ndarray,
        psd_orig: np.ndarray,
        slice_freq: float,
        vol_freq: Optional[float],
        harmonics: int = 5,
        tolerance: float = 0.5,
    ) -> Dict[str, Any]:
        """Compute power ratios (orig/corr in dB) at harmonic frequencies.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency array.
        psd_corr : np.ndarray
            PSD of corrected data, shape (n_channels, n_freqs).
        psd_orig : np.ndarray
            PSD of original data, shape (n_channels, n_freqs).
        slice_freq : float
            Fundamental slice frequency in Hz.
        vol_freq : float or None
            Fundamental volume frequency in Hz, or None to skip.
        harmonics : int, optional
            Number of harmonics to analyze (default: 5).
        tolerance : float, optional
            Half-bandwidth around each harmonic in Hz (default: 0.5).

        Returns
        -------
        Dict[str, Any]
            Nested dict with 'slice' and 'volume' harmonic ratios in dB.
        """
        results: Dict[str, Any] = {'slice': {}, 'volume': {}}

        def _ratio_db_at(f: float) -> Optional[float]:
            idx = np.logical_and(freqs >= f - tolerance, freqs <= f + tolerance)
            if not np.any(idx):
                return None
            p_corr = np.median(np.mean(psd_corr[:, idx], axis=1))
            p_orig = np.median(np.mean(psd_orig[:, idx], axis=1))
            if p_corr <= 0:
                return None
            return float(10 * np.log10(p_orig / p_corr))

        for h in range(1, harmonics + 1):
            ratio_db = _ratio_db_at(slice_freq * h)
            if ratio_db is not None:
                results['slice'][f"h{h}"] = ratio_db

        if vol_freq is not None:
            for h in range(1, harmonics + 1):
                ratio_db = _ratio_db_at(vol_freq * h)
                if ratio_db is not None:
                    results['volume'][f"h{h}"] = ratio_db

        return results


@register_processor
class MetricsReport(Processor):
    """Generate a summary report of all calculated metrics.

    Collects all metrics from context and logs a formatted summary. Can also
    store results in a shared dictionary for comparison and plotting.

    Parameters
    ----------
    name : str, optional
        Name of the result set (e.g., ``"Pipeline A"``). If ``None``, a
        default name is generated during processing.
    store : dict, optional
        Dictionary to accumulate results. Structure: ``{name: {metric: value}}``.

    Examples
    --------
    ::

        # Basic usage
        report = MetricsReport()
        context = report.execute(context)

        # Advanced usage (collecting results for comparison)
        results = {}
        report = MetricsReport(name="Pipeline A", store=results)
        context = report.execute(context)

        # Plot comparison
        MetricsReport.plot(results)

        # Plot specific metrics
        MetricsReport.plot(results, metrics=['snr', 'rms_ratio'])
    """

    name = "metrics_report"
    description = "Generate metrics summary report"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = False
    modifies_raw = False
    parallel_safe = False

    def __init__(self, name: Optional[str] = None, store: Optional[Dict] = None) -> None:
        self.report_name = name
        self.store = store
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        metrics = context.metadata.custom.get('metrics', {})

        # --- COMPUTE ---
        if not metrics:
            logger.warning("No metrics available")
            return context

        self._log_metrics(metrics)

        if self.store is not None:
            result_name = self.report_name or f"Result_{len(self.store) + 1}"
            self.store[result_name] = self._flatten_metrics(metrics)
            logger.info("Stored metrics for '{}'", result_name)

        # --- RETURN ---
        return context

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Render all available metrics — rich panel when available, plain log fallback."""
        from ..console import get_console

        if get_console().enabled:
            self._plain_log_metrics(metrics)
            return
        try:
            self._rich_log_metrics(metrics)
        except Exception:
            self._plain_log_metrics(metrics)

    def _rich_log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Render metrics as a rich-formatted panel."""
        from rich import box
        from rich.console import Console as RichConsole
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        console = RichConsole(highlight=False)
        table = Table(
            box=None, show_header=True, padding=(0, 2), expand=True,
            show_edge=False,
        )
        table.add_column("Metric", style="bold", ratio=3)
        table.add_column("Value", style="white", ratio=2, justify="left")
        table.add_column("", style="dim italic", ratio=1)

        def _section(title: str) -> None:
            table.add_row("", "", "")
            table.add_row(Text(title, style="bold yellow underline"), "", "")

        # --- Core Metrics ---
        core_keys = ('snr', 'rms_ratio', 'rms_residual', 'median_artifact', 'legacy_snr')
        if any(k in metrics for k in core_keys):
            _section("Core Metrics")

            if 'snr' in metrics:
                snr = metrics['snr']
                color = "green" if snr > 10 else ("yellow" if snr > 3 else "red")
                table.add_row("SNR (Signal-to-Noise Ratio)", f"[{color}]{snr:.2f}[/]", "")

            if 'rms_ratio' in metrics:
                table.add_row("RMS Ratio (improvement)", f"{metrics['rms_ratio']:.2f}", "×")

            if 'rms_residual' in metrics:
                r = metrics['rms_residual']
                color = "green" if abs(r - 1.0) < 0.1 else ("yellow" if abs(r - 1.0) < 0.3 else "red")
                table.add_row("RMS Residual Ratio", f"[{color}]{r:.2f}[/]", "target: 1.0")

            if 'median_artifact' in metrics:
                table.add_row(
                    "Median Artifact Amplitude", f"{metrics['median_artifact']:.2e}", ""
                )
                if 'median_artifact_ratio' in metrics:
                    r = metrics['median_artifact_ratio']
                    color = "green" if abs(r - 1.0) < 0.2 else ("yellow" if abs(r - 1.0) < 0.6 else "red")
                    table.add_row(
                        "Median Artifact Ratio", f"[{color}]{r:.2f}[/]", "target: 1.0"
                    )

            if 'legacy_snr' in metrics:
                table.add_row("Legacy SNR", f"{metrics['legacy_snr']:.2f}", "")

        # --- FFT Allen ---
        if 'fft_allen' in metrics:
            _section("FFT Allen — Spectral Diff to Reference")
            for band, val in metrics['fft_allen'].items():
                table.add_row(f"{band.capitalize()}", f"{val:.2f}%", "")

        # --- FFT Niazy ---
        if 'fft_niazy' in metrics:
            _section("FFT Niazy — Power Ratio (Uncorr / Corr)")
            if 'slice' in metrics['fft_niazy']:
                harmonics = "  ".join(
                    f"[cyan]{k}[/]: {v:.2f}" for k, v in metrics['fft_niazy']['slice'].items()
                )
                table.add_row("Slice Harmonics", harmonics, "dB")
            if 'volume' in metrics['fft_niazy']:
                harmonics = "  ".join(
                    f"[cyan]{k}[/]: {v:.2f}" for k, v in metrics['fft_niazy']['volume'].items()
                )
                table.add_row("Volume Harmonics", harmonics, "dB")

        console.print()
        console.print(
            Panel(
                table,
                title="[bold white] Evaluation Metrics Report [/]",
                border_style="cyan",
                box=box.ROUNDED,
                padding=(1, 2),
            )
        )

    def _plain_log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Fallback plain loguru output for the interactive console / no-TTY case."""
        logger.info("=" * 60)
        logger.info("EVALUATION METRICS REPORT")
        logger.info("=" * 60)

        if 'snr' in metrics:
            logger.info("SNR (Signal-to-Noise Ratio):     {:.2f}", metrics['snr'])

        if 'rms_ratio' in metrics:
            logger.info("RMS Ratio (improvement):         {:.2f}", metrics['rms_ratio'])

        if 'rms_residual' in metrics:
            logger.info(
                "RMS Residual Ratio (ref match):  {:.2f} (target: 1.0)",
                metrics['rms_residual'],
            )

        if 'median_artifact' in metrics:
            logger.info("Median Artifact Amplitude:       {:.2e}", metrics['median_artifact'])
            if 'median_artifact_ratio' in metrics:
                logger.info(
                    "Median Artifact Ratio (to ref):  {:.2f} (target: 1.0)",
                    metrics['median_artifact_ratio'],
                )

        if 'legacy_snr' in metrics:
            logger.info("Legacy SNR:                      {:.2f}", metrics['legacy_snr'])

        if 'fft_allen' in metrics:
            logger.info("FFT Allen (Diff to Ref):")
            for band, val in metrics['fft_allen'].items():
                logger.info("  - {}: {:.2f}%", band, val)

        if 'fft_niazy' in metrics:
            logger.info("FFT Niazy (Power Ratio Uncorr/Corr dB):")
            if 'slice' in metrics['fft_niazy']:
                logger.info("  Slice Harmonics:")
                for k, v in metrics['fft_niazy']['slice'].items():
                    logger.info("    - {}: {:.2f} dB", k, v)
            if 'volume' in metrics['fft_niazy']:
                logger.info("  Volume Harmonics:")
                for k, v in metrics['fft_niazy']['volume'].items():
                    logger.info("    - {}: {:.2f} dB", k, v)

        logger.info("=" * 60)

    @staticmethod
    def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
        """Flatten nested metric dicts to scalar values for plotting.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Raw metrics dict from context (may contain nested dicts).

        Returns
        -------
        Dict[str, float]
            Flat dict of scalar metric values.
        """
        scalar_metrics: Dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.number)):
                scalar_metrics[k] = float(v)
            elif k == 'fft_allen' and isinstance(v, dict):
                for band, val in v.items():
                    scalar_metrics[f"fft_allen_{band}"] = float(val)
            elif k == 'fft_niazy' and isinstance(v, dict):
                if 'slice' in v:
                    scalar_metrics["fft_niazy_slice_h1"] = float(v['slice'].get('h1', 0))
        return scalar_metrics

    @staticmethod
    def compare(
        results: Union[List, Dict],
        labels: Optional[List[str]] = None,
        title: str = "Metrics Comparison",
        save_path: Optional[str] = None,
        show: bool = True,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """Compare metrics from a list of ``PipelineResult`` objects or a plain dict.

        Accepts either:

        - A list of ``PipelineResult`` instances (optionally named via *labels*).
        - The legacy ``{name: {metric: value}}`` dict format used by
          ``MetricsReport.plot``.

        Parameters
        ----------
        results : list or dict
            List of ``PipelineResult`` objects **or** legacy dict.
        labels : list of str, optional
            Names for each result when passing a list. Defaults to
            ``["Result 1", "Result 2", …]``.
        title : str, optional
            Plot title (default: "Metrics Comparison").
        save_path : str, optional
            If set, save the figure to this path.
        show : bool, optional
            Whether to display the figure interactively (default: True).
        metrics : list of str, optional
            Subset of metric keys to show. If ``None``, all metrics are plotted.

        Examples
        --------
        ::

            result_a = pipeline_aas.run()
            result_b = pipeline_aas_pca.run()

            MetricsReport.compare(
                [result_a, result_b],
                labels=["AAS only", "AAS + PCA"],
            )
        """
        if isinstance(results, dict):
            MetricsReport.plot(results, title=title, save_path=save_path, show=show, metrics=metrics)
            return

        if labels is None:
            labels = [f"Result {i + 1}" for i in range(len(results))]

        results_dict: Dict[str, Dict[str, float]] = {}
        for label, result in zip(labels, results):
            raw_metrics = result.metrics if hasattr(result, 'metrics') else {}
            results_dict[label] = MetricsReport._flatten_metrics(raw_metrics)

        MetricsReport.plot(results_dict, title=title, save_path=save_path, show=show, metrics=metrics)

    @staticmethod
    def plot(
        results: Dict[str, Dict[str, float]],
        title: str = "Metrics Comparison",
        save_path: Optional[str] = None,
        show: bool = True,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """Plot comparison of metrics using Matplotlib.

        Parameters
        ----------
        results : Dict[str, Dict[str, float]]
            Dictionary of results ``{name: {metric: value}}``.
        title : str, optional
            Plot title (default: "Metrics Comparison").
        save_path : str, optional
            Path to save the figure.
        show : bool, optional
            Whether to show the plot (default: True).
        metrics : list of str, optional
            Subset of metric keys to plot. If ``None``, all are shown.
        """
        if not results:
            logger.warning("No results to plot")
            return

        df = pd.DataFrame.from_dict(results, orient='index')

        if df.empty:
            logger.warning("Results DataFrame is empty")
            return

        if metrics:
            existing_metrics = [m for m in metrics if m in df.columns]
            if not existing_metrics:
                logger.warning(
                    "None of the requested metrics {} found in results.", metrics
                )
                return
            df = df[existing_metrics]

        metrics_list = df.columns.tolist()
        n_metrics = len(metrics_list)

        if n_metrics == 0:
            logger.warning("No metrics found in results")
            return

        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
        axes = axes.flatten()

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0.2, 0.8, len(df))]

        for i, metric in enumerate(metrics_list):
            ax = axes[i]
            values = df[metric]
            values.plot(kind='bar', ax=ax, color=colors, rot=45)

            ax.set_title(metric)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            for p in ax.patches:
                height = p.get_height()
                ax.annotate(
                    f"{height:.2g}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=8,
                    rotation=0,
                )

        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            logger.info("Saved metrics plot to {}", save_path)

        if show:
            plt.show()
        else:
            plt.close()
