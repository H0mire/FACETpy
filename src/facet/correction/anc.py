"""
Adaptive Noise Cancellation (ANC) Module

This module contains processors for ANC-based artifact correction.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, Dict, Any
import mne
import numpy as np
from loguru import logger
from scipy.signal import filtfilt

from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError
from ..console import processor_progress


@register_processor
class ANCCorrection(Processor):
    """
    Adaptive Noise Cancellation (ANC) correction processor.

    This processor applies adaptive filtering to remove residual artifacts
    from EEG data using the estimated noise from previous correction steps (e.g., AAS).

    The ANC algorithm:
    1. Uses estimated noise as reference signal
    2. Applies highpass filtering to both EEG and reference
    3. Scales reference to match EEG amplitude (Alpha factor)
    4. Adapts filter coefficients using LMS algorithm (via fastranc C extension)
    5. Subtracts filtered noise from EEG data

    The algorithm parameters (mu, Alpha) are automatically derived from the data
    to ensure stability and optimal performance.

    Example:
        correction = ANCCorrection(filter_order=5, hp_freq=1.0)
        context = correction.execute(context)
    """

    name = "anc_correction"
    description = "Adaptive Noise Cancellation for residual artifacts"
    requires_triggers = True
    parallel_safe = False
    parallelize_by_channels = True

    def __init__(
        self,
        filter_order: Optional[int] = None,
        hp_freq: Optional[float] = None,
        hp_filter_weights: Optional[np.ndarray] = None,
        use_c_extension: bool = True,
        mu_factor: float = 0.05,
        max_gain: float = 50.0
    ):
        """
        Initialize ANC correction.

        Args:
            filter_order: Optional override for adaptive filter order (defaults to artifact length)
            hp_freq: Highpass frequency for preprocessing (None for no filter)
            hp_filter_weights: Pre-computed filter weights (overrides hp_freq)
            use_c_extension: Whether to use C extension for speed (fallback to Python)
            mu_factor: Learning rate factor (actual mu = mu_factor / (N * var(ref)))
            max_gain: Maximum allowed ratio between filtered noise and input segment
        """
        self.filter_order_override = (
            max(1, int(filter_order)) if filter_order is not None else None
        )
        self.hp_freq = hp_freq
        self.hp_filter_weights = hp_filter_weights
        self.use_c_extension = use_c_extension
        self.mu_factor = mu_factor
        self.max_gain = max_gain
        self._fastranc_available = None
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate prerequisites."""
        super().validate(context)
        if not context.has_estimated_noise():
            raise ProcessorValidationError(
                "Estimated noise not available. Run AAS or other correction first."
            )
        if not context.has_triggers():
            raise ProcessorValidationError(
                "Triggers not set. Run TriggerDetector first."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Apply ANC correction."""
        logger.info("Applying Adaptive Noise Cancellation (ANC)")

        raw = context.get_raw()
        estimated_noise = context.get_estimated_noise()
        sfreq = raw.info['sfreq']

        # Get EEG channels
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found, skipping ANC")
            return context

        # Derive ANC parameters (mirrors legacy behaviour)
        artifact_length = context.get_artifact_length()
        if artifact_length is None or artifact_length <= 0:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector before ANC."
            )

        derived_params = self._derive_parameters(
            context=context,
            artifact_length=artifact_length,
            sfreq=sfreq
        )

        if self.hp_filter_weights is not None:
            hp_weights = self.hp_filter_weights
            hp_cutoff = self.hp_freq if self.hp_freq is not None else derived_params['hp_freq']
        elif self.hp_freq is not None and self.hp_freq > 0:
            hp_weights = self._design_highpass(self.hp_freq, sfreq)
            hp_cutoff = self.hp_freq
        else:
            hp_weights = derived_params['hp_weights']
            hp_cutoff = derived_params['hp_freq']

        # Check if C extension is available
        if self.use_c_extension and self._fastranc_available is None:
            self._fastranc_available = self._check_fastranc()

        # Determine acquisition window
        s_acq_start, s_acq_end = self._get_acquisition_window(context)
        window_length = max(1, s_acq_end - s_acq_start)

        if self.filter_order_override is not None:
            filter_order = min(self.filter_order_override, window_length)
        else:
            filter_order = min(derived_params['filter_order'], window_length)

        filter_order = max(1, int(filter_order))

        # Apply ANC to each channel
        raw_corrected = raw.copy()
        noise_updated = estimated_noise.copy()

        with processor_progress(
            total=len(eeg_channels) or None,
            message="Adaptive noise cancellation",
        ) as progress:
            for idx, ch_idx in enumerate(eeg_channels):
                ch_name = raw.ch_names[ch_idx]
                logger.debug(f"  Applying ANC to {ch_name}")

                # Apply ANC to this channel
                try:
                    corrected_data, filtered_noise = self._anc_single_channel(
                        raw_corrected._data[ch_idx],
                        estimated_noise[ch_idx],
                        s_acq_start,
                        s_acq_end,
                        hp_weights,
                        filter_order,
                        ch_name 
                    )

                    raw_corrected._data[ch_idx] = corrected_data
                    noise_updated[ch_idx, s_acq_start:s_acq_end] += filtered_noise
                    status = f"\t{idx + 1}/{len(eeg_channels)} • {ch_name}"
                except Exception as ex:
                    logger.error(f"ANC failed for channel {ch_name}: {ex}")
                    logger.warning(f"Skipping ANC for channel {ch_name}")
                    status = f"\{idx + 1}/{len(eeg_channels)} • {ch_name} (skipped)"

                progress.advance(1, message=status)

        # Create new context
        new_context = context.with_raw(raw_corrected)
        new_context.set_estimated_noise(noise_updated)

        # Persist derived parameters for diagnostics
        anc_meta = new_context.metadata.custom.setdefault('anc', {})
        anc_meta.update(
            {
                'hp_frequency_hz': hp_cutoff,
                'filter_order': filter_order,
                'mu_factor': self.mu_factor,
                'max_gain': self.max_gain,
                'used_c_extension': bool(self._fastranc_available),
            }
        )

        logger.info("ANC correction completed")
        return new_context

    def _anc_single_channel(
        self,
        eeg_data: np.ndarray,
        noise_data: np.ndarray,
        s_acq_start: int,
        s_acq_end: int,
        hp_weights: Optional[np.ndarray],
        filter_order: int,
        ch_name: str
    ) -> tuple:
        """
        Apply ANC to a single channel.

        Args:
            eeg_data: EEG channel data
            noise_data: Estimated noise for this channel
            s_acq_start: Start of acquisition window
            s_acq_end: End of acquisition window
            hp_weights: Highpass filter weights (or None)

        Returns:
            Tuple of (corrected_data, filtered_noise)
        """
        # Extract acquisition segments
        reference = noise_data[s_acq_start:s_acq_end].astype(float, copy=True)
        segment_len = reference.size

        if segment_len == 0:
            logger.warning(f"[Channel: {ch_name}] ANC reference window is empty, skipping")
            return eeg_data, np.zeros(0, dtype=float)
        if segment_len <= filter_order:
            logger.warning(
                f"[Channel: {ch_name}] ANC reference window shorter than filter order, skipping"
            )
            return eeg_data, np.zeros(segment_len, dtype=float)

        # Apply highpass filter to data
        if hp_weights is not None:
            data = filtfilt(hp_weights, 1, eeg_data, axis=0, padtype='odd')
            data = data[s_acq_start:s_acq_end].astype(float)
        else:
            data = eeg_data[s_acq_start:s_acq_end].astype(float)

        # Calculate Alpha scaling factor
        ref_energy = np.sum(reference * reference)
        if not np.isfinite(ref_energy) or ref_energy <= np.finfo(float).eps:
            logger.warning(f"[Channel: {ch_name}] Reference energy is too small, skipping ANC")
            return eeg_data, np.zeros(segment_len, dtype=float)

        alpha = np.sum(data * reference) / ref_energy
        if not np.isfinite(alpha):
            logger.warning(f"[Channel: {ch_name}] Alpha scaling for ANC is not finite, skipping")
            return eeg_data, np.zeros(segment_len, dtype=float)

        reference = (alpha * reference).astype(float)

        # Calculate mu (learning rate)
        var_ref = np.var(reference)
        if not np.isfinite(var_ref) or var_ref <= np.finfo(float).eps:
            logger.warning(f"[Channel: {ch_name}] Reference variance is zero, skipping ANC")
            return eeg_data, np.zeros(segment_len, dtype=float)

        mu = float(self.mu_factor / (filter_order * var_ref))
        if not np.isfinite(mu) or mu <= 0:
            logger.warning(f"[Channel: {ch_name}] Computed ANC learning rate is invalid, skipping")
            return eeg_data, np.zeros(segment_len, dtype=float)

        # Apply adaptive filtering
        if self._fastranc_available:
            filtered_noise = self._anc_fast(reference, data, mu, filter_order)
        else:
            filtered_noise = self._anc_python(reference, data, mu, filter_order)

        # Check for numerical issues
        max_filtered = np.max(np.abs(filtered_noise))
        if not np.isfinite(max_filtered):
            logger.error(f"[Channel: {ch_name}] ANC produced invalid values (inf/nan), skipping")
            return eeg_data, np.zeros(segment_len, dtype=float)

        eeg_segment = eeg_data[s_acq_start:s_acq_end]
        baseline = np.max(np.abs(eeg_segment)) if eeg_segment.size else 0.0
        if baseline > 0:
            gain = max_filtered / baseline
        else:
            gain = np.inf if max_filtered > 0 else 0.0

        if gain > self.max_gain:
            logger.error(
                f"[Channel: {ch_name}] ANC produced unstable gain ({gain:.2e}), skipping correction"
            )
            return eeg_data, np.zeros(segment_len, dtype=float)

        # Subtract filtered noise from original data
        corrected_data = eeg_data.copy()
        corrected_data[s_acq_start:s_acq_end] -= filtered_noise

        return corrected_data, filtered_noise

    def _anc_fast(
        self,
        reference: np.ndarray,
        data: np.ndarray,
        mu: float,
        filter_order: int
    ) -> np.ndarray:
        """
        Apply ANC using C extension (fast).

        Args:
            reference: Reference signal (scaled noise)
            data: Data signal (EEG)
            mu: Learning rate

        Returns:
            Filtered noise signal
        """
        from ..helpers.fastranc import fastr_anc
        _, filtered_noise = fastr_anc(reference, data, filter_order, mu)
        return filtered_noise

    def _anc_python(
        self,
        reference: np.ndarray,
        data: np.ndarray,
        mu: float,
        filter_order: int
    ) -> np.ndarray:
        """
        Apply ANC using pure Python (fallback).

        Implements the LMS (Least Mean Squares) adaptive filter algorithm.

        Args:
            reference: Reference signal (scaled noise)
            data: Data signal (EEG)
            mu: Learning rate

        Returns:
            Filtered noise signal
        """
        N = max(1, int(filter_order))
        length = len(reference)

        # Initialize filter weights and output
        w = np.zeros(N)
        y = np.zeros(length)

        # LMS algorithm
        for n in range(N, length):
            # Get reference window
            x = reference[n - N:n][::-1]  # Reverse for convolution

            # Filter output
            y[n] = np.dot(w, x)

            # Error signal
            e = data[n] - y[n]

            # Update weights (LMS)
            w += mu * e * x

        return y

    def _derive_parameters(
        self,
        context: ProcessingContext,
        artifact_length: int,
        sfreq: float
    ) -> Dict[str, Any]:
        """
        Derive ANC parameters using legacy heuristics.
        """
        triggers = context.get_triggers()
        if triggers is None:
            triggers = np.array([], dtype=int)

        # Estimate trigger frequency (events per second)
        if len(triggers) >= 2:
            # Count triggers occurring within one second from the first event
            cutoff_samples = int(sfreq)
            count = 1
            while count < len(triggers):
                if triggers[count] - triggers[0] >= cutoff_samples:
                    break
                count += 1
            trigger_rate = max(count, 1)
        else:
            trigger_rate = 1

        hp_freq = 0.75 * trigger_rate if trigger_rate >= 1 else 2.0
        hp_freq = max(hp_freq, 0.5)  # ensure reasonable minimum

        hp_weights = self._design_highpass(hp_freq, sfreq)
        filter_order = max(artifact_length, 1)

        return {
            'hp_freq': hp_freq,
            'hp_weights': hp_weights,
            'filter_order': filter_order
        }

    def _design_highpass(self, cutoff_hz: float, sfreq: float) -> np.ndarray:
        """
        Design high-pass FIR filter using firls (legacy-style).
        """
        from scipy.signal import firls

        nyq = 0.5 * sfreq
        cutoff_hz = min(max(cutoff_hz, 0.5), nyq * 0.95)
        trans = 0.15
        pass_edge = cutoff_hz / nyq
        stop_edge = max(pass_edge * (1 - trans), 0.0)
        stop_edge = min(stop_edge, pass_edge * 0.999)

        taps = int(round(1.2 * sfreq / (cutoff_hz * (1 - trans))))
        taps = max(taps | 1, 3)  # make odd, at least 3

        f = [0.0, stop_edge, pass_edge, 1.0]
        a = [0.0, 0.0, 1.0, 1.0]

        try:
            weights = firls(taps, f, a)
        except Exception as ex:
            logger.warning(f"FIR design failed ({ex}); falling back to Butterworth")
            from scipy.signal import butter
            b, _ = butter(5, pass_edge, btype='high')
            weights = b

        return weights.astype(float)

    def _get_acquisition_window(self, context: ProcessingContext) -> tuple:
        """
        Get acquisition window (start and end samples).

        Args:
            context: Processing context

        Returns:
            Tuple of (s_acq_start, s_acq_end)
        """
        raw = context.get_raw()
        triggers = context.get_triggers()

        if len(triggers) == 0:
            return 0, raw.n_times

        artifact_length = context.get_artifact_length()
        if artifact_length is None:
            return 0, raw.n_times

        # Use trigger positions to define acquisition window
        s_acq_start = max(0, triggers[0] - artifact_length)
        s_acq_end = min(raw.n_times, triggers[-1] + artifact_length)

        return s_acq_start, s_acq_end

    def _check_fastranc(self) -> bool:
        """
        Check if fastranc C extension is available.

        Returns:
            True if available, False otherwise
        """
        try:
            from ..helpers.fastranc import fastranc
            if fastranc is not None:
                logger.debug("Using fastranc C extension for ANC")
                return True
            else:
                logger.info("fastranc C extension not available, using Python fallback")
                return False
        except Exception as ex:
            logger.info(f"fastranc C extension not available ({ex}), using Python fallback")
            return False


# Alias for backwards compatibility
AdaptiveNoiseCancellation = ANCCorrection
