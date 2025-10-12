"""
Adaptive Noise Cancellation (ANC) Module

This module contains processors for ANC-based artifact correction.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional
import mne
import numpy as np
from loguru import logger
from scipy.signal import filtfilt

from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError


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
    parallel_safe = True
    parallelize_by_channels = True

    def __init__(
        self,
        filter_order: int = 5,
        hp_freq: Optional[float] = None,
        hp_filter_weights: Optional[np.ndarray] = None,
        use_c_extension: bool = True,
        mu_factor: float = 0.05
    ):
        """
        Initialize ANC correction.

        Args:
            filter_order: Order of adaptive filter (number of coefficients)
            hp_freq: Highpass frequency for preprocessing (None for no filter)
            hp_filter_weights: Pre-computed filter weights (overrides hp_freq)
            use_c_extension: Whether to use C extension for speed (fallback to Python)
            mu_factor: Learning rate factor (actual mu = mu_factor / (N * var(ref)))
        """
        self.filter_order = filter_order
        self.hp_freq = hp_freq
        self.hp_filter_weights = hp_filter_weights
        self.use_c_extension = use_c_extension
        self.mu_factor = mu_factor
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

        # Prepare highpass filter if needed
        if self.hp_filter_weights is not None:
            hp_weights = self.hp_filter_weights
        elif self.hp_freq is not None and self.hp_freq > 0:
            hp_weights = self._create_hp_filter(sfreq)
        else:
            hp_weights = None

        # Check if C extension is available
        if self.use_c_extension and self._fastranc_available is None:
            self._fastranc_available = self._check_fastranc()

        # Determine acquisition window
        s_acq_start, s_acq_end = self._get_acquisition_window(context)

        # Apply ANC to each channel
        raw_corrected = raw.copy()
        noise_updated = estimated_noise.copy()

        for ch_idx in eeg_channels:
            ch_name = raw.ch_names[ch_idx]
            logger.debug(f"  Applying ANC to {ch_name}")

            # Apply ANC to this channel
            try:
                corrected_data, filtered_noise = self._anc_single_channel(
                    raw_corrected._data[ch_idx],
                    estimated_noise[ch_idx],
                    s_acq_start,
                    s_acq_end,
                    hp_weights
                )

                raw_corrected._data[ch_idx] = corrected_data
                noise_updated[ch_idx, s_acq_start:s_acq_end] += filtered_noise
            except Exception as ex:
                logger.error(f"ANC failed for channel {ch_name}: {ex}")
                logger.warning(f"Skipping ANC for channel {ch_name}")

        # Create new context
        new_context = context.with_raw(raw_corrected)
        new_context.set_estimated_noise(noise_updated)

        logger.info("ANC correction completed")
        return new_context

    def _anc_single_channel(
        self,
        eeg_data: np.ndarray,
        noise_data: np.ndarray,
        s_acq_start: int,
        s_acq_end: int,
        hp_weights: Optional[np.ndarray]
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
        reference = noise_data[s_acq_start:s_acq_end].copy()

        # Apply highpass filter to data
        if hp_weights is not None:
            data = filtfilt(hp_weights, 1, eeg_data, axis=0, padtype='odd')
            data = data[s_acq_start:s_acq_end].astype(float)
        else:
            data = eeg_data[s_acq_start:s_acq_end].astype(float)

        # Calculate Alpha scaling factor
        alpha = np.sum(data * reference) / np.sum(reference * reference)
        reference = (alpha * reference).astype(float)

        # Calculate mu (learning rate)
        var_ref = np.var(reference)
        if var_ref == 0:
            logger.warning("Reference variance is zero, skipping ANC")
            return eeg_data, np.zeros_like(reference)

        mu = float(self.mu_factor / (self.filter_order * var_ref))

        # Apply adaptive filtering
        if self._fastranc_available:
            filtered_noise = self._anc_fast(reference, data, mu)
        else:
            filtered_noise = self._anc_python(reference, data, mu)

        # Check for numerical issues
        if np.isinf(np.max(filtered_noise)) or np.isnan(np.max(filtered_noise)):
            logger.error("ANC produced invalid values (inf/nan), skipping")
            return eeg_data, np.zeros_like(reference)

        # Subtract filtered noise from original data
        corrected_data = eeg_data.copy()
        corrected_data[s_acq_start:s_acq_end] -= filtered_noise

        return corrected_data, filtered_noise

    def _anc_fast(
        self,
        reference: np.ndarray,
        data: np.ndarray,
        mu: float
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
        _, filtered_noise = fastr_anc(reference, data, self.filter_order, mu)
        return filtered_noise

    def _anc_python(
        self,
        reference: np.ndarray,
        data: np.ndarray,
        mu: float
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
        N = self.filter_order
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

    def _create_hp_filter(self, sfreq: float) -> np.ndarray:
        """
        Create highpass filter weights.

        Args:
            sfreq: Sampling frequency

        Returns:
            Filter weights for filtfilt
        """
        from scipy.signal import butter
        nyq = 0.5 * sfreq
        normalized_cutoff = self.hp_freq / nyq
        b, a = butter(5, normalized_cutoff, btype='high')
        return b

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
