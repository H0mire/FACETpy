"""
PCA-based Artifact Correction Module

This module contains processors for PCA-based artifact removal.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, Union
import mne
import numpy as np
from loguru import logger
from scipy.signal import filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError
from ..helpers.utils import split_vector


@register_processor
class PCACorrection(Processor):
    """
    PCA-based artifact correction processor.

    This processor applies Principal Component Analysis to remove artifacts
    from EEG data. The algorithm:

    1. Splits data into epochs based on trigger positions
    2. Applies highpass filtering to each epoch (optional)
    3. Standardizes epoch data
    4. Applies PCA to find principal components
    5. Reconstructs data using selected components
    6. Subtracts reconstruction from original to get residuals (artifacts)
    7. Removes residuals from original data

    The number of components can be specified as:
    - An integer: exact number of components
    - A float (0 < n < 1): fraction of variance to retain
    - 0: skip PCA for this channel

    Example:
        correction = PCACorrection(n_components=0.95)  # Keep 95% variance
        context = correction.execute(context)

        correction = PCACorrection(n_components=5)  # Keep 5 components
        context = correction.execute(context)
    """

    name = "pca_correction"
    description = "PCA-based artifact removal"
    requires_triggers = True
    parallel_safe = True
    parallelize_by_channels = True

    def __init__(
        self,
        n_components: Union[int, float] = 0.95,
        hp_freq: Optional[float] = None,
        hp_filter_weights: Optional[np.ndarray] = None,
        exclude_channels: Optional[list] = None
    ):
        """
        Initialize PCA correction.

        Args:
            n_components: Number of components to keep (int) or variance fraction (float)
            hp_freq: Highpass frequency for preprocessing (None to skip)
            hp_filter_weights: Pre-computed filter weights (overrides hp_freq)
            exclude_channels: List of channel indices to exclude from PCA
        """
        self.n_components = n_components
        self.hp_freq = hp_freq
        self.hp_filter_weights = hp_filter_weights
        self.exclude_channels = exclude_channels or []
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate prerequisites."""
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector first."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Apply PCA correction."""
        logger.info("Applying PCA artifact correction")

        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info['sfreq']

        # Get EEG channels
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found, skipping PCA")
            return context

        # Prepare highpass filter if needed
        if self.hp_filter_weights is not None:
            hp_weights = self.hp_filter_weights
        elif self.hp_freq is not None and self.hp_freq > 0:
            hp_weights = self._create_hp_filter(sfreq)
        else:
            hp_weights = None

        # Determine acquisition window
        s_acq_start, s_acq_end = self._get_acquisition_window(context)

        # Apply PCA to each channel
        raw_corrected = raw.copy()

        # Initialize or get estimated noise
        if not context.has_estimated_noise():
            estimated_noise = np.zeros(raw._data.shape)
        else:
            estimated_noise = context.get_estimated_noise().copy()

        for ch_idx in eeg_channels:
            ch_name = raw.ch_names[ch_idx]
            logger.debug(f"  Applying PCA to {ch_name}")

            # Check if channel should be excluded
            if ch_idx in self.exclude_channels:
                logger.debug(f"  Skipping {ch_name} (excluded)")
                continue

            # Skip if n_components is 0
            if self.n_components == 0:
                logger.debug(f"  Skipping {ch_name} (n_components=0)")
                continue

            # Apply PCA to this channel
            try:
                residuals = self._calc_pca_residuals(
                    raw_corrected._data[ch_idx],
                    triggers,
                    artifact_length,
                    s_acq_start,
                    s_acq_end,
                    hp_weights
                )

                # Subtract residuals from data
                raw_corrected._data[ch_idx][s_acq_start:s_acq_end] -= residuals

                # Add to estimated noise
                estimated_noise[ch_idx][s_acq_start:s_acq_end] += residuals

            except Exception as ex:
                logger.error(f"PCA failed for channel {ch_name}: {ex}")
                logger.warning(f"Skipping PCA for channel {ch_name}")

        # Create new context
        new_context = context.with_raw(raw_corrected)
        new_context.set_estimated_noise(estimated_noise)

        logger.info("PCA correction completed")
        return new_context

    def _calc_pca_residuals(
        self,
        ch_data: np.ndarray,
        triggers: np.ndarray,
        artifact_length: int,
        s_acq_start: int,
        s_acq_end: int,
        hp_weights: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate PCA residuals for a single channel.

        Args:
            ch_data: Channel data
            triggers: Trigger positions
            artifact_length: Length of artifacts in samples
            s_acq_start: Start of acquisition window
            s_acq_end: End of acquisition window
            hp_weights: Highpass filter weights (or None)

        Returns:
            Residuals (artifacts) as array
        """
        # Get acquisition data
        ch_data_acq = ch_data[s_acq_start:s_acq_end]

        # Apply highpass filter if specified
        if hp_weights is not None:
            ch_data_filtered = filtfilt(hp_weights, 1, ch_data_acq)
        else:
            ch_data_filtered = ch_data_acq

        # Split into epochs
        # Adjust triggers to acquisition window coordinates
        adjusted_triggers = triggers - s_acq_start
        offset = int(artifact_length * 0.1)  # Small offset for epoch extraction

        epochs = split_vector(
            ch_data_filtered,
            adjusted_triggers + offset,
            artifact_length
        )

        # Apply PCA
        residuals_per_epoch = self._calc_pca(epochs)

        # Reconstruct full residuals array
        fitted_res = np.zeros(len(ch_data_acq))

        for i, trigger in enumerate(adjusted_triggers):
            start_pos = trigger + offset
            end_pos = start_pos + artifact_length

            # Handle boundary conditions
            if start_pos < 0:
                continue

            if end_pos > len(ch_data_acq):
                epoch_length = len(ch_data_acq) - start_pos
                if epoch_length <= 0:
                    continue
                fitted_res[start_pos:] = residuals_per_epoch[i, :epoch_length]
            else:
                fitted_res[start_pos:end_pos] = residuals_per_epoch[i, :]

        return fitted_res

    def _calc_pca(self, epochs: np.ndarray) -> np.ndarray:
        """
        Apply PCA to epochs and calculate residuals.

        Args:
            epochs: Epochs array (n_epochs, n_times)

        Returns:
            Residuals array (n_epochs, n_times)
        """
        # Transpose to (n_times, n_epochs) for sklearn
        epochs_t = epochs.T

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(epochs_t)

        # Apply PCA
        pca = PCA(n_components=self.n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Reconstruct data
        X_reconstructed = pca.inverse_transform(X_pca)
        X_reconstructed = scaler.inverse_transform(X_reconstructed)

        # Calculate residuals (artifact = original - reconstructed)
        residuals = epochs_t - X_reconstructed

        # Transpose back to (n_epochs, n_times)
        return residuals.T

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
