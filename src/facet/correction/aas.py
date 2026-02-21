"""
Averaged Artifact Subtraction (AAS) Module

This module contains processors for AAS-based artifact correction.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, Dict
import mne
import numpy as np
from loguru import logger

from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError
from ..console import processor_progress
from ..helpers.utils import split_vector
from ..helpers.crosscorr import crosscorrelation


@register_processor
class AASCorrection(Processor):
    """
    Averaged Artifact Subtraction (AAS) correction processor.

    This processor implements the AAS algorithm for fMRI artifact removal:
    1. Creates epochs around each trigger
    2. For each epoch, finds highly correlated epochs within a sliding window
    3. Averages the correlated epochs to create an artifact template
    4. Subtracts the template from each epoch

    The algorithm adapts to non-stationary artifacts by using correlation-based
    epoch selection within a sliding window.

    Example:
        correction = AASCorrection(window_size=30, correlation_threshold=0.975)
        context = correction.execute(context)
    """

    name = "aas_correction"
    description = "Averaged Artifact Subtraction for fMRI artifacts"
    requires_triggers = True
    parallel_safe = True
    parallelize_by_channels = True

    def __init__(
        self,
        window_size: int = 30,
        rel_window_position: float = 0.0,
        correlation_threshold: float = 0.975,
        plot_artifacts: bool = False,
        realign_after_averaging: bool = True,
        search_window_factor: float = 3.0
    ):
        """
        Initialize AAS correction.

        Args:
            window_size: Number of epochs to consider in sliding window
            rel_window_position: Relative position of window (-1 to 1, 0 is centered)
            correlation_threshold: Correlation threshold for epoch selection
            plot_artifacts: Whether to plot averaged artifacts (for debugging)
            realign_after_averaging: Realign triggers after computing averages
            search_window_factor: Search window for realignment (multiplier of upsampling)
        """
        self.window_size = window_size
        self.rel_window_position = rel_window_position
        self.correlation_threshold = correlation_threshold
        self.plot_artifacts = plot_artifacts
        self.realign_after_averaging = realign_after_averaging
        self.search_window_factor = search_window_factor
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate prerequisites."""
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector first."
            )
        if len(context.get_triggers()) < self.window_size:
            logger.warning(
                f"Number of triggers ({len(context.get_triggers())}) is less than "
                f"window size ({self.window_size}). Using smaller window."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        logger.info("Applying Averaged Artifact Subtraction (AAS)")

        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info['sfreq']
        upsampling_factor = context.metadata.upsampling_factor

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
        )

        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found, skipping AAS")
            return context

        tmin = context.metadata.artifact_to_trigger_offset
        tmax = tmin + (artifact_length / sfreq)

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

        if len(epochs) != len(triggers):
            logger.warning(
                f"Number of epochs ({len(epochs)}) != number of triggers ({len(triggers)}). "
                "Data may be incomplete."
            )

        logger.debug(f"Computing averaging matrices for {len(eeg_channels)} channels")
        averaging_matrices = {}

        with processor_progress(
            total=len(eeg_channels) or None,
            message="Averaging matrices",
        ) as progress:
            for idx, ch_idx in enumerate(eeg_channels):
                ch_name = raw.ch_names[ch_idx]
                logger.debug(f"  Channel {ch_idx}: {ch_name}")

                epochs_single_channel = np.squeeze(epochs.get_data(copy=False)[:, idx, :])

                avg_matrix = self._calc_averaging_matrix(
                    epochs_single_channel,
                    window_size=self.window_size,
                    rel_window_offset=self.rel_window_position,
                    correlation_threshold=self.correlation_threshold
                )

                averaging_matrices[ch_idx] = avg_matrix
                progress.advance(
                    1,
                    message=f"{idx + 1}/{len(eeg_channels)} • {ch_name}",
                )

        logger.debug("Computing averaged artifacts")
        artifacts_per_channel = self._calc_averaged_artifacts(
            raw,
            averaging_matrices,
            triggers,
            artifact_length,
            context.metadata.artifact_to_trigger_offset,
            sfreq
        )

        if self.plot_artifacts and artifacts_per_channel:
            try:
                import random
                from matplotlib import pyplot as plt

                processed_channels = list(averaging_matrices.keys())
                random_ch_list_idx = random.randint(0, len(processed_channels) - 1)

                ch_idx = processed_channels[random_ch_list_idx]
                ch_name = raw.ch_names[ch_idx]

                artifacts = artifacts_per_channel[random_ch_list_idx]

                if len(artifacts) > 0:
                    random_epoch_idx = random.randint(0, len(artifacts) - 1)
                    artifact_segment = artifacts[random_epoch_idx]

                    logger.info(f"Plotting random artifact for channel {ch_name}, epoch {random_epoch_idx}")

                    plt.figure(figsize=(10, 4))
                    plt.plot(artifact_segment)
                    plt.title(f"Estimated Artifact: Channel {ch_name} (Epoch {random_epoch_idx})")
                    plt.xlabel("Samples")
                    plt.ylabel("Amplitude")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.show()
            except Exception as e:
                logger.warning(f"Failed to plot random artifact: {e}")

        if self.realign_after_averaging:
            logger.debug("Realigning triggers to averaged artifacts")
            search_window = int(self.search_window_factor * upsampling_factor)
            first_channel_data = raw.get_data(picks=[list(averaging_matrices.keys())[0]])[0]
            aligned_triggers = self._align_triggers_to_artifacts(
                first_channel_data,
                artifacts_per_channel[0],
                triggers,
                int(context.metadata.artifact_to_trigger_offset * sfreq),
                artifact_length,
                search_window
            )
        else:
            aligned_triggers = triggers

        logger.info(f"Removing artifacts from {len(eeg_channels)} channels")
        raw_corrected = raw.copy()

        if not context.has_estimated_noise():
            estimated_noise = np.zeros(raw._data.shape)
        else:
            estimated_noise = context.get_estimated_noise().copy()

        smin = int(context.metadata.artifact_to_trigger_offset * sfreq)
        smax = smin + artifact_length

        with processor_progress(
            total=len(averaging_matrices) or None,
            message="Removing artifacts",
        ) as progress:
            for ch_list_idx, ch_idx in enumerate(averaging_matrices.keys()):
                ch_name = raw.ch_names[ch_idx]
                logger.debug(f"  Removing artifacts from {ch_name}")

                artifacts = artifacts_per_channel[ch_list_idx]

                for epoch_idx, trigger_pos in enumerate(aligned_triggers):
                    start = trigger_pos + smin
                    stop = min(trigger_pos + smax, raw_corrected._data.shape[1])

                    if start < 0 or start >= raw_corrected._data.shape[1]:
                        continue

                    artifact_segment = artifacts[epoch_idx, :stop - start]
                    raw_corrected._data[ch_idx, start:stop] -= artifact_segment
                    estimated_noise[ch_idx, start:stop] += artifact_segment

                progress.advance(
                    1,
                    message=f"{ch_name} cleaned ({ch_list_idx + 1}/{len(averaging_matrices)})",
                )

        new_context = context.with_raw(raw_corrected)
        new_context.set_estimated_noise(estimated_noise)

        if self.realign_after_averaging and not np.array_equal(aligned_triggers, triggers):
            new_metadata = new_context.metadata.copy()
            new_metadata.triggers = aligned_triggers
            new_context._metadata = new_metadata
            logger.info(f"Triggers realigned after averaging")

        logger.info("AAS correction completed")
        return new_context

    def _calc_averaging_matrix(
        self,
        epochs: np.ndarray,
        window_size: int,
        rel_window_offset: float,
        correlation_threshold: float
    ) -> np.ndarray:
        """
        Calculate averaging matrix using correlation-based epoch selection.

        For each epoch, finds highly correlated epochs within a sliding window
        and creates a weighted average.

        Args:
            epochs: Epochs array (n_epochs, n_times)
            window_size: Size of sliding window
            rel_window_offset: Relative offset of window
            correlation_threshold: Correlation threshold

        Returns:
            Averaging matrix (n_epochs, n_epochs) where each row sums to 1
        """
        n_epochs = len(epochs)
        averaging_matrix = np.zeros((n_epochs, n_epochs))

        # Calculate window offset in epoch units
        window_offset = int(window_size * rel_window_offset)

        # Sliding window over epochs
        for idx in range(0, n_epochs, window_size):
            offset_idx = idx + window_offset

            # Reference epochs (first 5 in window)
            reference_indices = np.arange(idx, min(idx + 5, n_epochs))

            # Candidate epochs to average
            candidates = np.arange(offset_idx, min(offset_idx + window_size, n_epochs))
            candidates = candidates[candidates >= 0]  # Remove negative indices

            # Find highly correlated epochs
            chosen = self._find_correlated_epochs(
                epochs,
                candidates,
                reference_indices,
                correlation_threshold
            )

            if len(chosen) == 0:
                # Fallback: use reference epochs if no correlated found
                chosen = reference_indices

            # Set weights (uniform average over chosen epochs)
            target_indices = np.arange(idx, min(idx + window_size, n_epochs))
            weight = 1.0 / len(chosen)
            averaging_matrix[np.ix_(target_indices, chosen)] = weight

        return averaging_matrix

    def _find_correlated_epochs(
        self,
        all_epochs: np.ndarray,
        candidate_indices: np.ndarray,
        reference_indices: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Find epochs that are highly correlated with reference epochs.

        Iteratively adds epochs that correlate well with the running average.

        Args:
            all_epochs: All epochs (n_epochs, n_times)
            candidate_indices: Indices of candidate epochs to check
            reference_indices: Indices of reference epochs
            threshold: Correlation threshold

        Returns:
            Array of selected epoch indices
        """
        if len(reference_indices) == 0:
            return np.array([])

        # Start with reference epochs
        sum_data = np.sum(all_epochs[reference_indices], axis=0)
        chosen = list(reference_indices)

        # Check each candidate
        for idx in candidate_indices:
            if idx in chosen:
                continue

            # Current average
            avg_data = sum_data / len(chosen)

            # Correlation with candidate
            corr = np.corrcoef(avg_data.squeeze(), all_epochs[idx].squeeze())[0, 1]

            if corr > threshold:
                sum_data += all_epochs[idx]
                chosen.append(idx)

        return np.array(chosen)

    def _calc_averaged_artifacts(
        self,
        raw: mne.io.Raw,
        averaging_matrices: Dict[int, np.ndarray],
        triggers: np.ndarray,
        artifact_length: int,
        artifact_offset: float,
        sfreq: float
    ) -> list:
        """
        Calculate averaged artifacts for each channel.

        Args:
            raw: Raw EEG data
            averaging_matrices: Averaging matrix for each channel
            triggers: Trigger positions
            artifact_length: Artifact length in samples
            artifact_offset: Artifact offset in seconds
            sfreq: Sampling frequency

        Returns:
            List of averaged artifacts per channel (list of n_epochs x n_times arrays)
        """
        artifacts_per_channel = []
        trigger_offset_samples = int(artifact_offset * sfreq)

        for ch_idx, avg_matrix in averaging_matrices.items():
            # Get channel data (zero-mean for better averaging)
            ch_data = raw._data[ch_idx]
            ch_data_zero_mean = ch_data - np.mean(ch_data)

            # Split into epochs
            epoch_data = split_vector(
                ch_data_zero_mean,
                triggers + trigger_offset_samples,
                artifact_length
            )

            # Handle case where epochs don't match matrix size
            while len(epoch_data) > len(avg_matrix):
                epoch_data = epoch_data[:-1]

            # Calculate averaged artifacts using matrix multiplication
            averaged_artifacts = np.dot(avg_matrix, epoch_data)

            # Handle case where matrix size doesn't match triggers
            if len(averaged_artifacts) < len(triggers):
                # Pad with last artifact
                last_artifact = averaged_artifacts[-1].reshape(1, -1)
                padding_needed = len(triggers) - len(averaged_artifacts)
                padding = np.repeat(last_artifact, padding_needed, axis=0)
                averaged_artifacts = np.vstack([averaged_artifacts, padding])

            artifacts_per_channel.append(averaged_artifacts)

        return artifacts_per_channel

    def _align_triggers_to_artifacts(
        self,
        channel_data: np.ndarray,
        artifacts: np.ndarray,
        triggers: np.ndarray,
        smin: int,
        smax: int,
        search_window: int
    ) -> np.ndarray:
        """
        Align triggers to averaged artifacts using cross-correlation.

        Args:
            channel_data: Single channel data
            artifacts: Averaged artifacts (n_epochs, n_times)
            triggers: Original trigger positions
            smin: Start offset
            smax: End offset (relative to trigger)
            search_window: Search window size

        Returns:
            Aligned trigger positions
        """
        aligned_triggers = []

        for i, trigger in enumerate(triggers):
            # Get data segment with search window
            start = trigger + smin
            stop = trigger + smax + search_window

            if stop > len(channel_data):
                aligned_triggers.append(trigger)
                continue

            segment = channel_data[start:stop]
            artifact = artifacts[i, :]

            # Find best alignment using cross-correlation
            corr = crosscorrelation(segment, artifact, search_window)
            best_shift = np.argmax(corr) - search_window

            aligned_triggers.append(trigger + best_shift)

        return np.array(aligned_triggers)


# Alias for backwards compatibility
AveragedArtifactSubtraction = AASCorrection
