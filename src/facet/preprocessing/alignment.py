"""
Trigger Alignment Processors Module

This module contains processors for aligning triggers to artifact positions.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional
import mne
import numpy as np
from loguru import logger
from scipy.signal import fftconvolve, firls

from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError


@register_processor
class TriggerAligner(Processor):
    """
    Align triggers to artifact positions using cross-correlation.

    This processor refines trigger positions by aligning them to a reference
    artifact using cross-correlation.

    Example:
        aligner = TriggerAligner(ref_trigger_index=0)
        context = aligner.execute(context)
    """

    name = "trigger_aligner"
    description = "Align triggers using cross-correlation"
    requires_triggers = True

    def __init__(
        self,
        ref_trigger_index: int = 0,
        ref_channel: Optional[int] = None,
        search_window: Optional[int] = None,
        save_to_annotations: bool = False,
        upsample_for_alignment: bool = True
    ):
        """
        Initialize trigger aligner.

        Args:
            ref_trigger_index: Index of reference trigger to use as template
            ref_channel: Reference channel index (None for first EEG channel)
            search_window: Search window in samples (None for auto)
            save_to_annotations: Save aligned triggers as annotations
            upsample_for_alignment: Temporarily upsample for better alignment
        """
        self.ref_trigger_index = ref_trigger_index
        self.ref_channel = ref_channel
        self.search_window = search_window
        self.save_to_annotations = save_to_annotations
        self.upsample_for_alignment = upsample_for_alignment
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate prerequisites."""
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector first."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Align triggers."""
        logger.info("Aligning triggers using cross-correlation")

        raw = context.get_raw()
        triggers = context.get_triggers().copy()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info['sfreq']
        upsampling_factor = context.metadata.upsampling_factor

        # Get reference channel
        if self.ref_channel is None:
            eeg_channels = mne.pick_types(
                raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
            )
            ref_channel = eeg_channels[0] if len(eeg_channels) > 0 else 0
        else:
            ref_channel = self.ref_channel

        # Determine search window
        if self.search_window is None:
            search_window = 3 * upsampling_factor
        else:
            search_window = self.search_window

        # Check if we need to upsample
        working_raw = raw
        working_triggers = triggers
        needed_to_upsample = False

        if self.upsample_for_alignment and sfreq == context.get_raw_original().info['sfreq']:
            logger.debug("Temporarily upsampling for alignment")
            from .resampling import UpSample
            temp_context = UpSample(factor=upsampling_factor).execute(context)
            working_raw = temp_context.get_raw()
            working_triggers = temp_context.get_triggers()
            needed_to_upsample = True

        # Calculate sample offsets
        tmin = int(context.metadata.artifact_to_trigger_offset * working_raw.info['sfreq'])
        tmax = tmin + (artifact_length * upsampling_factor if needed_to_upsample else artifact_length)

        # Get data from reference channel
        ref_data = working_raw.get_data(picks=[ref_channel])[0]

        # Get reference artifact
        ref_trigger = working_triggers[self.ref_trigger_index]
        ref_artifact = ref_data[ref_trigger + tmin:ref_trigger + tmax]

        logger.debug(f"Using trigger {self.ref_trigger_index} as reference")
        logger.debug(f"Reference artifact shape: {ref_artifact.shape}")

        # Align all triggers
        aligned_triggers = working_triggers.copy()
        for i, trigger in enumerate(working_triggers):
            if i == self.ref_trigger_index:
                continue

            # Get current artifact with search window
            current_artifact = ref_data[
                trigger + tmin:trigger + tmax + search_window
            ]

            # Find max cross-correlation
            corr = self._cross_correlation(current_artifact, ref_artifact, search_window)
            max_idx = np.argmax(corr)
            shift = max_idx - search_window

            aligned_triggers[i] = trigger + shift

        logger.info(f"Aligned {len(aligned_triggers)} triggers")

        # Convert back to original sampling rate if upsampled
        if needed_to_upsample:
            aligned_triggers = (aligned_triggers / upsampling_factor).astype(int)

        # Update metadata
        new_metadata = context.metadata.copy()
        new_metadata.triggers = aligned_triggers

        # Recalculate artifact length
        if len(aligned_triggers) > 1:
            trigger_diffs = np.diff(aligned_triggers)
            if new_metadata.volume_gaps:
                mean_val = np.mean([np.median(trigger_diffs), np.max(trigger_diffs)])
                slice_diffs = trigger_diffs[trigger_diffs < mean_val]
                new_metadata.artifact_length = int(np.max(slice_diffs))
            else:
                new_metadata.artifact_length = int(np.max(trigger_diffs))

        # Save to annotations if requested
        if self.save_to_annotations:
            raw_copy = raw.copy()
            raw_copy.set_annotations(
                mne.Annotations(
                    onset=aligned_triggers / sfreq,
                    duration=np.zeros(len(aligned_triggers)),
                    description=["Aligned_Trigger"] * len(aligned_triggers)
                )
            )
            return context.with_raw(raw_copy).with_metadata(new_metadata)

        return context.with_metadata(new_metadata)

    def _cross_correlation(
        self,
        signal: np.ndarray,
        template: np.ndarray,
        search_window: int
    ) -> np.ndarray:
        """Calculate cross-correlation."""
        from ..helpers.crosscorr import crosscorrelation
        return crosscorrelation(signal, template, search_window)


@register_processor
class SubsampleAligner(Processor):
    """
    Align triggers at subsample precision using phase shifts.

    This processor uses FFT-based phase shifting to achieve subsample-level
    alignment of artifacts.

    Example:
        aligner = SubsampleAligner(ref_trigger_index=0)
        context = aligner.execute(context)
    """

    name = "subsample_aligner"
    description = "Align triggers at subsample precision"
    requires_triggers = True

    def __init__(
        self,
        ref_trigger_index: int = 0,
        ref_channel: Optional[int] = None,
        hp_frequency: Optional[float] = None,
        max_iterations: int = 15
    ):
        """
        Initialize subsample aligner.

        Args:
            ref_trigger_index: Index of reference trigger
            ref_channel: Reference channel index (None for first EEG)
            hp_frequency: Highpass frequency for preprocessing (None for auto)
            max_iterations: Maximum iterations for optimization
        """
        self.ref_trigger_index = ref_trigger_index
        self.ref_channel = ref_channel
        self.hp_frequency = hp_frequency
        self.max_iterations = max_iterations
        self._subsample_shifts = None
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate prerequisites."""
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector first."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Align at subsample precision."""
        logger.info("Aligning triggers at subsample precision")

        raw = context.get_raw()
        triggers = context.get_triggers()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info['sfreq']
        upsampling_factor = context.metadata.upsampling_factor

        # Get reference channel
        if self.ref_channel is None:
            eeg_channels = mne.pick_types(
                raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
            )
            ref_channel = eeg_channels[0] if len(eeg_channels) > 0 else 0
        else:
            ref_channel = self.ref_channel

        # Calculate offsets
        tmin = int(context.metadata.artifact_to_trigger_offset * sfreq)

        # Determine maximum trigger distance for FFT
        max_trig_dist = np.max(np.diff(triggers))
        num_samples = max_trig_dist + 20

        # Get acquisition data
        acq_start = max(0, triggers[0] - artifact_length)
        acq_end = min(raw.n_times, triggers[-1] + artifact_length)
        acq_data = raw.get_data(picks=[ref_channel], start=acq_start, stop=acq_end)[0]

        # Apply highpass filter if specified
        if self.hp_frequency and self.hp_frequency > 0:
            logger.debug(f"Applying {self.hp_frequency}Hz highpass for alignment")
            nyq = 0.5 * sfreq
            f = [
                0,
                (self.hp_frequency * 0.9) / (nyq * upsampling_factor),
                (self.hp_frequency * 1.1) / (nyq * upsampling_factor),
                1,
            ]
            a = [0, 0, 1, 1]
            fw = firls(101, f, a)
            hpeeg = fftconvolve(acq_data, fw, mode='same')
            hpeeg = np.concatenate([hpeeg[100:], np.zeros(100)])
        else:
            hpeeg = acq_data

        # Split into epochs
        from ..helpers.utils import split_vector
        adjusted_triggers = triggers - acq_start + tmin - 10
        eeg_matrix = split_vector(hpeeg, adjusted_triggers, num_samples)

        # Get reference epoch
        eeg_ref = eeg_matrix[self.ref_trigger_index, :]

        # Calculate phase shifts
        shift_angles = (
            np.arange(1, num_samples + 1) - np.floor(num_samples / 2) + 1
        ) / num_samples

        self._subsample_shifts = np.zeros(len(triggers))
        corrs = np.zeros((len(triggers), self.max_iterations))
        shifts = np.zeros((len(triggers), self.max_iterations))

        # Calculate shifts for each epoch
        for epoch_idx in range(len(triggers)):
            if epoch_idx == self.ref_trigger_index:
                continue

            eeg_m = eeg_matrix[epoch_idx, :]
            fft_m = np.fft.fftshift(np.fft.fft(eeg_m))

            # Initial shifts
            shift_l, shift_m, shift_r = -1, 0, 1

            # Calculate initial correlations
            fft_l = fft_m * np.exp(-1j * 2 * np.pi * shift_angles * shift_l)
            fft_r = fft_m * np.exp(-1j * 2 * np.pi * shift_angles * shift_r)
            eeg_l = np.real(np.fft.ifft(np.fft.ifftshift(fft_l)))
            eeg_r = np.real(np.fft.ifft(np.fft.ifftshift(fft_r)))

            corr_l = self._compare(eeg_ref, eeg_l)
            corr_m = self._compare(eeg_ref, eeg_m)
            corr_r = self._compare(eeg_ref, eeg_r)

            fft_ori = fft_m

            # Iterative optimization
            for iteration in range(self.max_iterations):
                corrs[epoch_idx, iteration] = corr_m
                shifts[epoch_idx, iteration] = shift_m

                # Update bounds
                if corr_l > corr_r:
                    corr_r, eeg_r, fft_r, shift_r = corr_m, eeg_m, fft_m, shift_m
                else:
                    corr_l, eeg_l, fft_l, shift_l = corr_m, eeg_m, fft_m, shift_m

                # Calculate new midpoint
                shift_m = (shift_l + shift_r) / 2
                fft_m = fft_ori * np.exp(-1j * 2 * np.pi * shift_angles * shift_m)
                eeg_m = np.real(np.fft.ifft(np.fft.ifftshift(fft_m)))
                corr_m = self._compare(eeg_ref, eeg_m)

            self._subsample_shifts[epoch_idx] = shift_m

        logger.info("Subsample alignment completed")

        # Store shifts in metadata for application during correction
        new_metadata = context.metadata.copy()
        new_metadata.custom['subsample_shifts'] = self._subsample_shifts

        return context.with_metadata(new_metadata)

    def _compare(self, ref: np.ndarray, arg: np.ndarray) -> float:
        """Compare two signals (negative sum of squared differences)."""
        return -np.sum((ref - arg) ** 2)
