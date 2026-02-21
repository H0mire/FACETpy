"""
Trigger Alignment Processors Module

This module contains processors for aligning triggers to artifact positions.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, Tuple
import mne
import numpy as np
from loguru import logger
from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError
from ..helpers.crosscorr import crosscorrelation


def _get_pre_post_samples(
    context: ProcessingContext,
    artifact_length: int,
) -> Tuple[int, int]:
    """Derive pre/post trigger sample lengths using metadata."""
    sfreq = context.get_raw().info["sfreq"]
    metadata = context.metadata

    if artifact_length is None or artifact_length <= 0:
        raise ProcessorValidationError("Artifact length must be positive for alignment")

    if metadata.pre_trigger_samples is not None:
        pre = int(max(0, min(metadata.pre_trigger_samples, artifact_length)))
    else:
        offset_samples = int(round(metadata.artifact_to_trigger_offset * sfreq))
        pre = int(max(0, min(-offset_samples, artifact_length))) if offset_samples < 0 else 0

    if metadata.post_trigger_samples is not None:
        post = int(max(0, min(metadata.post_trigger_samples, artifact_length)))
    else:
        post = int(max(artifact_length - pre, 0))

    if pre + post == 0:
        post = artifact_length
    if pre + post < artifact_length:
        post = artifact_length - pre

    return pre, post


def _extract_epoch_with_padding(
    data: np.ndarray,
    start_idx: int,
    length: int,
    total_length: int,
) -> np.ndarray:
    """
    Extract a window from the data. Pads with edge values when the window
    exceeds the available samples.
    """
    end_idx = start_idx + length

    pad_left = max(0, -start_idx)
    pad_right = max(0, end_idx - total_length)

    valid_start = max(0, start_idx)
    valid_end = min(total_length, end_idx)

    segment = data[valid_start:valid_end]
    if pad_left or pad_right:
        segment = np.pad(segment, (pad_left, pad_right), mode="edge")

    if len(segment) != length:
        # As a fallback, resize by repeating edge values to the required length
        if len(segment) == 0:
            segment = np.zeros(length, dtype=data.dtype)
        else:
            segment = np.resize(segment, length)

    return segment


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
        logger.info("Aligning triggers using cross-correlation")

        raw = context.get_raw()
        triggers = context.get_triggers().copy()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info['sfreq']
        upsampling_factor = context.metadata.upsampling_factor

        if self.ref_channel is None:
            eeg_channels = mne.pick_types(
                raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads'
            )
            ref_channel = eeg_channels[0] if len(eeg_channels) > 0 else 0
        else:
            ref_channel = self.ref_channel

        if self.search_window is None:
            search_window = 3 * upsampling_factor
        else:
            search_window = self.search_window

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

        tmin = int(context.metadata.artifact_to_trigger_offset * working_raw.info['sfreq'])
        tmax = tmin + (artifact_length * upsampling_factor if needed_to_upsample else artifact_length)

        ref_data = working_raw.get_data(picks=[ref_channel])[0]

        ref_trigger = working_triggers[self.ref_trigger_index]
        ref_artifact = ref_data[ref_trigger + tmin:ref_trigger + tmax]

        logger.debug(f"Using trigger {self.ref_trigger_index} as reference")
        logger.debug(f"Reference artifact shape: {ref_artifact.shape}")

        aligned_triggers = working_triggers.copy()
        for i, trigger in enumerate(working_triggers):
            if i == self.ref_trigger_index:
                continue

            current_artifact = ref_data[
                trigger + tmin:trigger + tmax + search_window
            ]

            corr = self._cross_correlation(current_artifact, ref_artifact, search_window)
            max_idx = np.argmax(corr)
            shift = max_idx - search_window

            aligned_triggers[i] = trigger + shift

        logger.info(f"Aligned {len(aligned_triggers)} triggers")

        if needed_to_upsample:
            aligned_triggers = (aligned_triggers / upsampling_factor).astype(int)

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
        from ..helpers.crosscorr import crosscorrelation
        return crosscorrelation(signal, template, search_window)


@register_processor
class SliceAligner(TriggerAligner):
    """Align slices on already upsampled data (skips internal upsampling)."""

    name = "slice_aligner"
    description = "Align slice triggers on upsampled data"

    def __init__(
        self,
        ref_trigger_index: int = 0,
        ref_channel: Optional[int] = None,
        search_window: Optional[int] = None,
        save_to_annotations: bool = False,
    ):
        super().__init__(
            ref_trigger_index=ref_trigger_index,
            ref_channel=ref_channel,
            search_window=search_window,
            save_to_annotations=save_to_annotations,
            upsample_for_alignment=False,
        )


@register_processor
class SubsampleAligner(Processor):
    """
    Refine trigger positions at subsample precision (post-upsampling).

    This implementation mirrors the intent of the MATLAB RAAlignSubSample step
    while operating directly on the already upsampled data.
    """

    name = "subsample_aligner"
    description = "Refine trigger alignment at subsample precision"
    requires_triggers = True

    def __init__(
        self,
        ref_trigger_index: int = 0,
        ref_channel: Optional[int] = None,
        search_window: Optional[int] = None,
        apply_to_raw: bool = False,
    ):
        self.ref_trigger_index = ref_trigger_index
        self.ref_channel = ref_channel
        self.search_window = search_window
        self.apply_to_raw = apply_to_raw
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector before SubsampleAligner."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        logger.info("Refining trigger alignment with subsample precision")

        raw = context.get_raw()
        triggers = context.get_triggers().copy()
        artifact_length = context.get_artifact_length()
        n_samples = raw.n_times
        upsampling_factor = max(1, context.metadata.upsampling_factor)

        if len(triggers) == 0:
            logger.warning("SubsampleAligner received empty triggers; skipping")
            return context

        if self.ref_trigger_index >= len(triggers):
            raise ProcessorValidationError(
                f"Reference trigger index {self.ref_trigger_index} "
                f"is out of range for {len(triggers)} triggers"
            )

        pre_samples, post_samples = _get_pre_post_samples(context, artifact_length)
        window_length = pre_samples + post_samples

        if window_length <= 0:
            raise ProcessorValidationError("SubsampleAligner window length is zero")

        if self.search_window is not None:
            search_radius = int(max(1, self.search_window))
        else:
            search_radius = int(max(1, upsampling_factor * 2))

        if self.ref_channel is None:
            eeg_channels = mne.pick_types(
                raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
            )
            ref_channel = eeg_channels[0] if len(eeg_channels) else 0
        else:
            ref_channel = self.ref_channel

        ref_signal = raw.get_data(picks=[ref_channel])[0]
        ref_trigger = triggers[self.ref_trigger_index]
        ref_epoch = _extract_epoch_with_padding(
            ref_signal,
            ref_trigger - pre_samples,
            window_length,
            n_samples,
        )

        shifts = np.zeros(len(triggers), dtype=int)

        for idx, trigger in enumerate(triggers):
            if idx == self.ref_trigger_index:
                continue

            segment_start = trigger - pre_samples - search_radius
            segment_length = window_length + 2 * search_radius

            segment = _extract_epoch_with_padding(
                ref_signal,
                segment_start,
                segment_length,
                n_samples,
            )

            corr = crosscorrelation(segment, ref_epoch, search_radius)
            corr = np.nan_to_num(corr)
            shift = int(np.argmax(corr) - search_radius)
            shifts[idx] = shift

        aligned_triggers = np.clip(triggers + shifts, 0, n_samples - 1).astype(int)

        if self.apply_to_raw and np.any(shifts):
            logger.debug("Applying subsample shifts to raw data segments")
            data = raw.get_data()
            for idx, shift in enumerate(shifts):
                if shift == 0:
                    continue
                trigger = triggers[idx]
                window_start = max(0, trigger - pre_samples)
                window_end = min(n_samples, trigger + post_samples)
                segment = data[:, window_start:window_end]
                data[:, window_start:window_end] = np.roll(segment, shift, axis=1)
            raw._data = data

        new_metadata = context.metadata.copy()
        new_metadata.triggers = aligned_triggers
        new_metadata.custom.setdefault("subsample_alignment", {}).update(
            {
                "shifts": shifts.tolist(),
                "ref_trigger_index": self.ref_trigger_index,
                "search_window": search_radius,
            }
        )

        if new_metadata.pre_trigger_samples is None:
            new_metadata.pre_trigger_samples = pre_samples
        if new_metadata.post_trigger_samples is None:
            new_metadata.post_trigger_samples = post_samples

        return context.with_metadata(new_metadata)
