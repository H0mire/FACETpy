"""
Trigger Detection Processors Module

This module contains processors for detecting triggers/events in EEG data.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional
import re
import mne
import numpy as np
from loguru import logger
from scipy.stats import pearsonr

from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError


@register_processor
class TriggerDetector(Processor):
    """
    Detect triggers/events based on regex pattern.

    Searches for triggers in annotations or stim channels.

    Example:
        detector = TriggerDetector(regex=r"\\b1\\b")
        context = detector.execute(context)
    """

    name = "trigger_detector"
    description = "Detect triggers using regex pattern"
    requires_triggers = False  # Creates triggers

    def __init__(
        self,
        regex: str,
        save_to_annotations: bool = False
    ):
        """
        Initialize trigger detector.

        Args:
            regex: Regular expression pattern to match trigger values
            save_to_annotations: Whether to save triggers as annotations in raw
        """
        self.regex = regex
        self.save_to_annotations = save_to_annotations
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Detect triggers."""
        logger.info(f"Detecting triggers with pattern: {self.regex}")

        raw = context.get_raw()
        pattern = re.compile(self.regex)

        # Try stim channels first
        stim_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)

        if len(stim_channels) > 0:
            logger.debug(f"Found {len(stim_channels)} stim channels")
            events = mne.find_events(
                raw,
                stim_channel=raw.ch_names[stim_channels[0]],
                initial_event=True,
                verbose=False
            )
            filtered_events = [
                event for event in events if pattern.search(str(event[2]))
            ]
        else:
            # Try annotations
            logger.debug("No stim channels, searching annotations")
            events_obj = mne.events_from_annotations(raw, verbose=False)
            logger.debug(f"Event types: {events_obj[1]}")
            filtered_events = mne.events_from_annotations(raw, regexp=self.regex, verbose=False)[0]

        if len(filtered_events) == 0:
            logger.warning("No triggers found!")
            return context

        # Extract trigger positions
        triggers = np.array([event[0] for event in filtered_events])

        logger.info(f"Found {len(triggers)} triggers")

        # Update metadata
        new_metadata = context.metadata.copy()
        new_metadata.triggers = triggers
        new_metadata.trigger_regex = self.regex

        # Derive artifact length from trigger spacing
        if len(triggers) > 1:
            trigger_diffs = np.diff(triggers)
            ptp = np.ptp(trigger_diffs)

            # Check for volume gaps (slice vs volume triggers)
            if ptp > 3:
                new_metadata.volume_gaps = True
                mean_val = np.mean([np.median(trigger_diffs), np.max(trigger_diffs)])
                slice_diffs = trigger_diffs[trigger_diffs < mean_val]
                new_metadata.artifact_length = int(np.max(slice_diffs))
            else:
                new_metadata.volume_gaps = False
                new_metadata.artifact_length = int(np.max(trigger_diffs))

            logger.debug(f"Artifact length: {new_metadata.artifact_length} samples")
            logger.debug(f"Volume gaps: {new_metadata.volume_gaps}")

        # Save to annotations if requested
        if self.save_to_annotations:
            raw_copy = raw.copy()
            raw_copy.set_annotations(
                mne.Annotations(
                    onset=triggers / raw.info['sfreq'],
                    duration=np.zeros(len(triggers)),
                    description=["Trigger"] * len(triggers)
                )
            )
            return context.with_raw(raw_copy).with_metadata(new_metadata)

        return context.with_metadata(new_metadata)


@register_processor
class QRSTriggerDetector(Processor):
    """
    Detect QRS complexes (heartbeats) for BCG artifact correction.

    Uses the BCG detector from helpers module.

    Example:
        detector = QRSTriggerDetector()
        context = detector.execute(context)
    """

    name = "qrs_trigger_detector"
    description = "Detect QRS complexes for BCG correction"
    requires_triggers = False

    def __init__(self, save_to_annotations: bool = False):
        """
        Initialize QRS detector.

        Args:
            save_to_annotations: Whether to save as annotations
        """
        self.save_to_annotations = save_to_annotations
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Detect QRS peaks."""
        from ..helpers import bcg_detector

        logger.info("Detecting QRS complexes")

        raw = context.get_raw()

        # Detect QRS peaks
        peaks = bcg_detector.fmrib_qrsdetect(raw)

        logger.info(f"Found {len(peaks)} QRS peaks")

        # Create events
        triggers = np.array(peaks, dtype=np.int32)

        # Update metadata
        new_metadata = context.metadata.copy()
        new_metadata.triggers = triggers
        new_metadata.trigger_regex = "QRS"
        new_metadata.volume_gaps = True  # QRS peaks have variable spacing

        # Estimate artifact length (half of median RR interval)
        if len(triggers) > 1:
            rr_intervals = np.diff(triggers)
            median_rr = int(np.median(rr_intervals))
            new_metadata.artifact_length = median_rr // 2
            new_metadata.artifact_to_trigger_offset = -new_metadata.artifact_length / (2 * raw.info['sfreq'])

        # Save to annotations if requested
        if self.save_to_annotations:
            raw_copy = raw.copy()
            raw_copy.set_annotations(
                mne.Annotations(
                    onset=triggers / raw.info['sfreq'],
                    duration=np.zeros(len(triggers)),
                    description=["QRS"] * len(triggers)
                )
            )
            return context.with_raw(raw_copy).with_metadata(new_metadata)

        return context.with_metadata(new_metadata)


@register_processor
class MissingTriggerDetector(Processor):
    """
    Detect and add missing triggers.

    Searches for gaps in trigger sequence and tries to find missing artifacts.

    Example:
        detector = MissingTriggerDetector(add_to_context=True)
        context = detector.execute(context)
    """

    name = "missing_trigger_detector"
    description = "Detect and add missing triggers"
    requires_triggers = True

    def __init__(
        self,
        add_to_context: bool = True,
        correlation_threshold: float = 0.9,
        search_window_factor: float = 0.1,
        ref_channel: int = 0
    ):
        """
        Initialize missing trigger detector.

        Args:
            add_to_context: Whether to add missing triggers to context
            correlation_threshold: Correlation threshold for artifact detection
            search_window_factor: Search window as fraction of artifact length
            ref_channel: Reference channel index for template matching
        """
        self.add_to_context = add_to_context
        self.correlation_threshold = correlation_threshold
        self.search_window_factor = search_window_factor
        self.ref_channel = ref_channel
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        """Validate prerequisites."""
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector first."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Detect missing triggers."""
        logger.info("Searching for missing triggers")

        raw = context.get_raw()
        triggers = context.get_triggers().copy()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info['sfreq']

        # Calculate offset in samples
        tmin = int(context.metadata.artifact_to_trigger_offset * sfreq)
        tmax = tmin + artifact_length

        search_window = int(self.search_window_factor * artifact_length)

        # Create template from first few triggers
        ref_channel_data = raw.get_data(picks=[self.ref_channel])[0]

        # Average first 5 triggers for template
        n_template = min(5, len(triggers))
        template_epochs = []
        for i in range(n_template):
            start = triggers[i] + tmin
            end = triggers[i] + tmax
            if end <= len(ref_channel_data):
                template_epochs.append(ref_channel_data[start:end])

        template = np.mean(template_epochs, axis=0)

        missing_triggers = []

        # Check for gaps in trigger sequence
        for i in range(len(triggers) - 1):
            gap = triggers[i + 1] - triggers[i]
            if gap > artifact_length * 1.9:  # Significant gap
                # Search for missing trigger
                search_pos = triggers[i] + artifact_length
                new_trigger = self._align_to_template(
                    ref_channel_data,
                    template,
                    search_pos,
                    search_window,
                    tmin,
                    tmax
                )

                # Validate with correlation
                if self._is_artifact(ref_channel_data, template, new_trigger, tmin, artifact_length):
                    missing_triggers.append(new_trigger)

        # Check beginning
        search_pos = triggers[0] - artifact_length
        if search_pos > 0:
            new_trigger = self._align_to_template(
                ref_channel_data,
                template,
                search_pos,
                search_window,
                tmin,
                tmax
            )
            if self._is_artifact(ref_channel_data, template, new_trigger, tmin, artifact_length):
                missing_triggers.insert(0, new_trigger)

        # Check end
        search_pos = triggers[-1] + artifact_length
        if search_pos + tmax < len(ref_channel_data):
            new_trigger = self._align_to_template(
                ref_channel_data,
                template,
                search_pos,
                search_window,
                tmin,
                tmax
            )
            if self._is_artifact(ref_channel_data, template, new_trigger, tmin, artifact_length):
                missing_triggers.append(new_trigger)

        logger.info(f"Found {len(missing_triggers)} missing triggers")

        if len(missing_triggers) == 0:
            return context

        # Add to context if requested
        if self.add_to_context:
            all_triggers = np.sort(np.concatenate([triggers, missing_triggers]))
            new_metadata = context.metadata.copy()
            new_metadata.triggers = all_triggers

            # Add annotations for missing triggers
            if self.add_to_context:
                raw_copy = raw.copy()
                existing_annot = raw.annotations
                new_annot = mne.Annotations(
                    onset=np.array(missing_triggers) / sfreq,
                    duration=np.zeros(len(missing_triggers)),
                    description=["missing_trigger"] * len(missing_triggers)
                )
                # Combine annotations
                combined = mne.Annotations(
                    onset=np.concatenate([existing_annot.onset, new_annot.onset]),
                    duration=np.concatenate([existing_annot.duration, new_annot.duration]),
                    description=list(existing_annot.description) + list(new_annot.description)
                )
                raw_copy.set_annotations(combined)
                return context.with_raw(raw_copy).with_metadata(new_metadata)

            return context.with_metadata(new_metadata)

        # Just return info about missing triggers in metadata
        new_metadata = context.metadata.copy()
        new_metadata.custom['missing_triggers'] = missing_triggers
        return context.with_metadata(new_metadata)

    def _align_to_template(
        self,
        data: np.ndarray,
        template: np.ndarray,
        position: int,
        search_window: int,
        tmin: int,
        tmax: int
    ) -> int:
        """Align position to template using cross-correlation."""
        segment = data[position + tmin:position + tmax + search_window]
        corr = self._cross_correlation(segment, template, search_window)
        max_idx = np.argmax(corr)
        shift = max_idx - search_window
        return position + shift

    def _cross_correlation(
        self,
        signal: np.ndarray,
        template: np.ndarray,
        search_window: int
    ) -> np.ndarray:
        """Calculate cross-correlation."""
        from ..helpers.crosscorr import crosscorrelation
        return crosscorrelation(signal, template, search_window)

    def _is_artifact(
        self,
        data: np.ndarray,
        template: np.ndarray,
        position: int,
        tmin: int,
        artifact_length: int
    ) -> bool:
        """Check if position contains an artifact."""
        start = position + tmin
        end = start + min(artifact_length, len(template))

        if end > len(data):
            return False

        segment = data[start:end]
        template_segment = template[:len(segment)]

        if len(template_segment) < 3:
            return False

        try:
            corr = np.abs(pearsonr(segment, template_segment)[0])
            return corr > self.correlation_threshold
        except:
            return False
