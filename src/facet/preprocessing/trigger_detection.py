"""Trigger Detection Processors Module

Processors for detecting triggers and events in EEG data recorded during
simultaneous EEG-fMRI acquisition.
"""

from typing import Optional, List
import re

import mne
import numpy as np
from loguru import logger

from ..core import (
    Processor,
    ProcessingContext,
    ProcessorError,
    register_processor,
    ProcessorValidationError,
)
from ..helpers.crosscorr import crosscorrelation


@register_processor
class TriggerDetector(Processor):
    """Detect triggers from annotations or stim channels using a regex pattern.

    Searches the raw data for events whose description (annotation) or integer
    value (stim channel) matches the supplied regular expression. Detected
    trigger sample positions are stored in ``context.metadata.triggers``.

    The artifact length is estimated from the median inter-trigger interval;
    volume-level gaps in slice-triggered acquisitions are detected
    automatically.

    Parameters
    ----------
    regex : str
        Regular expression pattern to match trigger values.
    save_to_annotations : bool, optional
        If ``True``, write detected triggers back to the raw annotations
        (default: False).
    """

    name = "trigger_detector"
    description = "Detect triggers using regex pattern"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, regex: str, save_to_annotations: bool = False) -> None:
        self.regex = regex
        self.save_to_annotations = save_to_annotations
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        sfreq = raw.info["sfreq"]

        # --- LOG ---
        logger.info("Detecting triggers with pattern: {}", self.regex)

        # --- COMPUTE ---
        filtered_events = self._find_events(raw)

        if len(filtered_events) == 0:
            logger.warning("No triggers found!")
            return context

        triggers = np.array([event[0] for event in filtered_events])
        logger.info("Found {} triggers", len(triggers))

        artifact_meta = self._compute_artifact_metadata(triggers)

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.triggers = triggers
        new_metadata.trigger_regex = self.regex
        new_metadata.artifact_length = artifact_meta["artifact_length"]
        new_metadata.volume_gaps = artifact_meta["volume_gaps"]
        if artifact_meta.get("slices_per_volume") is not None:
            new_metadata.slices_per_volume = artifact_meta["slices_per_volume"]

        logger.debug("Artifact length: {} samples", new_metadata.artifact_length)
        logger.debug("Volume gaps: {}", new_metadata.volume_gaps)

        if self.save_to_annotations:
            raw_copy = raw.copy()
            raw_copy.set_annotations(
                mne.Annotations(
                    onset=triggers / sfreq,
                    duration=np.zeros(len(triggers)),
                    description=["Trigger"] * len(triggers),
                )
            )
            return context.with_raw(raw_copy).with_metadata(new_metadata)

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    def _find_events(self, raw: mne.io.Raw) -> List:
        """Search stim channels then annotations for matching events.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw object to search.

        Returns
        -------
        list
            List of MNE-style event rows ``[sample, prev_id, event_id]``.
        """
        pattern = re.compile(self.regex)
        stim_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)

        if len(stim_channels) > 0:
            logger.debug("Found {} stim channels", len(stim_channels))
            events = mne.find_events(
                raw,
                stim_channel=raw.ch_names[stim_channels[0]],
                initial_event=True,
                verbose=False,
            )
            return [event for event in events if pattern.search(str(event[2]))]

        logger.debug("No stim channels, searching annotations")
        events_obj = mne.events_from_annotations(raw, verbose=False)
        logger.debug("Event types: {}", events_obj[1])
        return list(mne.events_from_annotations(raw, regexp=self.regex, verbose=False)[0])

    def _compute_artifact_metadata(self, triggers: np.ndarray) -> dict:
        """Estimate artifact length and detect volume gaps from trigger spacing.

        Parameters
        ----------
        triggers : np.ndarray
            Detected trigger sample positions.

        Returns
        -------
        dict
            Keys: ``artifact_length``, ``volume_gaps``, optionally
            ``slices_per_volume``.
        """
        if len(triggers) <= 1:
            return {"artifact_length": None, "volume_gaps": False}

        trigger_diffs = np.diff(triggers)
        ptp = np.ptp(trigger_diffs)

        if ptp > 3:
            return self._compute_slice_volume_metadata(triggers, trigger_diffs)

        return {
            "artifact_length": int(np.max(trigger_diffs)),
            "volume_gaps": False,
        }

    def _compute_slice_volume_metadata(
        self, triggers: np.ndarray, trigger_diffs: np.ndarray
    ) -> dict:
        """Compute metadata when volume-level gaps are present.

        Parameters
        ----------
        triggers : np.ndarray
            All trigger sample positions.
        trigger_diffs : np.ndarray
            Differences between consecutive triggers.

        Returns
        -------
        dict
            Keys: ``artifact_length``, ``volume_gaps``, ``slices_per_volume``.
        """
        mean_val = np.mean([np.median(trigger_diffs), np.max(trigger_diffs)])
        slice_diffs = trigger_diffs[trigger_diffs < mean_val]
        artifact_length = int(np.max(slice_diffs))

        gap_indices = np.where(trigger_diffs >= mean_val)[0]
        slices_per_volume = None

        if len(gap_indices) > 0:
            slice_counts = []
            last_idx = -1
            for idx in gap_indices:
                slice_counts.append(idx - last_idx)
                last_idx = idx
            if last_idx < len(triggers) - 1:
                slice_counts.append(len(triggers) - 1 - last_idx)
            if slice_counts:
                slices_per_volume = int(np.median(slice_counts))
                logger.info("Estimated slices per volume: {}", slices_per_volume)

        return {
            "artifact_length": artifact_length,
            "volume_gaps": True,
            "slices_per_volume": slices_per_volume,
        }


@register_processor
class QRSTriggerDetector(Processor):
    """Detect QRS complexes (heartbeats) for BCG artifact correction.

    Uses the FMRIB QRS detector from ``facet.helpers.bcg_detector``, which
    requires the ``neurokit2`` package (available via ``pip install
    facetpy[all]``).

    The artifact length is set to half the median RR interval, centred on each
    detected R-peak.

    Parameters
    ----------
    save_to_annotations : bool, optional
        If ``True``, write detected QRS peaks back to the raw annotations
        (default: False).
    """

    name = "qrs_trigger_detector"
    description = "Detect QRS complexes for BCG correction"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(self, save_to_annotations: bool = False) -> None:
        self.save_to_annotations = save_to_annotations
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        sfreq = raw.info["sfreq"]

        # --- LOG ---
        logger.info("Detecting QRS complexes")

        # --- COMPUTE ---
        try:
            from ..helpers import bcg_detector
        except ImportError:
            raise ProcessorError(
                "neurokit2 is required for QRSTriggerDetector. "
                "Install with: pip install facetpy[all]"
            )

        peaks = bcg_detector.fmrib_qrsdetect(raw)
        triggers = np.array(peaks, dtype=np.int32)
        logger.info("Found {} QRS peaks", len(triggers))

        # --- BUILD RESULT ---
        new_metadata = context.metadata.copy()
        new_metadata.triggers = triggers
        new_metadata.trigger_regex = "QRS"
        new_metadata.volume_gaps = True  # QRS peaks have variable spacing

        if len(triggers) > 1:
            rr_intervals = np.diff(triggers)
            median_rr = int(np.median(rr_intervals))
            new_metadata.artifact_length = median_rr // 2
            new_metadata.artifact_to_trigger_offset = (
                -new_metadata.artifact_length / (2 * sfreq)
            )

        if self.save_to_annotations:
            raw_copy = raw.copy()
            raw_copy.set_annotations(
                mne.Annotations(
                    onset=triggers / sfreq,
                    duration=np.zeros(len(triggers)),
                    description=["QRS"] * len(triggers),
                )
            )
            return context.with_raw(raw_copy).with_metadata(new_metadata)

        # --- RETURN ---
        return context.with_metadata(new_metadata)


@register_processor
class MissingTriggerDetector(Processor):
    """Detect and insert missing triggers by template matching.

    Scans the trigger sequence for gaps larger than 1.9× the artifact length
    and attempts to locate missing artifact epochs by cross-correlating a
    reference template against the signal. Optionally extends the search one
    step before the first trigger and one step after the last.

    Parameters
    ----------
    add_to_context : bool, optional
        If ``True``, insert found triggers into metadata and annotations
        (default: True).
    correlation_threshold : float, optional
        Minimum absolute Pearson correlation to accept a candidate trigger
        (default: 0.9).
    search_window_factor : float, optional
        Search window as a fraction of artifact length (default: 0.1).
    ref_channel : int, optional
        Reference channel index for template matching (default: 0).
    """

    name = "missing_trigger_detector"
    description = "Detect and add missing triggers"
    version = "1.0.0"

    requires_triggers = True
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        add_to_context: bool = True,
        correlation_threshold: float = 0.9,
        search_window_factor: float = 0.1,
        ref_channel: int = 0,
    ) -> None:
        self.add_to_context = add_to_context
        self.correlation_threshold = correlation_threshold
        self.search_window_factor = search_window_factor
        self.ref_channel = ref_channel
        super().__init__()

    def validate(self, context: ProcessingContext) -> None:
        super().validate(context)
        if context.get_artifact_length() is None:
            raise ProcessorValidationError(
                "Artifact length not set. Run TriggerDetector first."
            )

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()
        triggers = context.get_triggers().copy()
        artifact_length = context.get_artifact_length()
        sfreq = raw.info["sfreq"]
        tmin = int(context.metadata.artifact_to_trigger_offset * sfreq)
        tmax = tmin + artifact_length

        # --- LOG ---
        logger.info("Searching for missing triggers")

        # --- COMPUTE ---
        ref_channel_data = raw.get_data(picks=[self.ref_channel])[0]
        template = self._build_template(ref_channel_data, triggers, tmin, tmax)
        search_window = int(self.search_window_factor * artifact_length)

        missing_triggers = self._find_missing_triggers(
            ref_channel_data, triggers, template, artifact_length, search_window, tmin, tmax
        )
        logger.info("Found {} missing triggers", len(missing_triggers))

        if len(missing_triggers) == 0:
            return context

        # --- BUILD RESULT ---
        if self.add_to_context:
            return self._build_context_with_missing(
                context, raw, triggers, missing_triggers, sfreq
            )

        new_metadata = context.metadata.copy()
        new_metadata.custom["missing_triggers"] = missing_triggers

        # --- RETURN ---
        return context.with_metadata(new_metadata)

    def _build_template(
        self,
        ref_data: np.ndarray,
        triggers: np.ndarray,
        tmin: int,
        tmax: int,
    ) -> np.ndarray:
        """Build a reference artifact template from the first few triggers.

        Parameters
        ----------
        ref_data : np.ndarray
            1-D reference channel signal.
        triggers : np.ndarray
            Trigger sample positions.
        tmin : int
            Start offset from trigger to artifact onset.
        tmax : int
            End offset from trigger to artifact end.

        Returns
        -------
        np.ndarray
            Averaged template epoch.
        """
        n_template = min(5, len(triggers))
        template_epochs = []
        for i in range(n_template):
            start = triggers[i] + tmin
            end = triggers[i] + tmax
            if end <= len(ref_data):
                template_epochs.append(ref_data[start:end])
        return np.mean(template_epochs, axis=0)

    def _find_missing_triggers(
        self,
        ref_data: np.ndarray,
        triggers: np.ndarray,
        template: np.ndarray,
        artifact_length: int,
        search_window: int,
        tmin: int,
        tmax: int,
    ) -> list:
        """Search for missing triggers before, within, and after the sequence.

        Parameters
        ----------
        ref_data : np.ndarray
            1-D reference channel signal.
        triggers : np.ndarray
            Known trigger positions.
        template : np.ndarray
            Reference artifact template.
        artifact_length : int
            Expected artifact length in samples.
        search_window : int
            Cross-correlation search radius.
        tmin : int
            Artifact onset offset from trigger.
        tmax : int
            Artifact end offset from trigger.

        Returns
        -------
        list of int
            Positions of detected missing triggers.
        """
        missing_triggers = []

        for i in range(len(triggers) - 1):
            gap = triggers[i + 1] - triggers[i]
            if gap > artifact_length * 1.9:
                search_pos = triggers[i] + artifact_length
                candidate = self._align_to_template(
                    ref_data, template, search_pos, search_window, tmin, tmax
                )
                if self._is_artifact(ref_data, template, candidate, tmin, artifact_length):
                    missing_triggers.append(candidate)

        search_pos = triggers[0] - artifact_length
        if search_pos > 0:
            candidate = self._align_to_template(
                ref_data, template, search_pos, search_window, tmin, tmax
            )
            if self._is_artifact(ref_data, template, candidate, tmin, artifact_length):
                missing_triggers.insert(0, candidate)

        search_pos = triggers[-1] + artifact_length
        if search_pos + tmax < len(ref_data):
            candidate = self._align_to_template(
                ref_data, template, search_pos, search_window, tmin, tmax
            )
            if self._is_artifact(ref_data, template, candidate, tmin, artifact_length):
                missing_triggers.append(candidate)

        return missing_triggers

    def _build_context_with_missing(
        self,
        context: ProcessingContext,
        raw: mne.io.Raw,
        triggers: np.ndarray,
        missing_triggers: list,
        sfreq: float,
    ) -> ProcessingContext:
        """Merge missing triggers into the context and add annotations.

        Parameters
        ----------
        context : ProcessingContext
            Current processing context.
        raw : mne.io.Raw
            Current raw object.
        triggers : np.ndarray
            Original trigger positions.
        missing_triggers : list of int
            Newly detected missing trigger positions.
        sfreq : float
            Sampling frequency in Hz.

        Returns
        -------
        ProcessingContext
            Updated context with merged triggers and annotations.
        """
        all_triggers = np.sort(np.concatenate([triggers, missing_triggers]))
        new_metadata = context.metadata.copy()
        new_metadata.triggers = all_triggers

        raw_copy = raw.copy()
        existing_annot = raw.annotations
        new_annot = mne.Annotations(
            onset=np.array(missing_triggers) / sfreq,
            duration=np.zeros(len(missing_triggers)),
            description=["missing_trigger"] * len(missing_triggers),
        )
        combined = mne.Annotations(
            onset=np.concatenate([existing_annot.onset, new_annot.onset]),
            duration=np.concatenate([existing_annot.duration, new_annot.duration]),
            description=list(existing_annot.description) + list(new_annot.description),
        )
        raw_copy.set_annotations(combined)
        return context.with_raw(raw_copy).with_metadata(new_metadata)

    def _align_to_template(
        self,
        data: np.ndarray,
        template: np.ndarray,
        position: int,
        search_window: int,
        tmin: int,
        tmax: int,
    ) -> int:
        """Find the best-matching position near ``position`` via cross-correlation.

        Parameters
        ----------
        data : np.ndarray
            1-D signal array.
        template : np.ndarray
            Reference artifact template.
        position : int
            Initial candidate trigger position.
        search_window : int
            Search radius in samples.
        tmin : int
            Artifact onset offset from trigger.
        tmax : int
            Artifact end offset from trigger.

        Returns
        -------
        int
            Refined trigger position.
        """
        segment = data[position + tmin : position + tmax + search_window]
        corr = crosscorrelation(segment, template, search_window)
        shift = int(np.argmax(corr)) - search_window
        return position + shift

    def _is_artifact(
        self,
        data: np.ndarray,
        template: np.ndarray,
        position: int,
        tmin: int,
        artifact_length: int,
    ) -> bool:
        """Return True if the segment at ``position`` correlates with the template.

        Parameters
        ----------
        data : np.ndarray
            1-D signal array.
        template : np.ndarray
            Reference artifact template.
        position : int
            Candidate trigger position.
        tmin : int
            Artifact onset offset from trigger.
        artifact_length : int
            Length of artifact in samples.

        Returns
        -------
        bool
            ``True`` if the Pearson correlation exceeds
            ``self.correlation_threshold``.
        """
        start = position + tmin
        end = start + min(artifact_length, len(template))

        if end > len(data):
            return False

        segment = data[start:end]
        template_segment = template[: len(segment)]

        if len(template_segment) < 3:
            return False

        try:
            corr = float(np.abs(np.corrcoef(segment, template_segment)[0, 1]))
            return corr > self.correlation_threshold
        except ValueError:
            return False
