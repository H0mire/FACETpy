"""
Data Loaders Module

This module contains processors for loading EEG data from various formats.

Author: FACETpy Team
Date: 2025-01-12
"""

from numbers import Integral, Real
from pathlib import Path
from typing import Optional, List, Tuple

import mne
from mne.io import BaseRaw
from mne_bids import BIDSPath, read_raw_bids
from loguru import logger

from ..core import (
    Processor,
    ProcessingContext,
    ProcessingMetadata,
    ProcessorValidationError,
    register_processor,
)
from facet.logging_config import suppress_stdout


def _coerce_sample_index(value: Optional[float], default: int, name: str) -> int:
    """Convert a potentially optional numeric input into a valid sample index."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer-like value, got boolean")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if not float(value).is_integer():
            raise ValueError(f"{name} must be an integer number of samples")
        return int(value)
    raise ValueError(f"{name} must be an integer number of samples")


def _apply_sample_window(
    raw: BaseRaw,
    start_sample: Optional[int],
    stop_sample: Optional[int],
) -> Tuple[BaseRaw, int, int]:
    """Restrict a Raw object to the requested sample window."""
    n_times = raw.n_times
    if n_times == 0:
        raise ValueError("Cannot apply a sample window to an empty recording")

    start = _coerce_sample_index(start_sample, 0, "start_sample")
    stop = _coerce_sample_index(stop_sample, n_times, "stop_sample")

    if start < 0:
        raise ValueError("start_sample must be non-negative")
    if start >= n_times:
        raise ValueError(
            f"start_sample ({start}) is out of bounds for data with {n_times} samples"
        )
    if stop <= start:
        raise ValueError(
            f"stop_sample ({stop}) must be greater than start_sample ({start})"
        )
    if stop > n_times:
        raise ValueError(
            f"stop_sample ({stop}) exceeds total samples ({n_times})"
        )

    if start > 0 or stop < n_times:
        sfreq = raw.info["sfreq"]
        tmin = start / sfreq if start > 0 else None
        tmax = (stop - 1) / sfreq if stop < n_times else None
        raw.crop(tmin=tmin, tmax=tmax, verbose=False)

    return raw, start, stop


def _configure_bad_channels(raw: mne.io.Raw, bad_channels: List[str]) -> None:
    """Mark specified channels as bad in the Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        The Raw object to update in-place.
    bad_channels : list of str
        Channel names to mark as bad.
    """
    if not bad_channels:
        return

    existing = set(raw.ch_names)
    valid_bads = [ch for ch in bad_channels if ch in existing]
    missing = set(bad_channels) - existing

    if missing:
        logger.warning(
            "Skipping {} bad channel(s) not present in data: {}",
            len(missing),
            ", ".join(sorted(missing)),
        )
        logger.debug("Available channels: {}", ", ".join(raw.ch_names))

    raw.info["bads"] = valid_bads

    if valid_bads:
        logger.debug(
            "Marked {} bad channel(s): {}", len(valid_bads), ", ".join(valid_bads)
        )
    else:
        logger.info("No valid bad channels found to mark; leaving dataset unchanged.")


@register_processor
class EDFLoader(Processor):
    """Load EEG data from an EDF file.

    Creates a new ProcessingContext from the file at the given path.
    Optionally restricts the recording to a sample window and marks
    a list of channels as bad before returning.

    Parameters
    ----------
    path : str
        Path to the EDF file to load.
    bad_channels : list of str, optional
        Channel names to mark as bad (default: none).
    preload : bool, optional
        Whether to load data into memory immediately (default: True).
    artifact_to_trigger_offset : float, optional
        Offset of the artifact relative to the trigger, in seconds (default: 0.0).
    upsampling_factor : int, optional
        Upsampling factor stored in metadata for downstream processors (default: 1).
    start_sample : int, optional
        First sample index to keep, inclusive (default: first sample).
    stop_sample : int, optional
        Last sample index to keep, exclusive (default: last sample).
    """

    name = "edf_loader"
    description = "Load EEG data from EDF file"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = False
    modifies_raw = True
    parallel_safe = False

    def __init__(
        self,
        path: str,
        bad_channels: Optional[List[str]] = None,
        preload: bool = True,
        artifact_to_trigger_offset: float = 0.0,
        upsampling_factor: int = 1,
        start_sample: Optional[int] = None,
        stop_sample: Optional[int] = None,
    ) -> None:
        self.path = path
        self.bad_channels = bad_channels or []
        self.preload = preload
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.upsampling_factor = upsampling_factor
        self.start_sample = start_sample
        self.stop_sample = stop_sample
        super().__init__()

    def process(self, context: Optional[ProcessingContext]) -> ProcessingContext:
        # --- LOG ---
        logger.info("Loading EDF file: {}", self.path)

        # --- COMPUTE ---
        with suppress_stdout():
            raw = mne.io.read_raw_edf(self.path, preload=self.preload, verbose=False)

        _configure_bad_channels(raw, self.bad_channels)

        full_n_times = raw.n_times
        try:
            raw, start_idx, stop_idx = _apply_sample_window(
                raw, self.start_sample, self.stop_sample
            )
        except ValueError as exc:
            raise ProcessorValidationError(f"Invalid sample window: {exc}") from exc

        if start_idx != 0 or stop_idx != full_n_times:
            logger.info(
                "Applied sample window: start={}, stop={} (exclusive), kept {}/{} samples",
                start_idx,
                stop_idx,
                raw.n_times,
                full_n_times,
            )

        # --- BUILD RESULT ---
        acq_start = start_idx if start_idx != 0 else None
        acq_end = stop_idx if stop_idx != full_n_times else None
        if self.start_sample is not None:
            acq_start = start_idx
        if self.stop_sample is not None:
            acq_end = stop_idx

        metadata = ProcessingMetadata(
            artifact_to_trigger_offset=self.artifact_to_trigger_offset,
            upsampling_factor=self.upsampling_factor,
            acq_start_sample=acq_start,
            acq_end_sample=acq_end,
        )

        logger.info(
            "Loaded {} channels, {} samples, {}Hz",
            len(raw.ch_names),
            raw.n_times,
            raw.info["sfreq"],
        )

        # --- RETURN ---
        return ProcessingContext(raw=raw, metadata=metadata)


@register_processor
class BIDSLoader(Processor):
    """Load EEG data from a BIDS dataset.

    Creates a new ProcessingContext by reading EEG data identified by
    subject, task, and optional session from a BIDS-compliant directory.
    Optionally restricts the recording to a sample window and marks
    a list of channels as bad before returning.

    Parameters
    ----------
    root : str
        Path to the BIDS root directory.
    subject : str
        Subject identifier (without the ``sub-`` prefix).
    task : str
        Task name.
    session : str, optional
        Session identifier (without the ``ses-`` prefix).
    bad_channels : list of str, optional
        Channel names to mark as bad (default: none).
    preload : bool, optional
        Whether to load data into memory immediately (default: True).
    artifact_to_trigger_offset : float, optional
        Offset of the artifact relative to the trigger, in seconds (default: 0.0).
    upsampling_factor : int, optional
        Upsampling factor stored in metadata for downstream processors (default: 1).
    start_sample : int, optional
        First sample index to keep, inclusive (default: first sample).
    stop_sample : int, optional
        Last sample index to keep, exclusive (default: last sample).
    """

    name = "bids_loader"
    description = "Load EEG data from BIDS dataset"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = False
    modifies_raw = True
    parallel_safe = False

    def __init__(
        self,
        root: str,
        subject: str,
        task: str,
        session: Optional[str] = None,
        bad_channels: Optional[List[str]] = None,
        preload: bool = True,
        artifact_to_trigger_offset: float = 0.0,
        upsampling_factor: int = 1,
        start_sample: Optional[int] = None,
        stop_sample: Optional[int] = None,
    ) -> None:
        self.root = root
        self.subject = subject
        self.task = task
        self.session = session
        self.bad_channels = bad_channels or []
        self.preload = preload
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.upsampling_factor = upsampling_factor
        self.start_sample = start_sample
        self.stop_sample = stop_sample
        super().__init__()

    def validate(self, context: Optional[ProcessingContext]) -> None:
        # Don't call super() — loaders create the context, there is nothing to validate
        if not Path(self.root).exists():
            raise ProcessorValidationError(f"BIDS root directory not found: {self.root}")

    def process(self, context: Optional[ProcessingContext]) -> ProcessingContext:
        # --- LOG ---
        logger.info(
            "Loading BIDS data: subject={}, task={}, session={}",
            self.subject,
            self.task,
            self.session,
        )

        # --- COMPUTE ---
        bids_path = BIDSPath(
            subject=self.subject,
            session=self.session,
            task=self.task,
            root=self.root,
        )

        with suppress_stdout():
            raw = read_raw_bids(bids_path, verbose=False)

        if self.preload:
            raw.load_data()

        full_n_times = raw.n_times

        _configure_bad_channels(raw, self.bad_channels)

        try:
            raw, start_idx, stop_idx = _apply_sample_window(
                raw, self.start_sample, self.stop_sample
            )
        except ValueError as exc:
            raise ProcessorValidationError(f"Invalid sample window: {exc}") from exc

        if start_idx != 0 or stop_idx != full_n_times:
            logger.info(
                "Applied sample window: start={}, stop={} (exclusive), kept {}/{} samples",
                start_idx,
                stop_idx,
                raw.n_times,
                full_n_times,
            )

        # --- BUILD RESULT ---
        metadata = ProcessingMetadata(
            artifact_to_trigger_offset=self.artifact_to_trigger_offset,
            upsampling_factor=self.upsampling_factor,
        )
        metadata.custom["bids_path"] = bids_path
        metadata.acq_start_sample = start_idx
        metadata.acq_end_sample = stop_idx

        logger.info(
            "Loaded {} channels, {} samples, {}Hz",
            len(raw.ch_names),
            raw.n_times,
            raw.info["sfreq"],
        )

        # --- RETURN ---
        return ProcessingContext(raw=raw, metadata=metadata)


@register_processor
class GDFLoader(Processor):
    """Load EEG data from a GDF file.

    Creates a new ProcessingContext from the file at the given path.
    Optionally restricts the recording to a sample window and marks
    a list of channels as bad before returning.

    Parameters
    ----------
    path : str
        Path to the GDF file to load.
    bad_channels : list of str, optional
        Channel names to mark as bad (default: none).
    preload : bool, optional
        Whether to load data into memory immediately (default: True).
    artifact_to_trigger_offset : float, optional
        Offset of the artifact relative to the trigger, in seconds (default: 0.0).
    upsampling_factor : int, optional
        Upsampling factor stored in metadata for downstream processors (default: 1).
    start_sample : int, optional
        First sample index to keep, inclusive (default: first sample).
    stop_sample : int, optional
        Last sample index to keep, exclusive (default: last sample).
    """

    name = "gdf_loader"
    description = "Load EEG data from GDF file"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = False
    modifies_raw = True
    parallel_safe = False

    def __init__(
        self,
        path: str,
        bad_channels: Optional[List[str]] = None,
        preload: bool = True,
        artifact_to_trigger_offset: float = 0.0,
        upsampling_factor: int = 1,
        start_sample: Optional[int] = None,
        stop_sample: Optional[int] = None,
    ) -> None:
        self.path = path
        self.bad_channels = bad_channels or []
        self.preload = preload
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.upsampling_factor = upsampling_factor
        self.start_sample = start_sample
        self.stop_sample = stop_sample
        super().__init__()

    def process(self, context: Optional[ProcessingContext]) -> ProcessingContext:
        # --- LOG ---
        logger.info("Loading GDF file: {}", self.path)

        # --- COMPUTE ---
        with suppress_stdout():
            raw = mne.io.read_raw_gdf(self.path, preload=self.preload)

        full_n_times = raw.n_times

        _configure_bad_channels(raw, self.bad_channels)

        try:
            raw, start_idx, stop_idx = _apply_sample_window(
                raw, self.start_sample, self.stop_sample
            )
        except ValueError as exc:
            raise ProcessorValidationError(f"Invalid sample window: {exc}") from exc

        if start_idx != 0 or stop_idx != full_n_times:
            logger.info(
                "Applied sample window: start={}, stop={} (exclusive), kept {}/{} samples",
                start_idx,
                stop_idx,
                raw.n_times,
                full_n_times,
            )

        # --- BUILD RESULT ---
        metadata = ProcessingMetadata(
            artifact_to_trigger_offset=self.artifact_to_trigger_offset,
            upsampling_factor=self.upsampling_factor,
        )
        metadata.acq_start_sample = start_idx
        metadata.acq_end_sample = stop_idx

        logger.info(
            "Loaded {} channels, {} samples, {}Hz",
            len(raw.ch_names),
            raw.n_times,
            raw.info["sfreq"],
        )

        # --- RETURN ---
        return ProcessingContext(raw=raw, metadata=metadata)
