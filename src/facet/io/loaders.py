"""
Data Loaders Module

This module contains processors for loading EEG data from various formats.

Author: FACETpy Team
Date: 2025-01-12
"""

from numbers import Integral, Real
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
    stop_sample: Optional[int]
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


@register_processor
class EDFLoader(Processor):
    """
    Load EEG data from EDF format.

    Example:
        loader = EDFLoader(
            path="data.edf",
            bad_channels=["EKG", "EOG"],
            preload=True
        )
        context = loader.execute(None)
    """

    name = "edf_loader"
    description = "Load EEG data from EDF file"
    requires_raw = False  # Creates raw

    def __init__(
        self,
        path: str,
        bad_channels: Optional[List[str]] = None,
        preload: bool = True,
        artifact_to_trigger_offset: float = 0.0,
        upsampling_factor: int = 1,
        start_sample: Optional[int] = None,
        stop_sample: Optional[int] = None,
    ):
        """
        Initialize EDF loader.

        Args:
            path: Path to EDF file
            bad_channels: List of bad channel names to exclude
            preload: Whether to load data into memory
            artifact_to_trigger_offset: Offset of artifact relative to trigger
            upsampling_factor: Upsampling factor to apply later
            start_sample: Optional first sample (inclusive) to keep
            stop_sample: Optional last sample (exclusive) to keep
        """
        self.path = path
        self.bad_channels = bad_channels or []
        self.preload = preload
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.upsampling_factor = upsampling_factor
        self.start_sample = start_sample
        self.stop_sample = stop_sample
        super().__init__()

    def validate(self, context: Optional[ProcessingContext]) -> None:
        """No validation needed for loader."""
        pass

    def process(self, context: Optional[ProcessingContext]) -> ProcessingContext:
        """Load EDF file."""
        logger.info(f"Loading EDF file: {self.path}")

        # Load raw data (suppress MNE's verbose print output)
        with suppress_stdout():
            raw = mne.io.read_raw_edf(self.path, preload=self.preload, verbose=False)

        # Mark bad channels with validation against available channels
        if self.bad_channels:
            available_channels = set(raw.ch_names)
            valid_bad_channels = [ch for ch in self.bad_channels if ch in available_channels]
            missing_channels = [ch for ch in self.bad_channels if ch not in available_channels]

            if missing_channels:
                logger.warning(
                    f"Skipping {len(missing_channels)} bad channel(s) not present in data: "
                    f"{', '.join(missing_channels)}"
                )
                logger.debug(
                    f"Available channels: {', '.join(raw.ch_names)}"
                )

            raw.info['bads'] = valid_bad_channels
            if valid_bad_channels:
                logger.debug(
                    f"Marked {len(valid_bad_channels)} bad channel(s): "
                    f"{', '.join(valid_bad_channels)}"
                )
            else:
                logger.info("No valid bad channels found to mark; leaving dataset unchanged.")

        full_n_times = raw.n_times
        try:
            raw, start_idx, stop_idx = _apply_sample_window(raw, self.start_sample, self.stop_sample)
        except ValueError as exc:
            raise ProcessorValidationError(f"Invalid sample window: {exc}") from exc

        if start_idx != 0 or stop_idx != full_n_times:
            logger.info(
                "Applied sample window: start=%d, stop=%d (exclusive), kept %d/%d samples",
                start_idx,
                stop_idx,
                raw.n_times,
                full_n_times,
            )

        # Create metadata
        metadata = ProcessingMetadata(
            artifact_to_trigger_offset=self.artifact_to_trigger_offset,
            upsampling_factor=self.upsampling_factor
        )

        # Create context
        ctx = ProcessingContext(raw=raw, metadata=metadata)

        logger.info(
            f"Loaded {len(raw.ch_names)} channels, "
            f"{raw.n_times} samples, "
            f"{raw.info['sfreq']}Hz"
        )

        return ctx


@register_processor
class BIDSLoader(Processor):
    """
    Load EEG data from BIDS format.

    Example:
        loader = BIDSLoader(
            root="data/",
            subject="01",
            session="01",
            task="rest",
            bad_channels=["EKG"]
        )
    """

    name = "bids_loader"
    description = "Load EEG data from BIDS dataset"
    requires_raw = False

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
    ):
        """
        Initialize BIDS loader.

        Args:
            root: BIDS root directory
            subject: Subject ID
            task: Task name
            session: Session ID (optional)
            bad_channels: List of bad channel names
            preload: Whether to load data into memory
            artifact_to_trigger_offset: Offset of artifact relative to trigger
            upsampling_factor: Upsampling factor
            start_sample: Optional first sample (inclusive) to keep
            stop_sample: Optional last sample (exclusive) to keep
        """
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
        """No validation needed for loader."""
        pass

    def process(self, context: Optional[ProcessingContext]) -> ProcessingContext:
        """Load BIDS data."""
        logger.info(
            f"Loading BIDS data: subject={self.subject}, task={self.task}, "
            f"session={self.session}"
        )

        # Create BIDS path
        bids_path = BIDSPath(
            subject=self.subject,
            session=self.session,
            task=self.task,
            root=self.root
        )

        # Load raw data
        raw = read_raw_bids(bids_path, verbose=False)

        if self.preload:
            raw.load_data()

        full_n_times = raw.n_times

        # Mark bad channels with validation against available channels
        if self.bad_channels:
            available_channels = set(raw.ch_names)
            valid_bad_channels = [ch for ch in self.bad_channels if ch in available_channels]
            missing_channels = [ch for ch in self.bad_channels if ch not in available_channels]

            if missing_channels:
                logger.warning(
                    f"Skipping {len(missing_channels)} bad channel(s) not present in data: "
                    f"{', '.join(missing_channels)}"
                )
                logger.debug(
                    f"Available channels: {', '.join(raw.ch_names)}"
                )

            raw.info['bads'] = valid_bad_channels
            if valid_bad_channels:
                logger.debug(
                    f"Marked {len(valid_bad_channels)} bad channel(s): "
                    f"{', '.join(valid_bad_channels)}"
                )
            else:
                logger.info("No valid bad channels found to mark; leaving dataset unchanged.")

        try:
            raw, start_idx, stop_idx = _apply_sample_window(raw, self.start_sample, self.stop_sample)
        except ValueError as exc:
            raise ProcessorValidationError(f"Invalid sample window: {exc}") from exc

        if start_idx != 0 or stop_idx != full_n_times:
            logger.info(
                "Applied sample window: start=%d, stop=%d (exclusive), kept %d/%d samples",
                start_idx,
                stop_idx,
                raw.n_times,
                full_n_times,
            )

        # Create metadata
        metadata = ProcessingMetadata(
            artifact_to_trigger_offset=self.artifact_to_trigger_offset,
            upsampling_factor=self.upsampling_factor
        )
        metadata.custom['bids_path'] = bids_path
        metadata.acq_start_sample = start_idx
        metadata.acq_end_sample = stop_idx

        # Create context
        ctx = ProcessingContext(raw=raw, metadata=metadata)

        logger.info(
            f"Loaded {len(raw.ch_names)} channels, "
            f"{raw.n_times} samples, "
            f"{raw.info['sfreq']}Hz"
        )

        return ctx


@register_processor
class GDFLoader(Processor):
    """
    Load EEG data from GDF format.

    Example:
        loader = GDFLoader(
            path="data.gdf",
            bad_channels=["EKG"]
        )
    """

    name = "gdf_loader"
    description = "Load EEG data from GDF file"
    requires_raw = False

    def __init__(
        self,
        path: str,
        bad_channels: Optional[List[str]] = None,
        preload: bool = True,
        artifact_to_trigger_offset: float = 0.0,
        upsampling_factor: int = 1,
        start_sample: Optional[int] = None,
        stop_sample: Optional[int] = None,
    ):
        """
        Initialize GDF loader.

        Args:
            path: Path to GDF file
            bad_channels: List of bad channel names
            preload: Whether to load data into memory
            artifact_to_trigger_offset: Offset of artifact relative to trigger
            upsampling_factor: Upsampling factor
            start_sample: Optional first sample (inclusive) to keep
            stop_sample: Optional last sample (exclusive) to keep
        """
        self.path = path
        self.bad_channels = bad_channels or []
        self.preload = preload
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.upsampling_factor = upsampling_factor
        self.start_sample = start_sample
        self.stop_sample = stop_sample
        super().__init__()

    def validate(self, context: Optional[ProcessingContext]) -> None:
        """No validation needed for loader."""
        pass

    def process(self, context: Optional[ProcessingContext]) -> ProcessingContext:
        """Load GDF file."""
        logger.info(f"Loading GDF file: {self.path}")

        # Load raw data
        raw = mne.io.read_raw_gdf(self.path, preload=self.preload)

        full_n_times = raw.n_times

        # Mark bad channels with validation against available channels
        if self.bad_channels:
            available_channels = set(raw.ch_names)
            valid_bad_channels = [ch for ch in self.bad_channels if ch in available_channels]
            missing_channels = [ch for ch in self.bad_channels if ch not in available_channels]

            if missing_channels:
                logger.warning(
                    f"Skipping {len(missing_channels)} bad channel(s) not present in data: "
                    f"{', '.join(missing_channels)}"
                )
                logger.debug(
                    f"Available channels: {', '.join(raw.ch_names)}"
                )

            raw.info['bads'] = valid_bad_channels
            if valid_bad_channels:
                logger.debug(
                    f"Marked {len(valid_bad_channels)} bad channel(s): "
                    f"{', '.join(valid_bad_channels)}"
                )
            else:
                logger.info("No valid bad channels found to mark; leaving dataset unchanged.")

        try:
            raw, start_idx, stop_idx = _apply_sample_window(raw, self.start_sample, self.stop_sample)
        except ValueError as exc:
            raise ProcessorValidationError(f"Invalid sample window: {exc}") from exc

        if start_idx != 0 or stop_idx != full_n_times:
            logger.info(
                "Applied sample window: start=%d, stop=%d (exclusive), kept %d/%d samples",
                start_idx,
                stop_idx,
                raw.n_times,
                full_n_times,
            )

        # Create metadata
        metadata = ProcessingMetadata(
            artifact_to_trigger_offset=self.artifact_to_trigger_offset,
            upsampling_factor=self.upsampling_factor
        )
        metadata.acq_start_sample = start_idx
        metadata.acq_end_sample = stop_idx

        # Create context
        ctx = ProcessingContext(raw=raw, metadata=metadata)

        logger.info(
            f"Loaded {len(raw.ch_names)} channels, "
            f"{raw.n_times} samples, "
            f"{raw.info['sfreq']}Hz"
        )

        return ctx
