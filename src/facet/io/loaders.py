"""
Data Loaders Module

This module contains processors for loading EEG data from various formats.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, List
import mne
from mne_bids import BIDSPath, read_raw_bids
from loguru import logger
import numpy as np

from ..core import Processor, ProcessingContext, ProcessingMetadata, register_processor


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
        upsampling_factor: int = 1
    ):
        """
        Initialize EDF loader.

        Args:
            path: Path to EDF file
            bad_channels: List of bad channel names to exclude
            preload: Whether to load data into memory
            artifact_to_trigger_offset: Offset of artifact relative to trigger
            upsampling_factor: Upsampling factor to apply later
        """
        self.path = path
        self.bad_channels = bad_channels or []
        self.preload = preload
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.upsampling_factor = upsampling_factor
        super().__init__()

    def validate(self, context: Optional[ProcessingContext]) -> None:
        """No validation needed for loader."""
        pass

    def process(self, context: Optional[ProcessingContext]) -> ProcessingContext:
        """Load EDF file."""
        logger.info(f"Loading EDF file: {self.path}")

        # Load raw data
        raw = mne.io.read_raw_edf(self.path, preload=self.preload)

        # Mark bad channels
        if self.bad_channels:
            raw.info['bads'] = self.bad_channels
            logger.debug(f"Marked {len(self.bad_channels)} bad channels")

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
        upsampling_factor: int = 1
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
        """
        self.root = root
        self.subject = subject
        self.task = task
        self.session = session
        self.bad_channels = bad_channels or []
        self.preload = preload
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.upsampling_factor = upsampling_factor
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

        # Mark bad channels
        if self.bad_channels:
            raw.info['bads'] = self.bad_channels
            logger.debug(f"Marked {len(self.bad_channels)} bad channels")

        # Create metadata
        metadata = ProcessingMetadata(
            artifact_to_trigger_offset=self.artifact_to_trigger_offset,
            upsampling_factor=self.upsampling_factor
        )
        metadata.custom['bids_path'] = bids_path

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
        upsampling_factor: int = 1
    ):
        """
        Initialize GDF loader.

        Args:
            path: Path to GDF file
            bad_channels: List of bad channel names
            preload: Whether to load data into memory
            artifact_to_trigger_offset: Offset of artifact relative to trigger
            upsampling_factor: Upsampling factor
        """
        self.path = path
        self.bad_channels = bad_channels or []
        self.preload = preload
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.upsampling_factor = upsampling_factor
        super().__init__()

    def validate(self, context: Optional[ProcessingContext]) -> None:
        """No validation needed for loader."""
        pass

    def process(self, context: Optional[ProcessingContext]) -> ProcessingContext:
        """Load GDF file."""
        logger.info(f"Loading GDF file: {self.path}")

        # Load raw data
        raw = mne.io.read_raw_gdf(self.path, preload=self.preload)

        # Mark bad channels
        if self.bad_channels:
            raw.info['bads'] = self.bad_channels
            logger.debug(f"Marked {len(self.bad_channels)} bad channels")

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
