"""
Data Exporters Module

This module contains processors for exporting EEG data to various formats.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, Dict
import mne
from mne_bids import BIDSPath, write_raw_bids
from loguru import logger

from ..core import Processor, ProcessingContext, register_processor


@register_processor
class EDFExporter(Processor):
    """
    Export EEG data to EDF format.

    Example:
        exporter = EDFExporter(path="output.edf", overwrite=True)
        context = exporter.execute(context)
    """

    name = "edf_exporter"
    description = "Export EEG data to EDF file"
    modifies_raw = False  # Doesn't modify raw

    def __init__(
        self,
        path: str,
        overwrite: bool = True
    ):
        """
        Initialize EDF exporter.

        Args:
            path: Output file path
            overwrite: Whether to overwrite existing file
        """
        self.path = path
        self.overwrite = overwrite
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Export to EDF."""
        logger.info(f"Exporting to EDF: {self.path}")

        raw = context.get_raw()
        raw.export(self.path, fmt='edf', overwrite=self.overwrite)

        logger.info("Export completed")
        return context


@register_processor
class BIDSExporter(Processor):
    """
    Export EEG data to BIDS format.

    Example:
        exporter = BIDSExporter(
            root="output_bids/",
            subject="01",
            task="rest"
        )
    """

    name = "bids_exporter"
    description = "Export EEG data to BIDS dataset"
    modifies_raw = False

    def __init__(
        self,
        root: str,
        subject: str,
        task: str,
        session: Optional[str] = None,
        event_id: Optional[Dict] = None,
        overwrite: bool = True
    ):
        """
        Initialize BIDS exporter.

        Args:
            root: BIDS root directory
            subject: Subject ID
            task: Task name
            session: Session ID (optional)
            event_id: Event ID mapping (optional)
            overwrite: Whether to overwrite existing files
        """
        self.root = root
        self.subject = subject
        self.task = task
        self.session = session
        self.event_id = event_id
        self.overwrite = overwrite
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Export to BIDS."""
        logger.info(
            f"Exporting to BIDS: subject={self.subject}, task={self.task}"
        )

        # Create BIDS path
        bids_path = BIDSPath(
            subject=self.subject,
            session=self.session,
            task=self.task,
            root=self.root
        )

        raw = context.get_raw().copy()

        # Drop stim channels (BIDS convention)
        stim_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
        if len(stim_channels) > 0:
            raw.drop_channels([raw.ch_names[ch] for ch in stim_channels])

        # Get events if triggers exist
        events = None
        if context.has_triggers():
            triggers = context.get_triggers()
            events = []
            for trigger in triggers:
                events.append([trigger, 0, 1])
            import numpy as np
            events = np.array(events, dtype=np.int32)

        # Write BIDS
        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            overwrite=self.overwrite,
            format='EDF',
            allow_preload=True,
            events=events,
            event_id=self.event_id,
            verbose=False
        )

        logger.info("Export completed")
        return context
