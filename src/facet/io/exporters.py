"""
Data Exporters Module

This module contains processors for exporting EEG data to various formats.

Author: FACETpy Team
Date: 2025-01-12
"""

from pathlib import Path
from typing import Optional, Dict

import numpy as np
import mne
from mne_bids import BIDSPath, write_raw_bids
from loguru import logger

from ..core import Processor, ProcessingContext, register_processor


@register_processor
class EDFExporter(Processor):
    """Export EEG data to EDF format.

    Writes the current Raw object to an EDF file at the specified path.
    Parent directories are created automatically if they do not exist.
    The context is returned unchanged; only the file is written.

    Parameters
    ----------
    path : str
        Destination file path for the exported EDF.
    overwrite : bool, optional
        Whether to overwrite an existing file (default: True).
    """

    name = "edf_exporter"
    description = "Export EEG data to EDF file"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        path: str,
        overwrite: bool = True,
    ) -> None:
        self.path = path
        self.overwrite = overwrite
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw()

        # --- LOG ---
        logger.info("Exporting to EDF: {}", self.path)

        # --- COMPUTE ---
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        raw.export(self.path, fmt="edf", overwrite=self.overwrite)

        logger.info("Export completed")

        # --- RETURN ---
        return context


@register_processor
class BIDSExporter(Processor):
    """Export EEG data to BIDS format.

    Writes the current Raw object into a BIDS-compliant directory structure
    using MNE-BIDS. Stimulus channels are dropped before writing as per BIDS
    convention. If triggers are available in the context they are written as
    events. Parent directories are created automatically.

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
    event_id : dict, optional
        Mapping of event description strings to integer event codes.
    overwrite : bool, optional
        Whether to overwrite existing BIDS files (default: True).
    """

    name = "bids_exporter"
    description = "Export EEG data to BIDS dataset"
    version = "1.0.0"

    requires_triggers = False
    requires_raw = True
    modifies_raw = False
    parallel_safe = False

    def __init__(
        self,
        root: str,
        subject: str,
        task: str,
        session: Optional[str] = None,
        event_id: Optional[Dict] = None,
        overwrite: bool = True,
    ) -> None:
        self.root = root
        self.subject = subject
        self.task = task
        self.session = session
        self.event_id = event_id
        self.overwrite = overwrite
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        # --- EXTRACT ---
        raw = context.get_raw().copy()

        # --- LOG ---
        logger.info(
            "Exporting to BIDS: subject={}, task={}",
            self.subject,
            self.task,
        )

        # --- COMPUTE ---
        Path(self.root).mkdir(parents=True, exist_ok=True)

        bids_path = BIDSPath(
            subject=self.subject,
            session=self.session,
            task=self.task,
            root=self.root,
        )

        # Drop stim channels (BIDS convention)
        stim_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
        if len(stim_channels) > 0:
            raw.drop_channels([raw.ch_names[ch] for ch in stim_channels])

        events = None
        if context.has_triggers():
            triggers = context.get_triggers()
            events = np.array([[t, 0, 1] for t in triggers], dtype=np.int32)

        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            overwrite=self.overwrite,
            format="EDF",
            allow_preload=True,
            events=events,
            event_id=self.event_id,
            verbose=False,
        )

        logger.info("Export completed")

        # --- RETURN ---
        return context
