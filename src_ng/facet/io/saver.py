from typing import Optional, Dict, Any
import mne
from mne_bids import BIDSPath, write_raw_bids
from loguru import logger

class EEGSaver:
    """Handles saving EEG data to various file formats."""
    
    @classmethod
    def save(cls, raw: mne.io.Raw, path: str, format: str = "auto", **kwargs) -> None:
        """
        Save EEG data to file.
        
        Args:
            raw: MNE Raw object to save
            path: Output file path
            format: Output format ('edf', 'bdf', 'gdf', 'bids', etc.)
            **kwargs: Additional arguments for specific formats
                     For BIDS: subject, session, task, event_id
                     For all: overwrite
        """
        if format == "auto":
            format = cls._detect_format(path)
            
        saver = getattr(cls, f"_save_{format}", None)
        if saver is None:
            raise ValueError(f"Unsupported format: {format}")
            
        return saver(raw, path, **kwargs)
    
    @staticmethod
    def _detect_format(path: str) -> str:
        """Detect file format from path extension."""
        ext = path.lower().split('.')[-1]
        format_map = {
            'edf': 'edf',
            'bdf': 'bdf',
            'gdf': 'gdf',
            'set': 'eeglab'
        }
        return format_map.get(ext, 'edf')
    
    @staticmethod
    def _save_edf(raw: mne.io.Raw, path: str, **kwargs) -> None:
        """Save as EDF format."""
        raw.export(path, fmt='edf', overwrite=kwargs.get('overwrite', False))
    
    @staticmethod
    def _save_bids(raw: mne.io.Raw, path: str, **kwargs) -> None:
        """Save as BIDS format."""
        bids_path = BIDSPath(
            subject=kwargs.get('subject', 'subjectid'),
            session=kwargs.get('session', 'sessionid'),
            task=kwargs.get('task', 'task'),
            root=path
        )
        
        # Remove stim channels before saving
        raw_save = raw.copy()
        stim_channels = mne.pick_types(raw_save.info, meg=False, stim=True)
        if len(stim_channels) > 0:
            raw_save.drop_channels([raw_save.ch_names[ch] for ch in stim_channels])
            
        write_raw_bids(
            raw=raw_save,
            bids_path=bids_path,
            overwrite=kwargs.get('overwrite', True),
            format='EDF',
            allow_preload=True,
            events=kwargs.get('events'),
            event_id=kwargs.get('event_id')
        ) 