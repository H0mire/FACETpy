from typing import Optional
import mne
from mne_bids import BIDSPath, read_raw_bids
from loguru import logger

class EEGLoader:
    """Handles loading EEG data from various file formats."""
    
    @classmethod
    def load(cls, path: str, format: str = "auto", **kwargs) -> mne.io.Raw:
        """
        Load EEG data from file.
        
        Args:
            path: Path to EEG file
            format: File format ('edf', 'bdf', 'gdf', 'bids', 'eeglab', etc.)
            **kwargs: Additional arguments for specific formats
                     For BIDS: subject, session, task
                     For all: preload, bads
        
        Returns:
            mne.io.Raw: Loaded EEG data
        """
        if format == "auto":
            format = cls._detect_format(path)
            
        loader = getattr(cls, f"_load_{format}", None)
        if loader is None:
            raise ValueError(f"Unsupported format: {format}")
            
        raw = loader(path, **kwargs)
        
        # Apply common configurations
        if 'bads' in kwargs:
            raw.info['bads'] = kwargs['bads']
            
        return raw
    
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
    def _load_edf(path: str, **kwargs) -> mne.io.Raw:
        """Load EDF format."""
        return mne.io.read_raw_edf(path, preload=kwargs.get('preload', True))
    
    @staticmethod
    def _load_gdf(path: str, **kwargs) -> mne.io.Raw:
        """Load GDF format."""
        return mne.io.read_raw_gdf(path, preload=kwargs.get('preload', True))
        
    @staticmethod
    def _load_bids(path: str, **kwargs) -> mne.io.Raw:
        """Load BIDS format."""
        bids_path = BIDSPath(
            subject=kwargs.get('subject', 'subjectid'),
            session=kwargs.get('session', 'sessionid'),
            task=kwargs.get('task', 'task'),
            root=path
        )
        return read_raw_bids(bids_path, verbose=False)
    
    @staticmethod
    def _load_eeglab(path: str, **kwargs) -> mne.io.Raw:
        """Load EEGLAB format."""
        return mne.io.read_raw_eeglab(path, preload=kwargs.get('preload', True)) 