from typing import Any, Dict, Optional
import json
import os
from pathlib import Path
import mne
from loguru import logger

class MetadataHandler:
    """Handles FACET metadata storage and retrieval in MNE objects and BIDS."""
    
    METADATA_KEY = 'temp'
    FACET_KEY = 'facet'
    
    @classmethod
    def get_metadata(cls, raw: mne.io.Raw) -> Dict[str, Any]:
        """Get FACET metadata from an MNE Raw object."""
        if cls.METADATA_KEY not in raw.info:
            raw.info[cls.METADATA_KEY] = {}
        
        if cls.FACET_KEY not in raw.info[cls.METADATA_KEY]:
            raw.info[cls.METADATA_KEY][cls.FACET_KEY] = {}
            
        return raw.info[cls.METADATA_KEY][cls.FACET_KEY]

    @classmethod
    def set_metadata(cls, raw: mne.io.Raw, key: str, value: Any) -> None:
        """Set a metadata value in an MNE Raw object."""
        metadata = cls.get_metadata(raw)
        metadata[key] = value

    @classmethod
    def save_to_bids(cls, raw: mne.io.Raw, bids_path: str) -> None:
        """Save metadata to a JSON file in BIDS format."""
        metadata = cls.get_metadata(raw)
        
        # Create metadata directory if it doesn't exist
        metadata_dir = Path(bids_path).parent / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata filename based on BIDS path
        metadata_file = metadata_dir / f"{Path(bids_path).stem}_facet.json"
        
        # Save metadata to JSON file
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved FACET metadata to {metadata_file}")

    @classmethod
    def load_from_bids(cls, raw: mne.io.Raw, bids_path: str) -> None:
        """Load metadata from a JSON file in BIDS format."""
        metadata_file = Path(bids_path).parent / "metadata" / f"{Path(bids_path).stem}_facet.json"
        
        if not metadata_file.exists():
            logger.warning(f"No FACET metadata file found at {metadata_file}")
            return
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Update raw.info with loaded metadata
        raw.info[cls.METADATA_KEY] = {cls.FACET_KEY: metadata}
        
        logger.info(f"Loaded FACET metadata from {metadata_file}") 