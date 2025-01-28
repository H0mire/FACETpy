from typing import List, Optional, Union
import numpy as np
import mne
from loguru import logger
import re
import neurokit2 as nk
from scipy.signal import find_peaks

class TriggerProcessor:
    """Handles trigger detection and manipulation in EEG data."""
    
    def __init__(self, facet):
        """
        Initialize TriggerProcessor.
        
        Args:
            facet: Parent FACET instance
        """
        self._facet = facet
        self._triggers: Optional[List[int]] = None
        
    @property
    def triggers(self) -> Optional[List[int]]:
        """Get current trigger positions."""
        return self._triggers
        
    def find(self, regex: str, save: bool = True) -> 'TriggerProcessor':
        """
        Find triggers matching regex pattern.
        
        Args:
            regex: Regular expression to match trigger values
            save: Whether to save triggers to annotations
        """
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        # Try stim channels first
        stim_channels = mne.pick_types(raw.info, meg=False, stim=True)
        if len(stim_channels) > 0:
            events = mne.find_events(
                raw, 
                stim_channel=raw.ch_names[stim_channels[0]], 
                initial_event=True
            )
            pattern = re.compile(regex)
            self._triggers = [
                event[0] for event in events 
                if pattern.search(str(event[2]))
            ]
        else:
            # Try annotations
            events = mne.events_from_annotations(raw, regexp=regex)[0]
            self._triggers = list(events[:, 0])
            
        if save:
            self._save_triggers_to_annotations()
            
        # Save to metadata
        self._facet.set_metadata('triggers', self._triggers)
        self._facet.set_metadata('trigger_regex', regex)
        
        return self
        
    def find_qrs(self, save: bool = True) -> 'TriggerProcessor':
        """Find QRS complexes in ECG channel."""
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        # Find ECG channel
        ecg_channels = mne.pick_types(raw.info, meg=False, ecg=True)
        if len(ecg_channels) == 0:
            # Look for channels with ECG/EKG in name
            ecg_channels = [
                i for i, ch in enumerate(raw.ch_names)
                if "ECG" in ch.upper() or "EKG" in ch.upper()
            ]
            if not ecg_channels:
                raise ValueError("No ECG channels found")
                
        # Get ECG data
        ecg_data = raw.get_data(picks=ecg_channels[0])
        
        # Clean ECG signal
        cleaned_ecg = nk.ecg_clean(
            ecg_data[0], 
            sampling_rate=raw.info['sfreq'], 
            method="neurokit"
        )
        
        # Find R-peaks
        peaks, _ = find_peaks(
            cleaned_ecg,
            distance=int(0.5 * raw.info['sfreq']),  # Minimum 0.5s between peaks
            height=0.7
        )
        
        self._triggers = list(peaks)
        
        if save:
            self._save_triggers_to_annotations()
            
        # Save to metadata
        self._facet.set_metadata('triggers', self._triggers)
        self._facet.set_metadata('trigger_type', 'qrs')
        
        return self
        
    def _save_triggers_to_annotations(self) -> None:
        """Save current triggers as MNE annotations."""
        if not self._triggers:
            return
            
        raw = self._facet.data
        onsets = np.array(self._triggers) / raw.info['sfreq']
        durations = np.zeros_like(onsets)
        descriptions = ['Trigger'] * len(onsets)
        
        raw.set_annotations(mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions
        )) 