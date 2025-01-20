from typing import List, Optional, Union, Tuple
import numpy as np
import mne
from loguru import logger
from scipy.signal import correlate
from scipy.interpolate import interp1d

class AlignmentProcessor:
    """Handles alignment of triggers and artifacts."""
    
    def __init__(self, facet):
        """
        Initialize AlignmentProcessor.
        
        Args:
            facet: Parent FACET instance
        """
        self._facet = facet
        self._subsample_shifts: Optional[np.ndarray] = None
        
    def align_triggers(self, 
                      ref_trigger_idx: int,
                      search_window: int = 100,
                      upsample: bool = True) -> 'AlignmentProcessor':
        """
        Align triggers using cross-correlation with a reference trigger.
        
        Args:
            ref_trigger_idx: Index of the reference trigger
            search_window: Number of samples to search around each trigger
            upsample: Whether to use upsampled data for more precise alignment
        """
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        triggers = self._facet.get_metadata('triggers')
        if not triggers:
            raise ValueError("No triggers found")
            
        # Get reference epoch
        ref_trigger = triggers[ref_trigger_idx]
        ref_epoch = self._extract_epoch(raw, ref_trigger, search_window)
        
        # Calculate shifts for each trigger
        shifts = []
        for trigger in triggers:
            if trigger == ref_trigger:
                shifts.append(0)
                continue
                
            epoch = self._extract_epoch(raw, trigger, search_window)
            shift = self._find_max_correlation(ref_epoch, epoch)
            shifts.append(shift)
            
        # Apply shifts
        new_triggers = [t + s for t, s in zip(triggers, shifts)]
        
        # Update triggers in metadata
        self._facet.set_metadata('triggers', new_triggers)
        self._facet.set_metadata('trigger_alignment_ref', ref_trigger_idx)
        
        # Store subsample shifts if upsampling was used
        if upsample:
            self._subsample_shifts = np.array(shifts) % 1
            
        return self
        
    def align_subsample(self, 
                       window_size: Optional[int] = None) -> 'AlignmentProcessor':
        """
        Perform subsample alignment using interpolation.
        
        Args:
            window_size: Size of the window around triggers to interpolate
        """
        if self._subsample_shifts is None:
            raise ValueError("No subsample shifts available. Run align_triggers first.")
            
        raw = self._facet.data
        triggers = self._facet.get_metadata('triggers')
        
        if window_size is None:
            window_size = self._facet.get_metadata('artifact_window_size', 100)
            
        # Apply subsample shifts using interpolation
        for ch_idx in range(raw.info['nchan']):
            data = raw.get_data(picks=[ch_idx])[0]
            new_data = data.copy()
            
            for trigger, shift in zip(triggers, self._subsample_shifts):
                if shift == 0:
                    continue
                    
                start = max(0, trigger - window_size//2)
                end = min(len(data), trigger + window_size//2)
                
                # Create interpolation function
                x = np.arange(start, end)
                f = interp1d(x, data[start:end], kind='cubic')
                
                # Apply shift
                x_new = x + shift
                new_data[start:end] = f(x_new)
                
            raw._data[ch_idx] = new_data
            
        return self
        
    def detect_volume_gaps(self, 
                         tolerance: float = 0.1) -> Tuple[bool, List[int]]:
        """
        Detect gaps between volume acquisitions.
        
        Args:
            tolerance: Relative tolerance for trigger interval variation
            
        Returns:
            Tuple of (has_gaps, gap_indices)
        """
        triggers = self._facet.get_metadata('triggers')
        if not triggers:
            raise ValueError("No triggers found")
            
        intervals = np.diff(triggers)
        mean_interval = np.mean(intervals)
        
        # Find intervals that are significantly larger
        gaps = np.where(intervals > mean_interval * (1 + tolerance))[0]
        
        has_gaps = len(gaps) > 0
        self._facet.set_metadata('volume_gaps', has_gaps)
        self._facet.set_metadata('gap_locations', gaps.tolist())
        
        return has_gaps, gaps.tolist()
        
    def _extract_epoch(self, 
                      raw: mne.io.Raw,
                      trigger: int,
                      window_size: int) -> np.ndarray:
        """Extract data epoch around trigger."""
        start = max(0, trigger - window_size//2)
        end = min(raw.n_times, trigger + window_size//2)
        
        # Use first EEG channel for alignment
        ch_idx = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)[0]
        return raw.get_data(picks=[ch_idx], start=start, stop=end)[0]
        
    def _find_max_correlation(self, 
                            ref_epoch: np.ndarray,
                            epoch: np.ndarray) -> int:
        """Find the shift that maximizes correlation between epochs."""
        corr = correlate(ref_epoch, epoch, mode='full')
        max_idx = np.argmax(corr)
        return max_idx - len(ref_epoch) + 1 