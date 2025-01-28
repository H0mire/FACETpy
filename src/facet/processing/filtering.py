from typing import Optional
import numpy as np
from loguru import logger
from scipy.signal import firls, filtfilt

from ..core.facet import FACET

class FilterProcessor:
    """Handles signal filtering operations."""
    
    def __init__(self, facet: 'FACET'):
        """
        Initialize FilterProcessor.
        
        Args:
            facet: Parent FACET instance
        """
        self._facet: 'FACET' = facet
        
    def bandpass(self, l_freq: float, h_freq: float) -> 'FilterProcessor':
        """Apply bandpass filter to the data."""
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        raw.filter(l_freq=l_freq, h_freq=h_freq)
        
        # Save to metadata
        self._facet.set_metadata('filter_l_freq', l_freq)
        self._facet.set_metadata('filter_h_freq', h_freq)
        
        return self
        
    def highpass(self, freq: float) -> 'FilterProcessor':
        """Apply highpass filter to the data."""
        return self.bandpass(l_freq=freq, h_freq=None)
        
    def lowpass(self, freq: float) -> 'FilterProcessor':
        """Apply lowpass filter to the data."""
        return self.bandpass(l_freq=None, h_freq=freq)
        
    def notch(self, freqs: float = 50) -> 'FilterProcessor':
        """Apply notch filter to remove power line noise."""
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        raw.notch_filter(freqs)
        
        # Save to metadata
        self._facet.set_metadata('notch_freq', freqs)
        
        return self
        
    def design_filter(self, 
                     freq: float, 
                     filter_type: str = 'highpass',
                     order: Optional[int] = None) -> np.ndarray:
        """
        Design FIR filter coefficients.
        
        Args:
            freq: Cutoff frequency
            filter_type: Type of filter ('highpass', 'lowpass', 'bandpass')
            order: Filter order (if None, automatically determined)
            
        Returns:
            np.ndarray: Filter coefficients
        """
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        nyq = 0.5 * raw.info['sfreq']
        
        if order is None:
            order = int(3.3 * raw.info['sfreq'] / freq)
            if order % 2 == 0:
                order += 1
                
        if filter_type == 'highpass':
            f = [0, 0.8*freq/nyq, freq/nyq, 1]
            a = [0, 0, 1, 1]
        elif filter_type == 'lowpass':
            f = [0, freq/nyq, 1.2*freq/nyq, 1]
            a = [1, 1, 0, 0]
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
            
        return firls(order, f, a) 