from typing import List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mne
from loguru import logger

class Plotter:
    """Handles visualization of EEG data and analysis results."""
    
    def __init__(self, facet):
        """
        Initialize Plotter.
        
        Args:
            facet: Parent FACET instance
        """
        self._facet = facet
        
    def plot_raw(self, 
                 duration: float = 10.0,
                 start: float = 0.0,
                 n_channels: int = 10) -> None:
        """Plot raw EEG data."""
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        raw.plot(
            duration=duration,
            start=start,
            n_channels=n_channels,
            scalings='auto',
            title='Raw EEG Data'
        )
        
    def plot_triggers(self, 
                     window: float = 1.0,
                     n_triggers: int = 5) -> None:
        """Plot data around triggers."""
        raw = self._facet.data
        triggers = self._facet.get_metadata('triggers')
        if not triggers:
            raise ValueError("No triggers found")
            
        # Convert window from seconds to samples
        window_samples = int(window * raw.info['sfreq'])
        
        # Select subset of triggers
        trigger_subset = triggers[:n_triggers]
        
        fig, axes = plt.subplots(n_triggers, 1, figsize=(10, 3*n_triggers))
        if n_triggers == 1:
            axes = [axes]
            
        for ax, trigger in zip(axes, trigger_subset):
            start = max(0, trigger - window_samples//2)
            end = min(raw.n_times, trigger + window_samples//2)
            
            times = np.arange(start, end) / raw.info['sfreq']
            data = raw.get_data(start=start, stop=end)
            
            ax.plot(times, data.T)
            ax.axvline(trigger/raw.info['sfreq'], color='r', linestyle='--')
            ax.set_title(f'Trigger at {trigger/raw.info["sfreq"]:.3f}s')
            
        plt.tight_layout()
        plt.show()
        
    def plot_psd(self, 
                 fmin: float = 0,
                 fmax: float = 100,
                 tmin: Optional[float] = None,
                 tmax: Optional[float] = None) -> None:
        """Plot power spectral density."""
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        raw.plot_psd(
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            picks='eeg',
            average=True
        )
        
    def plot_artifact_comparison(self, 
                               n_artifacts: int = 5,
                               window: float = 0.5) -> None:
        """Plot comparison of original vs. corrected artifacts."""
        raw = self._facet.data
        raw_orig = self._facet.raw_orig
        if raw is None or raw_orig is None:
            raise ValueError("Need both current and original data")
            
        triggers = self._facet.get_metadata('triggers')
        if not triggers:
            raise ValueError("No triggers found")
            
        window_samples = int(window * raw.info['sfreq'])
        trigger_subset = triggers[:n_artifacts]
        
        fig = plt.figure(figsize=(15, 3*n_artifacts))
        gs = GridSpec(n_artifacts, 2, figure=fig)
        
        for i, trigger in enumerate(trigger_subset):
            start = max(0, trigger - window_samples//2)
            end = min(raw.n_times, trigger + window_samples//2)
            times = np.arange(start, end) / raw.info['sfreq']
            
            # Original data
            ax1 = fig.add_subplot(gs[i, 0])
            data_orig = raw_orig.get_data(start=start, stop=end)
            ax1.plot(times, data_orig.T)
            ax1.set_title(f'Original - Trigger {i+1}')
            
            # Corrected data
            ax2 = fig.add_subplot(gs[i, 1])
            data_corr = raw.get_data(start=start, stop=end)
            ax2.plot(times, data_corr.T)
            ax2.set_title(f'Corrected - Trigger {i+1}')
            
        plt.tight_layout()
        plt.show() 