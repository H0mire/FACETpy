from typing import Dict, List, Optional, Union
import numpy as np
import mne
from loguru import logger
import matplotlib.pyplot as plt

class EvaluationProcessor:
    """Handles evaluation of EEG data quality and artifact correction."""
    
    def __init__(self, facet):
        """
        Initialize EvaluationProcessor.
        
        Args:
            facet: Parent FACET instance
        """
        self._facet = facet
        self._results: Dict[str, Dict] = {}
        
    def calculate_snr(self, name: str = "current") -> 'EvaluationProcessor':
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            name: Name for this evaluation result
        """
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        # Get artifact-free segments (between triggers)
        triggers = self._facet.get_metadata('triggers')
        if not triggers:
            raise ValueError("No triggers found")
            
        # Calculate power of signal and noise
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
        snr_values = []
        
        for ch_idx in eeg_channels:
            data = raw.get_data(picks=[ch_idx])[0]
            signal_power = self._calculate_signal_power(data, triggers)
            noise_power = self._calculate_noise_power(data, triggers)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
            snr_values.append(snr)
            
        self._results[name] = {
            'snr': np.mean(snr_values),
            'snr_per_channel': dict(zip(raw.ch_names, snr_values))
        }
        
        return self
        
    def calculate_rms(self, name: str = "current") -> 'EvaluationProcessor':
        """Calculate Root Mean Square of the signal."""
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
        rms_values = []
        
        for ch_idx in eeg_channels:
            data = raw.get_data(picks=[ch_idx])[0]
            rms = np.sqrt(np.mean(data ** 2))
            rms_values.append(rms)
            
        self._results[name] = {
            'rms': np.mean(rms_values),
            'rms_per_channel': dict(zip(raw.ch_names, rms_values))
        }
        
        return self
        
    def calculate_artifact_metrics(self, name: str = "current") -> 'EvaluationProcessor':
        """Calculate metrics specific to artifact correction quality."""
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        triggers = self._facet.get_metadata('triggers')
        if not triggers:
            raise ValueError("No triggers found")
            
        window_size = self._facet.get_metadata('artifact_window_size')
        if not window_size:
            raise ValueError("No artifact window size found")
            
        # Calculate artifact reduction metrics
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
        reduction_values = []
        
        for ch_idx in eeg_channels:
            data = raw.get_data(picks=[ch_idx])[0]
            reduction = self._calculate_artifact_reduction(data, triggers, window_size)
            reduction_values.append(reduction)
            
        self._results[name] = {
            'artifact_reduction': np.mean(reduction_values),
            'reduction_per_channel': dict(zip(raw.ch_names, reduction_values))
        }
        
        return self
        
    def plot_results(self, 
                    measures: List[str] = None,
                    names: List[str] = None) -> None:
        """
        Plot evaluation results.
        
        Args:
            measures: List of measures to plot ('snr', 'rms', 'artifact_reduction')
            names: List of result names to include
        """
        if not measures:
            measures = ['snr', 'rms', 'artifact_reduction']
            
        if not names:
            names = list(self._results.keys())
            
        n_measures = len(measures)
        fig, axes = plt.subplots(1, n_measures, figsize=(5*n_measures, 4))
        if n_measures == 1:
            axes = [axes]
            
        for ax, measure in zip(axes, measures):
            values = [self._results[name].get(measure, 0) for name in names]
            ax.bar(names, values)
            ax.set_title(measure.upper())
            ax.set_ylabel('Value')
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        plt.show()
        
    def _calculate_signal_power(self, data: np.ndarray, triggers: List[int]) -> float:
        """Calculate signal power from artifact-free segments."""
        mask = np.ones_like(data, dtype=bool)
        window = self._facet.get_metadata('artifact_window_size', 100)
        
        for trigger in triggers:
            start = max(0, trigger - window//2)
            end = min(len(data), trigger + window//2)
            mask[start:end] = False
            
        return np.var(data[mask])
        
    def _calculate_noise_power(self, data: np.ndarray, triggers: List[int]) -> float:
        """Calculate noise power from segments around triggers."""
        segments = []
        window = self._facet.get_metadata('artifact_window_size', 100)
        
        for trigger in triggers:
            start = max(0, trigger - window//2)
            end = min(len(data), trigger + window//2)
            segments.append(data[start:end])
            
        return np.var(np.concatenate(segments))
        
    def _calculate_artifact_reduction(self, 
                                   data: np.ndarray,
                                   triggers: List[int],
                                   window_size: int) -> float:
        """Calculate percentage of artifact reduction."""
        orig_data = self._facet._facet.raw_orig.get_data()[0]
        reduction = []
        
        for trigger in triggers:
            start = max(0, trigger - window_size//2)
            end = min(len(data), trigger + window_size//2)
            
            orig_power = np.var(orig_data[start:end])
            current_power = np.var(data[start:end])
            
            if orig_power > 0:
                reduction.append(1 - (current_power / orig_power))
                
        return np.mean(reduction) if reduction else 0.0 