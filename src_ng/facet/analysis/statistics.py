from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from scipy import stats
import mne
from loguru import logger
import pandas as pd

class StatisticsProcessor:
    """Handles statistical analysis of EEG data."""
    
    def __init__(self, facet):
        """
        Initialize StatisticsProcessor.
        
        Args:
            facet: Parent FACET instance
        """
        self._facet = facet
        
    def compute_channel_stats(self) -> pd.DataFrame:
        """Compute basic statistics for each channel."""
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
        stats_dict = {
            'channel': [],
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
            'kurtosis': [],
            'skewness': []
        }
        
        for ch_idx in eeg_channels:
            data = raw.get_data(picks=[ch_idx])[0]
            stats_dict['channel'].append(raw.ch_names[ch_idx])
            stats_dict['mean'].append(np.mean(data))
            stats_dict['std'].append(np.std(data))
            stats_dict['min'].append(np.min(data))
            stats_dict['max'].append(np.max(data))
            stats_dict['kurtosis'].append(stats.kurtosis(data))
            stats_dict['skewness'].append(stats.skew(data))
            
        return pd.DataFrame(stats_dict)
        
    def compute_artifact_stats(self) -> Dict[str, float]:
        """Compute statistics about artifacts."""
        triggers = self._facet.get_metadata('triggers')
        if not triggers:
            raise ValueError("No triggers found")
            
        # Calculate inter-trigger intervals
        intervals = np.diff(triggers) / self._facet.data.info['sfreq']
        
        return {
            'n_triggers': len(triggers),
            'mean_interval': np.mean(intervals),
            'std_interval': np.std(intervals),
            'min_interval': np.min(intervals),
            'max_interval': np.max(intervals)
        }
        
    def compute_frequency_stats(self, 
                              fmin: float = 0,
                              fmax: float = 100) -> pd.DataFrame:
        """
        Compute frequency domain statistics.
        
        Args:
            fmin: Minimum frequency to analyze
            fmax: Maximum frequency to analyze
        """
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        # Compute power spectral density
        psd, freqs = mne.time_frequency.psd_welch(
            raw,
            fmin=fmin,
            fmax=fmax,
            picks='eeg',
            n_fft=int(raw.info['sfreq'] * 2)
        )
        
        stats_dict = {
            'channel': [],
            'peak_freq': [],
            'peak_power': [],
            'mean_power': [],
            'median_power': [],
            'total_power': []
        }
        
        for ch_idx, ch_name in enumerate(raw.ch_names):
            if ch_idx >= psd.shape[0]:
                continue
                
            peak_idx = np.argmax(psd[ch_idx])
            stats_dict['channel'].append(ch_name)
            stats_dict['peak_freq'].append(freqs[peak_idx])
            stats_dict['peak_power'].append(psd[ch_idx][peak_idx])
            stats_dict['mean_power'].append(np.mean(psd[ch_idx]))
            stats_dict['median_power'].append(np.median(psd[ch_idx]))
            stats_dict['total_power'].append(np.sum(psd[ch_idx]))
            
        return pd.DataFrame(stats_dict)
        
    def test_artifact_removal(self) -> Dict[str, float]:
        """Perform statistical tests to evaluate artifact removal."""
        raw = self._facet.data
        if raw is None or self._facet.raw_orig is None:
            raise ValueError("Need both current and original data")
            
        triggers = self._facet.get_metadata('triggers')
        if not triggers:
            raise ValueError("No triggers found")
            
        window_size = self._facet.get_metadata('artifact_window_size', 100)
        
        # Collect artifact segments before and after correction
        orig_segments = []
        curr_segments = []
        
        for trigger in triggers:
            start = max(0, trigger - window_size//2)
            end = min(raw.n_times, trigger + window_size//2)
            
            orig_segments.append(
                self._facet.raw_orig.get_data(start=start, stop=end).ravel()
            )
            curr_segments.append(
                raw.get_data(start=start, stop=end).ravel()
            )
            
        # Perform statistical tests
        t_stat, p_value = stats.ttest_rel(
            np.concatenate(orig_segments),
            np.concatenate(curr_segments)
        )
        
        effect_size = self._compute_cohens_d(
            np.concatenate(orig_segments),
            np.concatenate(curr_segments)
        )
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size
        }
        
    def _compute_cohens_d(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(x1), len(x2)
        var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
        
        # Pooled standard deviation
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        return (np.mean(x1) - np.mean(x2)) / pooled_se 