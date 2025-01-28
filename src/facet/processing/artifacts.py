from typing import Optional, List, Dict, Union
import numpy as np
import mne
from loguru import logger
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ArtifactProcessor:
    """Handles artifact detection and removal."""
    
    def __init__(self, facet):
        """
        Initialize ArtifactProcessor.
        
        Args:
            facet: Parent FACET instance
        """
        self._facet = facet
        self._artifact_template: Optional[np.ndarray] = None
        self._artifact_matrix: Optional[Dict[int, np.ndarray]] = None
        
    def detect_artifacts(self, 
                        window_size: int = 30,
                        threshold: float = 0.975) -> 'ArtifactProcessor':
        """
        Detect artifacts using template matching.
        
        Args:
            window_size: Size of the sliding window
            threshold: Correlation threshold for artifact detection
        """
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        triggers = self._facet.get_metadata('triggers')
        if not triggers:
            raise ValueError("No triggers found. Run trigger detection first.")
            
        # Get artifact template
        self._create_artifact_template(window_size)
        
        # Calculate artifact matrix for each channel
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
        self._artifact_matrix = {}
        
        for ch_idx in eeg_channels:
            self._artifact_matrix[ch_idx] = self._calc_artifact_matrix(
                ch_idx, window_size, threshold
            )
            
        # Save to metadata
        self._facet.set_metadata('artifact_window_size', window_size)
        self._facet.set_metadata('artifact_threshold', threshold)
        
        return self
        
    def remove_artifacts(self) -> 'ArtifactProcessor':
        """Remove detected artifacts using template subtraction."""
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        if self._artifact_matrix is None:
            raise ValueError("No artifacts detected. Run detect_artifacts first.")
            
        triggers = self._facet.get_metadata('triggers')
        
        # Remove artifacts from each channel
        for ch_idx, matrix in self._artifact_matrix.items():
            self._remove_channel_artifacts(ch_idx, matrix, triggers)
            
        return self
        
    def _create_artifact_template(self, window_size: int) -> None:
        """Create artifact template from data around triggers."""
        raw = self._facet.data
        triggers = self._facet.get_metadata('triggers')
        
        # Use first EEG channel for template
        ch_idx = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)[0]
        data = raw.get_data(picks=[ch_idx])[0]
        
        # Extract segments around triggers
        segments = []
        for trigger in triggers:
            start = max(0, trigger - window_size//2)
            end = min(len(data), trigger + window_size//2)
            if end - start == window_size:
                segments.append(data[start:end])
                
        # Average segments to create template
        self._artifact_template = np.mean(segments, axis=0)
        
    def _calc_artifact_matrix(self, 
                            ch_idx: int, 
                            window_size: int,
                            threshold: float) -> np.ndarray:
        """Calculate artifact weighting matrix for a channel."""
        raw = self._facet.data
        data = raw.get_data(picks=[ch_idx])[0]
        triggers = self._facet.get_metadata('triggers')
        n_triggers = len(triggers)
        
        # Initialize weighting matrix
        weights = np.zeros((n_triggers, n_triggers))
        
        # Calculate correlations between epochs
        for i in range(n_triggers):
            for j in range(max(0, i-window_size), min(n_triggers, i+window_size)):
                if i == j:
                    continue
                    
                # Extract epochs
                epoch_i = self._extract_epoch(data, triggers[i])
                epoch_j = self._extract_epoch(data, triggers[j])
                
                # Calculate correlation
                corr = np.corrcoef(epoch_i, epoch_j)[0, 1]
                if corr > threshold:
                    weights[i, j] = corr
                    
        # Normalize weights
        row_sums = weights.sum(axis=1)
        weights[row_sums > 0] /= row_sums[row_sums > 0, np.newaxis]
        
        return weights
        
    def _extract_epoch(self, data: np.ndarray, trigger: int) -> np.ndarray:
        """Extract epoch around trigger."""
        start = max(0, trigger - self._facet.get_metadata('artifact_window_size')//2)
        end = min(len(data), trigger + self._facet.get_metadata('artifact_window_size')//2)
        return data[start:end]
        
    def _remove_channel_artifacts(self, 
                                ch_idx: int,
                                weights: np.ndarray,
                                triggers: List[int]) -> None:
        """Remove artifacts from a single channel."""
        raw = self._facet.data
        data = raw.get_data(picks=[ch_idx])[0]
        
        for i, trigger in enumerate(triggers):
            # Skip if no similar artifacts found
            if not np.any(weights[i]):
                continue
                
            # Calculate weighted average artifact
            artifact = np.zeros_like(self._artifact_template)
            for j, weight in enumerate(weights[i]):
                if weight > 0:
                    artifact += weight * self._extract_epoch(data, triggers[j])
                    
            # Subtract artifact
            start = max(0, trigger - len(self._artifact_template)//2)
            end = min(len(data), start + len(self._artifact_template))
            data[start:end] -= artifact[:end-start]
            
        # Update data
        raw._data[ch_idx] = data 