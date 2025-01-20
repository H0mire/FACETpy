from typing import Optional, List, Dict, Union
import numpy as np
import mne
from loguru import logger
from scipy.signal import firls, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class CorrectionProcessor:
    """Handles artifact correction methods."""
    
    def __init__(self, facet):
        """
        Initialize CorrectionProcessor.
        
        Args:
            facet: Parent FACET instance
        """
        self._facet = facet
        self._artifact_templates: Dict[str, np.ndarray] = {}
        self._noise_estimates: Dict[str, np.ndarray] = {}
        
    def apply_anc(self, 
                  hp_freq: float = 70.0,
                  filter_order: Optional[int] = None) -> 'CorrectionProcessor':
        """
        Apply Adaptive Noise Cancellation.
        
        Args:
            hp_freq: Highpass frequency for artifact template
            filter_order: Filter order (if None, automatically determined)
        """
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        # Design highpass filter for artifact template
        if filter_order is None:
            filter_order = int(3.3 * raw.info['sfreq'] / hp_freq)
            if filter_order % 2 == 0:
                filter_order += 1
                
        nyq = 0.5 * raw.info['sfreq']
        freq = [0, 0.8*hp_freq/nyq, hp_freq/nyq, 1]
        gain = [0, 0, 1, 1]
        coeffs = firls(filter_order, freq, gain)
        
        # Process each channel
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
        for ch_idx in eeg_channels:
            data = raw.get_data(picks=[ch_idx])[0]
            
            # Create artifact template
            template = filtfilt(coeffs, 1, data)
            self._artifact_templates[raw.ch_names[ch_idx]] = template
            
            # Apply ANC
            clean_data = self._apply_anc_channel(data, template)
            raw._data[ch_idx] = clean_data
            
        # Save parameters to metadata
        self._facet.set_metadata('anc_hp_freq', hp_freq)
        self._facet.set_metadata('anc_filter_order', filter_order)
        
        return self
        
    def apply_obs(self, 
                  n_components: int = 3,
                  hp_freq: float = 70.0) -> 'CorrectionProcessor':
        """
        Apply Optimal Basis Set correction.
        
        Args:
            n_components: Number of PCA components to remove
            hp_freq: Highpass frequency for artifact template
        """
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        # Get artifact epochs
        triggers = self._facet.get_metadata('triggers')
        if not triggers:
            raise ValueError("No triggers found")
            
        window_size = self._facet.get_metadata('artifact_window_size', 100)
        
        # Process each channel
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
        for ch_idx in eeg_channels:
            data = raw.get_data(picks=[ch_idx])[0]
            
            # Extract artifact epochs
            epochs = []
            for trigger in triggers:
                start = max(0, trigger - window_size//2)
                end = min(len(data), trigger + window_size//2)
                if end - start == window_size:
                    epochs.append(data[start:end])
                    
            if not epochs:
                continue
                
            # Apply OBS correction
            epochs = np.array(epochs)
            clean_data = self._apply_obs_channel(
                data, epochs, triggers, window_size, n_components
            )
            raw._data[ch_idx] = clean_data
            
        # Save parameters to metadata
        self._facet.set_metadata('obs_n_components', n_components)
        self._facet.set_metadata('obs_hp_freq', hp_freq)
        
        return self
        
    def apply_ssa(self, 
                  window_size: int = 100,
                  n_components: int = 5) -> 'CorrectionProcessor':
        """
        Apply Singular Spectrum Analysis correction.
        
        Args:
            window_size: Size of the sliding window
            n_components: Number of components to remove
        """
        raw = self._facet.data
        if raw is None:
            raise ValueError("No EEG data loaded")
            
        # Process each channel
        eeg_channels = mne.pick_types(raw.info, meg=False, eeg=True, stim=False)
        for ch_idx in eeg_channels:
            data = raw.get_data(picks=[ch_idx])[0]
            
            # Apply SSA
            clean_data = self._apply_ssa_channel(data, window_size, n_components)
            raw._data[ch_idx] = clean_data
            
        # Save parameters to metadata
        self._facet.set_metadata('ssa_window_size', window_size)
        self._facet.set_metadata('ssa_n_components', n_components)
        
        return self
        
    def _apply_anc_channel(self, 
                          data: np.ndarray,
                          template: np.ndarray,
                          mu: float = 0.01) -> np.ndarray:
        """Apply ANC to a single channel."""
        w = np.zeros(1)  # Initial weight
        y = np.zeros_like(data)
        
        # LMS algorithm
        for i in range(len(data)):
            y[i] = w * template[i]
            e = data[i] - y[i]
            w = w + mu * e * template[i]
            
        return data - y
        
    def _apply_obs_channel(self,
                          data: np.ndarray,
                          epochs: np.ndarray,
                          triggers: List[int],
                          window_size: int,
                          n_components: int) -> np.ndarray:
        """Apply OBS to a single channel."""
        # PCA on artifact epochs
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(epochs)
        
        # Reconstruct artifacts
        artifacts = pca.inverse_transform(components)
        
        # Subtract artifacts
        clean_data = data.copy()
        for i, trigger in enumerate(triggers):
            if i >= len(artifacts):
                break
                
            start = max(0, trigger - window_size//2)
            end = min(len(data), trigger + window_size//2)
            if end - start == window_size:
                clean_data[start:end] -= artifacts[i]
                
        return clean_data
        
    def _apply_ssa_channel(self,
                          data: np.ndarray,
                          window_size: int,
                          n_components: int) -> np.ndarray:
        """Apply SSA to a single channel."""
        # Create trajectory matrix
        K = len(data) - window_size + 1
        X = np.zeros((window_size, K))
        for i in range(K):
            X[:, i] = data[i:i+window_size]
            
        # SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Reconstruct without artifact components
        X_clean = np.zeros_like(X)
        for i in range(n_components, len(S)):
            X_clean += S[i] * np.outer(U[:, i], Vt[i, :])
            
        # Average anti-diagonals to get clean signal
        clean_data = np.zeros_like(data)
        for i in range(len(data)):
            if i < window_size:
                clean_data[i] = np.mean(np.diag(X_clean[:i+1, :i+1][::-1]))
            elif i < len(data) - window_size:
                clean_data[i] = np.mean(np.diag(X_clean[:, i-window_size+1:i+1][::-1]))
            else:
                clean_data[i] = np.mean(np.diag(X_clean[-(len(data)-i):, -(len(data)-i):][::-1]))
                
        return clean_data 