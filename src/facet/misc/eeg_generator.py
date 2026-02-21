"""
EEG Generation Module

This module generates synthetic EEG data that can be used as ground truth for testing
artifact correction algorithms. It simulates realistic neural oscillations, 1/f noise,
and optional physiological artifacts.

Author: FACETpy Team
Date: 2025-01-12
"""

from typing import Optional, Dict, List, Tuple, Union
import mne
import numpy as np
from loguru import logger
from scipy import signal
from dataclasses import dataclass, field

from ..core import Processor, ProcessingContext, register_processor, ProcessorValidationError
from ..console import processor_progress


@dataclass
class ChannelSchema:
    """
    Schema defining the channel configuration for synthetic EEG generation.
    
    Attributes:
        eeg_channels: Number of EEG channels (default: 32)
        eog_channels: Number of EOG channels (default: 2)
        ecg_channels: Number of ECG channels (default: 1)
        emg_channels: Number of EMG channels (default: 0)
        misc_channels: Number of miscellaneous channels (default: 0)
        eeg_montage: MNE montage name for EEG electrode positions (default: 'standard_1020')
        eeg_channel_names: Optional custom EEG channel names
        eog_channel_names: Optional custom EOG channel names
    """
    eeg_channels: int = 32
    eog_channels: int = 2
    ecg_channels: int = 1
    emg_channels: int = 0
    misc_channels: int = 0
    eeg_montage: str = "standard_1020"
    eeg_channel_names: Optional[List[str]] = None
    eog_channel_names: Optional[List[str]] = None
    
    def get_total_channels(self) -> int:
        """Return total number of channels."""
        return (self.eeg_channels + self.eog_channels + self.ecg_channels + 
                self.emg_channels + self.misc_channels)


@dataclass 
class OscillationParams:
    """
    Parameters for neural oscillation generation.
    
    Attributes:
        delta: Amplitude for delta band (0.5-4 Hz)
        theta: Amplitude for theta band (4-8 Hz)
        alpha: Amplitude for alpha band (8-13 Hz)
        beta: Amplitude for beta band (13-30 Hz)
        gamma: Amplitude for gamma band (30-100 Hz)
    """
    delta: float = 20.0  # μV
    theta: float = 10.0  # μV
    alpha: float = 15.0  # μV - typically dominant at rest
    beta: float = 5.0    # μV
    gamma: float = 2.0   # μV
    
    
@dataclass
class NoiseParams:
    """
    Parameters for noise generation.
    
    Attributes:
        pink_noise_amplitude: Amplitude of 1/f (pink) noise
        white_noise_amplitude: Amplitude of measurement (white) noise
        line_noise_amplitude: Amplitude of power line noise (50/60 Hz)
        line_noise_freq: Frequency of power line noise
    """
    pink_noise_amplitude: float = 10.0  # μV
    white_noise_amplitude: float = 1.0  # μV
    line_noise_amplitude: float = 0.5   # μV
    line_noise_freq: float = 50.0       # Hz (50 Hz EU, 60 Hz US)


@dataclass
class ArtifactParams:
    """
    Parameters for physiological artifact simulation.
    
    Attributes:
        blink_rate: Average blinks per minute
        blink_amplitude: Amplitude of blink artifacts in EOG
        saccade_rate: Average saccades per minute
        saccade_amplitude: Amplitude of saccade artifacts
        heart_rate: Heart rate in BPM
        ecg_amplitude: Amplitude of ECG signal
        emg_amplitude: Amplitude of EMG noise
    """
    blink_rate: float = 15.0        # blinks per minute
    blink_amplitude: float = 100.0  # μV in EOG channels
    saccade_rate: float = 30.0      # saccades per minute
    saccade_amplitude: float = 50.0 # μV
    heart_rate: float = 70.0        # BPM
    ecg_amplitude: float = 500.0    # μV
    emg_amplitude: float = 5.0      # μV


@dataclass
class SpikeParams:
    """
    Parameters for epileptic spike simulation based on clinical morphology criteria.
    
    Simulates interictal epileptiform discharges (IEDs) using Gabor-wavelet based
    parametric modeling. This approach provides precise control over spike morphology
    while maintaining physiological plausibility.
    
    References:
        - IFCN criteria: Spike duration 20-70ms, Sharp Wave 70-200ms
        - Bergen Epileptiform Morphology Score (BEMS) parameters
        - Gabor-wavelet representation for optimal time-frequency localization
    
    Attributes:
        enabled: Flag to enable/disable spike generation
        spike_rate: Average spikes per minute (typical: 1-30 for interictal)
        spike_amplitude: Peak amplitude in μV (typical: 50-200 μV)
        spike_duration_ms: Duration of sharp transient in ms (20-70 for spike, 70-200 for sharp wave)
        asymmetry_ratio: Ratio of rise to fall time (>1 = steeper rise, typical: 1.5-3.0)
        slow_wave_enabled: Whether to add slow after-wave (IPSP component)
        slow_wave_amplitude_ratio: Amplitude of slow wave relative to spike (typical: 0.3-0.7)
        slow_wave_duration_ms: Duration of slow after-wave in ms (typical: 150-300)
        polyspike_probability: Probability of generating poly-spikes (0-1)
        polyspike_count_range: Range of spikes in a poly-spike complex
        amplitude_variability: Variation in amplitude across spikes (0-1)
        duration_variability: Variation in duration across spikes (0-1)
        focal_channels: List of channel names/indices for focal spikes (None = random)
        spatial_spread: How much spikes spread to neighboring channels (0-1)
        gabor_frequency_hz: Central frequency of Gabor wavelet (affects sharpness)
    """
    enabled: bool = False
    spike_rate: float = 10.0              # spikes per minute
    spike_amplitude: float = 100.0        # μV
    spike_duration_ms: float = 50.0       # ms (spike: 20-70, sharp wave: 70-200)
    asymmetry_ratio: float = 2.0          # rise/fall ratio
    slow_wave_enabled: bool = True
    slow_wave_amplitude_ratio: float = 0.5
    slow_wave_duration_ms: float = 200.0  # ms
    polyspike_probability: float = 0.1    # 10% chance of poly-spikes
    polyspike_count_range: Tuple[int, int] = (2, 4)
    amplitude_variability: float = 0.3    # 30% variation
    duration_variability: float = 0.2     # 20% variation
    focal_channels: Optional[List[Union[str, int]]] = None
    spatial_spread: float = 0.5           # 50% spread to neighbors
    gabor_frequency_hz: float = 25.0      # Hz - central frequency


def generate_pink_noise(n_samples: int, amplitude: float = 1.0) -> np.ndarray:
    """
    Generate 1/f (pink) noise using the Voss-McCartney algorithm.
    
    Pink noise has equal power per octave and is characteristic of
    natural neural background activity.
    
    Args:
        n_samples: Number of samples to generate
        amplitude: Amplitude scaling factor
        
    Returns:
        Pink noise signal array
    """
    # Use FFT-based method for efficiency
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1  # Avoid division by zero
    
    # 1/f spectrum
    spectrum = 1.0 / np.sqrt(freqs)
    spectrum[0] = 0  # DC component
    
    # Random phases
    phases = np.random.uniform(0, 2 * np.pi, len(spectrum))
    
    # Construct complex spectrum
    complex_spectrum = spectrum * np.exp(1j * phases)
    
    # Inverse FFT
    pink = np.fft.irfft(complex_spectrum, n_samples)
    
    # Normalize and scale
    pink = pink / np.std(pink) * amplitude
    
    return pink


def generate_oscillation(
    n_samples: int, 
    sfreq: float, 
    freq_range: Tuple[float, float], 
    amplitude: float,
    n_components: int = 3
) -> np.ndarray:
    """
    Generate neural oscillation in a specific frequency band.
    
    Uses multiple sinusoids with random frequencies within the band
    and random phases to create realistic band-limited activity.
    
    Args:
        n_samples: Number of samples
        sfreq: Sampling frequency in Hz
        freq_range: Tuple of (low_freq, high_freq) in Hz
        amplitude: Peak amplitude
        n_components: Number of sinusoidal components
        
    Returns:
        Oscillation signal array
    """
    t = np.arange(n_samples) / sfreq
    oscillation = np.zeros(n_samples)
    
    # Generate multiple components within the frequency range
    freqs = np.random.uniform(freq_range[0], freq_range[1], n_components)
    phases = np.random.uniform(0, 2 * np.pi, n_components)
    amps = np.random.uniform(0.5, 1.0, n_components)
    
    for freq, phase, amp in zip(freqs, phases, amps):
        oscillation += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Normalize and scale
    oscillation = oscillation / np.max(np.abs(oscillation)) * amplitude
    
    return oscillation


def generate_blink_artifact(
    n_samples: int, 
    sfreq: float, 
    blink_rate: float, 
    amplitude: float
) -> np.ndarray:
    """
    Generate realistic eye blink artifacts.
    
    Blinks are modeled as asymmetric peaks with rapid onset and slower decay,
    occurring at random intervals with the specified average rate.
    
    Args:
        n_samples: Number of samples
        sfreq: Sampling frequency
        blink_rate: Average blinks per minute
        amplitude: Peak amplitude
        
    Returns:
        Blink artifact signal
    """
    duration = n_samples / sfreq
    n_blinks = int(blink_rate * duration / 60)
    
    blink_signal = np.zeros(n_samples)
    
    if n_blinks == 0:
        return blink_signal
    
    # Random blink times
    blink_times = np.sort(np.random.uniform(0.5, duration - 0.5, n_blinks))
    
    # Blink template: asymmetric Gaussian-like shape
    blink_duration = 0.3  # seconds
    blink_samples = int(blink_duration * sfreq)
    t_blink = np.linspace(0, blink_duration, blink_samples)
    
    # Asymmetric peak (faster rise, slower decay)
    rise_tau = 0.05
    decay_tau = 0.15
    template = np.where(
        t_blink < blink_duration * 0.3,
        np.exp(-((t_blink - blink_duration * 0.3) ** 2) / (2 * rise_tau ** 2)),
        np.exp(-((t_blink - blink_duration * 0.3) ** 2) / (2 * decay_tau ** 2))
    )
    
    for blink_time in blink_times:
        start_idx = int(blink_time * sfreq)
        end_idx = min(start_idx + blink_samples, n_samples)
        template_end = end_idx - start_idx
        
        # Add random amplitude variation
        amp_variation = np.random.uniform(0.7, 1.3)
        blink_signal[start_idx:end_idx] += template[:template_end] * amplitude * amp_variation
    
    return blink_signal


def generate_ecg_artifact(
    n_samples: int, 
    sfreq: float, 
    heart_rate: float, 
    amplitude: float
) -> np.ndarray:
    """
    Generate realistic ECG signal with QRS complex.
    
    Models the characteristic PQRST waveform of cardiac activity.
    
    Args:
        n_samples: Number of samples
        sfreq: Sampling frequency
        heart_rate: Heart rate in BPM
        amplitude: R-peak amplitude
        
    Returns:
        ECG signal array
    """
    duration = n_samples / sfreq
    n_beats = int(heart_rate * duration / 60) + 1
    
    ecg_signal = np.zeros(n_samples)
    
    # Average R-R interval with some variability (HRV)
    mean_rr = 60.0 / heart_rate
    rr_intervals = np.random.normal(mean_rr, mean_rr * 0.05, n_beats)
    
    # Generate beat times
    beat_times = np.cumsum(rr_intervals)
    beat_times = beat_times[beat_times < duration]
    
    # QRS complex template (simplified)
    qrs_duration = 0.1  # seconds
    qrs_samples = int(qrs_duration * sfreq)
    t_qrs = np.linspace(-qrs_duration/2, qrs_duration/2, qrs_samples)
    
    # Simplified QRS: sharp peak with small surrounding deflections
    qrs_template = (
        -0.1 * np.exp(-((t_qrs + 0.02) ** 2) / (2 * 0.005 ** 2)) +  # Q wave
        1.0 * np.exp(-(t_qrs ** 2) / (2 * 0.008 ** 2)) +             # R wave
        -0.2 * np.exp(-((t_qrs - 0.025) ** 2) / (2 * 0.008 ** 2))    # S wave
    )
    
    for beat_time in beat_times:
        center_idx = int(beat_time * sfreq)
        start_idx = max(0, center_idx - qrs_samples // 2)
        end_idx = min(n_samples, center_idx + qrs_samples // 2)
        
        template_start = max(0, qrs_samples // 2 - center_idx)
        template_end = template_start + (end_idx - start_idx)
        
        if template_end <= qrs_samples:
            ecg_signal[start_idx:end_idx] += qrs_template[template_start:template_end] * amplitude
    
    return ecg_signal


def generate_saccade_artifact(
    n_samples: int, 
    sfreq: float, 
    saccade_rate: float, 
    amplitude: float
) -> np.ndarray:
    """
    Generate horizontal eye movement (saccade) artifacts.
    
    Saccades create step-like changes in EOG signal.
    
    Args:
        n_samples: Number of samples
        sfreq: Sampling frequency
        saccade_rate: Saccades per minute
        amplitude: Step amplitude
        
    Returns:
        Saccade artifact signal
    """
    duration = n_samples / sfreq
    n_saccades = int(saccade_rate * duration / 60)
    
    saccade_signal = np.zeros(n_samples)
    
    if n_saccades == 0:
        return saccade_signal
    
    saccade_times = np.sort(np.random.uniform(0.2, duration - 0.2, n_saccades))
    
    for saccade_time in saccade_times:
        idx = int(saccade_time * sfreq)
        direction = np.random.choice([-1, 1])
        step_amplitude = direction * amplitude * np.random.uniform(0.3, 1.0)
        
        # Rapid step with slight overshoot
        step_duration = int(0.05 * sfreq)
        overshoot_duration = int(0.1 * sfreq)
        
        end_idx = min(idx + step_duration + overshoot_duration, n_samples)
        
        if idx + step_duration < n_samples:
            # Rapid transition
            saccade_signal[idx:idx + step_duration] = np.linspace(
                0, step_amplitude * 1.1, step_duration
            )
            # Settle to final position
            if idx + step_duration + overshoot_duration <= n_samples:
                saccade_signal[idx + step_duration:end_idx] = np.linspace(
                    step_amplitude * 1.1, step_amplitude, overshoot_duration
                )
    
    # Cumulative effect (eye position is integral of velocity)
    saccade_signal = np.cumsum(saccade_signal) * 0.01
    
    # Remove DC drift
    saccade_signal = signal.detrend(saccade_signal)
    
    return saccade_signal


def generate_gabor_spike(
    n_samples: int,
    sfreq: float,
    duration_ms: float = 50.0,
    frequency_hz: float = 25.0,
    asymmetry_ratio: float = 2.0,
    phase: float = 0.0
) -> np.ndarray:
    """
    Generate a single epileptic spike using Gabor wavelet representation.
    
    Gabor functions provide optimal time-frequency localization (minimize 
    Heisenberg uncertainty) and are well-suited for representing transient
    neural events like epileptic spikes.
    
    The spike is modeled as a Gaussian-modulated sinusoid with asymmetric
    envelope to capture the characteristic steeper rise and slower decay
    of epileptic discharges.
    
    Args:
        n_samples: Number of samples for the spike template
        sfreq: Sampling frequency in Hz
        duration_ms: Total duration of the spike in milliseconds
        frequency_hz: Central frequency of the Gabor wavelet (affects sharpness)
        asymmetry_ratio: Ratio of decay to rise time (>1 = steeper rise)
        phase: Phase offset in radians (π/2 gives odd symmetry like real spikes)
        
    Returns:
        Spike template array normalized to peak amplitude of 1.0
        
    Reference:
        Gabor representation: g(t) = exp(-t²/2σ²) * cos(2πf₀t + φ)
    """
    duration_s = duration_ms / 1000.0
    t = np.linspace(-duration_s / 2, duration_s / 2, n_samples)
    
    # Asymmetric Gaussian envelope
    # σ_rise for t < 0, σ_decay for t >= 0
    sigma_base = duration_s / 6  # Base sigma so ~99% energy within duration
    sigma_rise = sigma_base / asymmetry_ratio
    sigma_decay = sigma_base * (asymmetry_ratio / 2)
    
    envelope = np.where(
        t < 0,
        np.exp(-(t ** 2) / (2 * sigma_rise ** 2)),
        np.exp(-(t ** 2) / (2 * sigma_decay ** 2))
    )
    
    # Sinusoidal carrier with phase
    carrier = np.cos(2 * np.pi * frequency_hz * t + phase)
    
    # Gabor atom
    spike = envelope * carrier
    
    # Normalize to unit peak
    spike = spike / np.max(np.abs(spike))
    
    return spike


def generate_slow_wave(
    n_samples: int,
    sfreq: float,
    duration_ms: float = 200.0,
    polarity: int = -1
) -> np.ndarray:
    """
    Generate the slow after-wave component following an epileptic spike.
    
    The slow wave represents the inhibitory postsynaptic potential (IPSP)
    that follows the excitatory synchronization of the spike. It typically
    has opposite polarity to the main spike deflection.
    
    Args:
        n_samples: Number of samples for the slow wave
        sfreq: Sampling frequency in Hz
        duration_ms: Duration of the slow wave in milliseconds
        polarity: Polarity of the wave (-1 or +1)
        
    Returns:
        Slow wave template normalized to peak amplitude of 1.0
    """
    duration_s = duration_ms / 1000.0
    t = np.linspace(0, duration_s, n_samples)
    
    # Asymmetric slow wave: gradual rise, slightly faster decay
    # Using a gamma-like function for natural shape
    alpha = 2.5  # Shape parameter
    beta = duration_s / 3  # Scale parameter
    
    wave = (t / beta) ** alpha * np.exp(-t / beta)
    
    # Normalize and apply polarity
    wave = wave / np.max(np.abs(wave)) * polarity
    
    return wave


def generate_spike_wave_complex(
    sfreq: float,
    spike_params: SpikeParams,
    amplitude: float = 100.0
) -> Tuple[np.ndarray, int]:
    """
    Generate a complete spike-wave complex (SWC).
    
    Combines a sharp spike transient with an optional slow after-wave,
    and can generate poly-spike complexes based on configuration.
    
    Args:
        sfreq: Sampling frequency in Hz
        spike_params: SpikeParams configuration
        amplitude: Peak amplitude in μV
        
    Returns:
        Tuple of (spike_wave_complex array, center sample index)
    """
    # Add variability to parameters
    duration_ms = spike_params.spike_duration_ms * (
        1 + np.random.uniform(-spike_params.duration_variability, 
                              spike_params.duration_variability)
    )
    amp = amplitude * (
        1 + np.random.uniform(-spike_params.amplitude_variability,
                              spike_params.amplitude_variability)
    )
    
    # Spike template samples
    spike_samples = int(duration_ms * sfreq / 1000)
    spike_samples = max(spike_samples, 10)  # Minimum samples
    
    # Generate main spike with slight phase variation for natural morphology
    phase = np.pi / 2 + np.random.uniform(-0.3, 0.3)  # Odd symmetry + variation
    
    spike = generate_gabor_spike(
        n_samples=spike_samples,
        sfreq=sfreq,
        duration_ms=duration_ms,
        frequency_hz=spike_params.gabor_frequency_hz,
        asymmetry_ratio=spike_params.asymmetry_ratio,
        phase=phase
    )
    
    # Check for poly-spike
    if np.random.random() < spike_params.polyspike_probability:
        n_spikes = np.random.randint(
            spike_params.polyspike_count_range[0],
            spike_params.polyspike_count_range[1] + 1
        )
        poly_spike = spike.copy()
        
        for i in range(1, n_spikes):
            # Subsequent spikes are slightly smaller and shifted
            sub_amp = np.random.uniform(0.5, 0.9)
            sub_phase = phase + np.random.uniform(-0.5, 0.5)
            sub_spike = generate_gabor_spike(
                n_samples=spike_samples,
                sfreq=sfreq,
                duration_ms=duration_ms * np.random.uniform(0.8, 1.0),
                frequency_hz=spike_params.gabor_frequency_hz,
                asymmetry_ratio=spike_params.asymmetry_ratio,
                phase=sub_phase
            ) * sub_amp
            
            # Shift and add - inter-spike interval within poly-spike
            inter_spike_gap = int(spike_samples * 0.3)  # 30% overlap
            
            # Extend array and add sub-spike
            start_idx = len(poly_spike) - inter_spike_gap
            start_idx = max(0, start_idx)
            
            # Ensure we have room for the sub-spike
            needed_length = start_idx + len(sub_spike)
            if needed_length > len(poly_spike):
                poly_spike = np.concatenate([poly_spike, np.zeros(needed_length - len(poly_spike))])
            
            # Add sub-spike
            end_idx = start_idx + len(sub_spike)
            poly_spike[start_idx:end_idx] += sub_spike
        
        spike = poly_spike
        spike_samples = len(spike)
    
    # Scale spike
    spike = spike * amp
    
    # Add slow wave if enabled
    if spike_params.slow_wave_enabled:
        sw_duration_ms = spike_params.slow_wave_duration_ms * (
            1 + np.random.uniform(-0.2, 0.2)
        )
        sw_samples = int(sw_duration_ms * sfreq / 1000)
        sw_amp = amp * spike_params.slow_wave_amplitude_ratio
        
        # Determine slow wave polarity (opposite to main spike deflection)
        spike_polarity = 1 if np.sum(spike) > 0 else -1
        slow_wave = generate_slow_wave(
            n_samples=sw_samples,
            sfreq=sfreq,
            duration_ms=sw_duration_ms,
            polarity=-spike_polarity
        ) * sw_amp
        
        # Combine spike and slow wave
        complex_signal = np.concatenate([spike, slow_wave])
    else:
        complex_signal = spike
    
    # Center index (for alignment)
    center_idx = spike_samples // 2
    
    return complex_signal, center_idx


def generate_epileptic_spikes(
    n_samples: int,
    sfreq: float,
    n_channels: int,
    spike_params: SpikeParams,
    channel_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generate epileptic spike artifacts for multiple EEG channels.
    
    Creates realistic interictal epileptiform discharges (IEDs) with:
    - Focal onset with spatial propagation
    - Variable morphology (spikes, sharp waves, poly-spikes)
    - Spike-wave complexes with slow after-waves
    - Clinically plausible timing and amplitude distributions
    
    Args:
        n_samples: Total number of samples
        sfreq: Sampling frequency in Hz
        n_channels: Number of EEG channels
        spike_params: SpikeParams configuration
        channel_names: Optional list of channel names for focal targeting
        
    Returns:
        Tuple of:
            - Array of shape (n_channels, n_samples) with spike signals
            - List of spike event dictionaries with timing and metadata
    """
    if not spike_params.enabled:
        return np.zeros((n_channels, n_samples)), []
    
    duration = n_samples / sfreq
    n_spikes = int(spike_params.spike_rate * duration / 60)
    
    spike_signals = np.zeros((n_channels, n_samples))
    spike_events = []
    
    if n_spikes == 0:
        return spike_signals, spike_events
    
    # Generate random spike times with minimum inter-spike interval
    min_isi = 0.5  # Minimum 500ms between spikes
    spike_times = []
    
    current_time = np.random.uniform(0.5, min(2.0, duration / 2))
    while current_time < duration - 0.5 and len(spike_times) < n_spikes:
        spike_times.append(current_time)
        # Variable ISI with minimum
        isi = np.random.exponential(60.0 / spike_params.spike_rate)
        current_time += max(isi, min_isi)
    
    # Determine focal channels
    if spike_params.focal_channels is not None:
        # Convert names to indices if necessary
        focal_indices = []
        for ch in spike_params.focal_channels:
            if isinstance(ch, int):
                if 0 <= ch < n_channels:
                    focal_indices.append(ch)
            elif channel_names is not None and ch in channel_names:
                focal_indices.append(channel_names.index(ch))
        if not focal_indices:
            focal_indices = [np.random.randint(0, n_channels)]
    else:
        # Random focal channel(s)
        n_focal = np.random.randint(1, min(4, n_channels))
        focal_indices = list(np.random.choice(n_channels, n_focal, replace=False))
    
    logger.debug(f"Generating {len(spike_times)} epileptic spikes on {len(focal_indices)} focal channels")
    
    for spike_time in spike_times:
        # Generate spike-wave complex
        swc, center_idx = generate_spike_wave_complex(
            sfreq=sfreq,
            spike_params=spike_params,
            amplitude=spike_params.spike_amplitude
        )
        
        # Calculate insertion point
        start_sample = int(spike_time * sfreq) - center_idx
        end_sample = start_sample + len(swc)
        
        # Boundary checks
        if start_sample < 0:
            swc = swc[-start_sample:]
            start_sample = 0
        if end_sample > n_samples:
            swc = swc[:n_samples - start_sample]
            end_sample = n_samples
        
        if len(swc) == 0:
            continue
        
        # Randomly select focal channel for this spike
        primary_channel = np.random.choice(focal_indices)
        
        # Add to primary channel
        spike_signals[primary_channel, start_sample:end_sample] += swc
        
        # Spatial spread to neighboring channels
        if spike_params.spatial_spread > 0:
            for ch_idx in range(n_channels):
                if ch_idx == primary_channel:
                    continue
                
                # Distance-based attenuation (simplified - assumes linear arrangement)
                distance = abs(ch_idx - primary_channel)
                attenuation = spike_params.spatial_spread * np.exp(-distance / 3)
                
                if attenuation > 0.05:  # Threshold for adding
                    # Add phase jitter for realistic propagation
                    jitter_samples = int(np.random.uniform(0, 5))
                    shifted_start = min(start_sample + jitter_samples, n_samples - len(swc))
                    shifted_end = shifted_start + len(swc)
                    
                    if shifted_end <= n_samples:
                        spike_signals[ch_idx, shifted_start:shifted_end] += swc * attenuation
        
        # Record event
        spike_events.append({
            'time': spike_time,
            'sample': int(spike_time * sfreq),
            'channel': primary_channel,
            'channel_name': channel_names[primary_channel] if channel_names else f'CH{primary_channel}',
            'amplitude': float(np.max(np.abs(swc))),
            'duration_ms': len(swc) / sfreq * 1000
        })
    
    return spike_signals, spike_events


def generate_synthetic_eeg(
    duration: float,
    sfreq: float,
    channel_schema: ChannelSchema,
    oscillation_params: Optional[OscillationParams] = None,
    noise_params: Optional[NoiseParams] = None,
    artifact_params: Optional[ArtifactParams] = None,
    spike_params: Optional[SpikeParams] = None,
    random_seed: Optional[int] = None
) -> Tuple[mne.io.RawArray, List[Dict]]:
    """
    Generate a complete synthetic EEG dataset as an MNE Raw object.
    
    Creates realistic multi-channel EEG data including:
    - Neural oscillations in standard frequency bands
    - 1/f (pink) background noise
    - White measurement noise
    - Optional power line noise
    - EOG artifacts (blinks and saccades)
    - ECG signal
    - EMG noise
    - Optional epileptic spikes (interictal epileptiform discharges)
    
    Args:
        duration: Duration of recording in seconds
        sfreq: Sampling frequency in Hz
        channel_schema: ChannelSchema defining channel configuration
        oscillation_params: OscillationParams for neural activity (default values if None)
        noise_params: NoiseParams for noise generation (default values if None)
        artifact_params: ArtifactParams for artifact simulation (default values if None)
        spike_params: SpikeParams for epileptic spike simulation (disabled if None)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
            - MNE RawArray object containing the synthetic EEG data
            - List of spike event dictionaries (empty if spikes disabled)
        
    Example:
        >>> schema = ChannelSchema(eeg_channels=64, eog_channels=2, ecg_channels=1)
        >>> # Without epileptic spikes
        >>> raw, events = generate_synthetic_eeg(
        ...     duration=60.0,
        ...     sfreq=1000,
        ...     channel_schema=schema
        ... )
        >>> # With epileptic spikes enabled
        >>> spike_config = SpikeParams(enabled=True, spike_rate=15.0, spike_amplitude=150.0)
        >>> raw, events = generate_synthetic_eeg(
        ...     duration=60.0,
        ...     sfreq=1000,
        ...     channel_schema=schema,
        ...     spike_params=spike_config
        ... )
        >>> print(f"Generated {len(events)} spike events")
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if oscillation_params is None:
        oscillation_params = OscillationParams()
    if noise_params is None:
        noise_params = NoiseParams()
    if artifact_params is None:
        artifact_params = ArtifactParams()
    if spike_params is None:
        spike_params = SpikeParams(enabled=False)

    n_samples = int(duration * sfreq)
    n_total_channels = channel_schema.get_total_channels()

    spike_info = "with epileptic spikes" if spike_params.enabled else "without spikes"
    logger.info(f"Generating synthetic EEG: {duration}s, {sfreq}Hz, {n_total_channels} channels ({spike_info})")

    data = np.zeros((n_total_channels, n_samples))
    ch_names = []
    ch_types = []
    
    # === Generate EEG channels ===
    if channel_schema.eeg_channel_names:
        eeg_names = channel_schema.eeg_channel_names[:channel_schema.eeg_channels]
    else:
        # Standard 10-20 system names based on channel count
        standard_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
            'T7', 'C3', 'Cz', 'C4', 'T8',
            'P7', 'P3', 'Pz', 'P4', 'P8',
            'O1', 'Oz', 'O2',
            'FC1', 'FC2', 'FC5', 'FC6',
            'CP1', 'CP2', 'CP5', 'CP6',
            'AF3', 'AF4', 'AF7', 'AF8',
            'F1', 'F2', 'F5', 'F6',
            'C1', 'C2', 'C5', 'C6',
            'P1', 'P2', 'P5', 'P6',
            'PO3', 'PO4', 'PO7', 'PO8',
            'FT7', 'FT8', 'TP7', 'TP8',
            'PO9', 'PO10', 'Fpz', 'CPz',
            'POz', 'FCz', 'FT9', 'FT10',
            'TP9', 'TP10', 'P9', 'P10',
        ]
        eeg_names = standard_names[:channel_schema.eeg_channels]
        # If we need more channels than standard names
        if channel_schema.eeg_channels > len(standard_names):
            for i in range(len(standard_names), channel_schema.eeg_channels):
                eeg_names.append(f'EEG{i+1:03d}')
    
    ch_names.extend(eeg_names)
    ch_types.extend(['eeg'] * channel_schema.eeg_channels)
    
    # Frequency bands
    bands = [
        ((0.5, 4.0), oscillation_params.delta),   # Delta
        ((4.0, 8.0), oscillation_params.theta),   # Theta
        ((8.0, 13.0), oscillation_params.alpha),  # Alpha
        ((13.0, 30.0), oscillation_params.beta),  # Beta
        ((30.0, 100.0), oscillation_params.gamma) # Gamma
    ]
    
    for ch_idx in range(channel_schema.eeg_channels):
        eeg_signal = np.zeros(n_samples)
        
        for freq_range, amp in bands:
            # Vary amplitude across channels (posterior alpha, frontal theta, etc.)
            spatial_factor = np.random.uniform(0.5, 1.5)
            eeg_signal += generate_oscillation(n_samples, sfreq, freq_range, amp * spatial_factor)
        
        eeg_signal += generate_pink_noise(n_samples, noise_params.pink_noise_amplitude)
        eeg_signal += np.random.normal(0, noise_params.white_noise_amplitude, n_samples)

        if noise_params.line_noise_amplitude > 0:
            t = np.arange(n_samples) / sfreq
            line_phase = np.random.uniform(0, 2 * np.pi)
            eeg_signal += noise_params.line_noise_amplitude * np.sin(
                2 * np.pi * noise_params.line_noise_freq * t + line_phase
            )
        
        data[ch_idx] = eeg_signal
    
    current_idx = channel_schema.eeg_channels
    
    # === Generate EOG channels ===
    if channel_schema.eog_channels > 0:
        if channel_schema.eog_channel_names:
            eog_names = channel_schema.eog_channel_names[:channel_schema.eog_channels]
        else:
            eog_names = ['EOG1', 'EOG2'][:channel_schema.eog_channels]
            if channel_schema.eog_channels > 2:
                eog_names.extend([f'EOG{i+1}' for i in range(2, channel_schema.eog_channels)])
        
        ch_names.extend(eog_names)
        ch_types.extend(['eog'] * channel_schema.eog_channels)
        
        blink_signal = generate_blink_artifact(
            n_samples, sfreq, 
            artifact_params.blink_rate, 
            artifact_params.blink_amplitude
        )
        
        saccade_signal = generate_saccade_artifact(
            n_samples, sfreq,
            artifact_params.saccade_rate,
            artifact_params.saccade_amplitude
        )
        
        for i in range(channel_schema.eog_channels):
            eog_signal = np.zeros(n_samples)
            
            eog_signal += blink_signal * np.random.uniform(0.8, 1.2)
            polarity = 1 if i % 2 == 0 else -1  # opposite polarity for left/right
            eog_signal += saccade_signal * polarity
            eog_signal += generate_pink_noise(n_samples, 5.0)
            eog_signal += np.random.normal(0, 2.0, n_samples)
            
            data[current_idx + i] = eog_signal
        
        current_idx += channel_schema.eog_channels
        
        # Propagate EOG artifacts to frontal EEG channels (volume conduction)
        frontal_channels = [i for i, name in enumerate(eeg_names) 
                          if any(x in name.upper() for x in ['FP', 'AF', 'F1', 'F2', 'FZ'])]
        for ch_idx in frontal_channels:
            propagation_factor = np.random.uniform(0.1, 0.3)
            data[ch_idx] += blink_signal * propagation_factor
    
    # === Generate ECG channels ===
    if channel_schema.ecg_channels > 0:
        ecg_names = [f'ECG{i+1}' for i in range(channel_schema.ecg_channels)]
        ch_names.extend(ecg_names)
        ch_types.extend(['ecg'] * channel_schema.ecg_channels)
        
        ecg_signal = generate_ecg_artifact(
            n_samples, sfreq,
            artifact_params.heart_rate,
            artifact_params.ecg_amplitude
        )
        
        for i in range(channel_schema.ecg_channels):
            # Slightly different for each ECG channel
            ecg_variation = ecg_signal * np.random.uniform(0.9, 1.1)
            ecg_variation += np.random.normal(0, 5.0, n_samples)
            data[current_idx + i] = ecg_variation
        
        current_idx += channel_schema.ecg_channels
        
        # BCG artifact: ECG propagates weakly to all EEG channels
        for ch_idx in range(channel_schema.eeg_channels):
            bcg_factor = np.random.uniform(0.01, 0.05)
            data[ch_idx] += ecg_signal * bcg_factor
    
    # === Generate Epileptic Spikes (if enabled) ===
    spike_events = []
    if spike_params.enabled and channel_schema.eeg_channels > 0:
        spike_signals, spike_events = generate_epileptic_spikes(
            n_samples=n_samples,
            sfreq=sfreq,
            n_channels=channel_schema.eeg_channels,
            spike_params=spike_params,
            channel_names=eeg_names
        )
        
        data[:channel_schema.eeg_channels] += spike_signals
        
        logger.info(f"Added {len(spike_events)} epileptic spikes to EEG data")
    
    # === Generate EMG channels ===
    if channel_schema.emg_channels > 0:
        emg_names = [f'EMG{i+1}' for i in range(channel_schema.emg_channels)]
        ch_names.extend(emg_names)
        ch_types.extend(['emg'] * channel_schema.emg_channels)
        
        for i in range(channel_schema.emg_channels):
            # EMG is broadband high-frequency activity
            emg_signal = np.random.normal(0, artifact_params.emg_amplitude, n_samples)
            
            n_bursts = np.random.randint(5, 20)
            for _ in range(n_bursts):
                burst_start = np.random.randint(0, n_samples - int(sfreq))
                burst_duration = np.random.randint(int(0.1 * sfreq), int(0.5 * sfreq))
                burst_end = min(burst_start + burst_duration, n_samples)
                burst_amp = np.random.uniform(2, 5)
                emg_signal[burst_start:burst_end] *= burst_amp
            
            # High-pass filter to remove low frequencies
            sos = signal.butter(4, 20, btype='high', fs=sfreq, output='sos')
            emg_signal = signal.sosfilt(sos, emg_signal)
            
            data[current_idx + i] = emg_signal
        
        current_idx += channel_schema.emg_channels
    
    # === Generate MISC channels ===
    if channel_schema.misc_channels > 0:
        misc_names = [f'MISC{i+1}' for i in range(channel_schema.misc_channels)]
        ch_names.extend(misc_names)
        ch_types.extend(['misc'] * channel_schema.misc_channels)
        
        for i in range(channel_schema.misc_channels):
            # Random noise for misc channels
            data[current_idx + i] = np.random.normal(0, 10, n_samples)
        
        current_idx += channel_schema.misc_channels
    
    data = data * 1e-6  # MNE expects Volts; we generated in μV

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )
    
    raw = mne.io.RawArray(data, info, verbose=False)
    
    # Try to set montage for EEG channels
    try:
        montage = mne.channels.make_standard_montage(channel_schema.eeg_montage)
        raw.set_montage(montage, on_missing='ignore')
    except Exception as e:
        logger.debug(f"Could not set montage: {e}")
    
    logger.info(f"Generated synthetic EEG: {raw}")
    
    return raw, spike_events


@register_processor
class EEGGenerator(Processor):
    """
    EEG Generator processor for creating synthetic EEG data.
    
    Generates realistic multi-channel EEG data including neural oscillations
    in standard frequency bands, 1/f background noise, and optional 
    physiological artifacts (EOG, ECG, EMG) and epileptic spikes.
    
    Parameters:
        sampling_rate: Sampling frequency in Hz (default: 1000)
        duration: Recording duration in seconds (default: 10.0)
        channel_schema: ChannelSchema object or dict with channel configuration
        oscillation_params: OscillationParams or dict for neural oscillations
        noise_params: NoiseParams or dict for noise parameters
        artifact_params: ArtifactParams or dict for artifact simulation
        spike_params: SpikeParams or dict for epileptic spike simulation
        random_seed: Random seed for reproducibility
        
    Example:
        >>> # Without epileptic spikes
        >>> generator = EEGGenerator(
        ...     duration=60.0,
        ...     channel_schema={'eeg_channels': 64, 'eog_channels': 2}
        ... )
        >>> context = generator.process(context)
        >>> raw = context.raw  # Generated synthetic EEG
        
        >>> # With epileptic spikes enabled
        >>> generator = EEGGenerator(
        ...     duration=60.0,
        ...     channel_schema={'eeg_channels': 64},
        ...     spike_params={'enabled': True, 'spike_rate': 15.0, 'spike_amplitude': 150.0}
        ... )
        >>> context = generator.process(context)
        >>> spike_events = context.metadata['eeg_generator']['spike_events']
    """
    name = "eeg_generator"
    description = "Generates synthetic EEG data with realistic neural oscillations, artifacts, and optional epileptic spikes."
    requires_raw = False
    requires_triggers = False
    parallel_safe = True
    parallelize_by_channels = False

    def __init__(
        self, 
        sampling_rate: int = 1000, 
        duration: float = 10.0,
        channel_schema: Optional[Union[ChannelSchema, Dict]] = None,
        oscillation_params: Optional[Union[OscillationParams, Dict]] = None,
        noise_params: Optional[Union[NoiseParams, Dict]] = None,
        artifact_params: Optional[Union[ArtifactParams, Dict]] = None,
        spike_params: Optional[Union[SpikeParams, Dict]] = None,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sampling_rate = sampling_rate
        self.duration = duration
        self.random_seed = random_seed
        
        if isinstance(channel_schema, dict):
            self.channel_schema = ChannelSchema(**channel_schema)
        elif channel_schema is None:
            self.channel_schema = ChannelSchema()
        else:
            self.channel_schema = channel_schema
            
        if isinstance(oscillation_params, dict):
            self.oscillation_params = OscillationParams(**oscillation_params)
        else:
            self.oscillation_params = oscillation_params
            
        if isinstance(noise_params, dict):
            self.noise_params = NoiseParams(**noise_params)
        else:
            self.noise_params = noise_params
            
        if isinstance(artifact_params, dict):
            self.artifact_params = ArtifactParams(**artifact_params)
        else:
            self.artifact_params = artifact_params
            
        if isinstance(spike_params, dict):
            self.spike_params = SpikeParams(**spike_params)
        elif spike_params is None:
            self.spike_params = SpikeParams(enabled=False)
        else:
            self.spike_params = spike_params

    def validate(self, context: ProcessingContext) -> None:
        """Validate processor parameters."""
        if self.sampling_rate <= 0:
            raise ProcessorValidationError("sampling_rate must be positive")
        if self.duration <= 0:
            raise ProcessorValidationError("duration must be positive")
        if self.channel_schema.get_total_channels() == 0:
            raise ProcessorValidationError("At least one channel must be specified")

    def process(self, context: Optional[ProcessingContext]) -> ProcessingContext:
        """
        Generate synthetic EEG data and return it in a new context.

        Can be used as the **first** step in a pipeline without any prior
        context — pass ``None`` or omit ``initial_context`` in
        :meth:`~facet.core.Pipeline.run`.

        Args:
            context: Existing processing context whose metadata will be
                carried forward, or ``None`` when starting a pipeline from
                scratch.

        Returns:
            New :class:`~facet.core.ProcessingContext` containing the
            generated synthetic EEG recording.
        """
        spike_info = " with epileptic spikes" if self.spike_params.enabled else ""
        logger.info(
            f"Generating synthetic EEG: {self.duration}s @ {self.sampling_rate}Hz, "
            f"{self.channel_schema.get_total_channels()} channels{spike_info}"
        )

        raw, spike_events = generate_synthetic_eeg(
            duration=self.duration,
            sfreq=self.sampling_rate,
            channel_schema=self.channel_schema,
            oscillation_params=self.oscillation_params,
            noise_params=self.noise_params,
            artifact_params=self.artifact_params,
            spike_params=self.spike_params,
            random_seed=self.random_seed
        )

        if context is not None:
            new_ctx = context.with_raw(raw)
        else:
            new_ctx = ProcessingContext(raw)

        new_ctx.metadata.custom['eeg_generator'] = {
            'duration': self.duration,
            'sampling_rate': self.sampling_rate,
            'n_channels': self.channel_schema.get_total_channels(),
            'channel_schema': {
                'eeg': self.channel_schema.eeg_channels,
                'eog': self.channel_schema.eog_channels,
                'ecg': self.channel_schema.ecg_channels,
                'emg': self.channel_schema.emg_channels,
                'misc': self.channel_schema.misc_channels
            },
            'random_seed': self.random_seed,
            'spikes_enabled': self.spike_params.enabled,
            'spike_events': spike_events,
            'n_spikes': len(spike_events)
        }

        logger.info(f"Generated synthetic EEG: {raw.info['nchan']} channels, {raw.n_times} samples")
        if self.spike_params.enabled:
            logger.info(f"Generated {len(spike_events)} epileptic spike events")

        return new_ctx
