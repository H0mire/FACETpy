from mne.annotations import Annotations
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mne.io import Raw
from scipy.signal import find_peaks
from mne.preprocessing import ICA
import mne
from mne.time_frequency import psd_array_welch as psd_welch
from scipy.signal import correlate


def extract_spike_windows(
        raw,
        spike_times_s,
        channel,
        half_win_s=0.15,
        baseline_ms=0,
):
    """Extract ±half_win_s-second snippets around each spike time for one channel.

    Parameters
    ----------
    raw : mne.io.Raw
        Your MNE Raw object (in µV already).
    spike_times_s : sequence of float
        Spike times in seconds.
    channel : int
        Index of the channel to extract.
    half_win_s : float
        Half‐window length in seconds (so total window is 2*half_win_s).
    baseline_ms : float
        Duration in ms before spike to use for baseline subtraction (default 0 = no subtraction).

    Returns
    -------
    windows : ndarray, shape (n_spikes, n_samples)
        Each row is the raw data segment around one spike.
    times : ndarray, shape (n_samples,)
        Time axis in seconds, from -half_win_s → +half_win_s.
    """
    sfreq    = raw.info['sfreq']
    half_samp = int(round(half_win_s * sfreq))
    data      = raw.get_data(picks=[channel])[0]

    segments = []
    for t in spike_times_s:
        idx   = int(round(t * sfreq))
        start = idx - half_samp
        stop  = idx + half_samp
        if 0 <= start and stop <= data.size:
            segments.append(data[start:stop])

    segments = np.stack(segments)
    times = np.linspace(-half_win_s, half_win_s, segments.shape[1])

    # Baseline subtraction
    if baseline_ms > 0:
        baseline_samples = int(round(baseline_ms / 1000 * sfreq))
        for i in range(segments.shape[0]):
            baseline_mean = np.mean(segments[i, :baseline_samples])
            segments[i] -= baseline_mean

    #  plot first snippet
    plt.plot(times * 1000, segments[0], lw=1)
    plt.axvline(0, ls="--", color="k")
    plt.xlabel("Time (ms)")
    plt.ylabel("µV")
    plt.title("IED‐centred window")
    plt.show()
    return segments, times


def basic_clean(
        raw,
        *,
        l_freq      = 1. ,
        h_freq      = 40.,
        notch       = 50 ,
        crop_tmin   = None,
        crop_tmax   = None,
        ica_var     = 0.99,      # keep components that explain 99 % variance
        reject_uV   = None,       # artefact rejection when fitting ICA
        random_state= 42
):
    """
    Minimal EEG clean-up helper (band-pass + notch + ICA).

    Parameters
    ----------
    raw : mne.io.Raw
        EEG already loaded (in µV!).
    l_freq , h_freq : float
        Band-pass corner frequencies (Hz).
    notch : float | sequence
        Notch frequency / frequencies (e.g. 50 or [50,100]).
    crop_tmin , crop_tmax : float | None
        Optional time window (sec) to use for ICA (skip bad edges).
    ica_var : float | int
        If < 1 → fraction of variance to keep, else exact number of ICs.
    reject_uV : float | None
        If set, segments with peak-to-peak > `reject_uV` are ignored
        while fitting ICA.
    random_state : int
        Reproducibility for FastICA.
    
    Returns
    -------
    raw_clean : mne.io.Raw      (copy with artefacts removed)
    ica       : mne.preprocessing.ICA
    """
    # ------------------------------------------------------------------
    raw = raw.copy()                    # do NOT scale again (already µV)
    raw.set_eeg_reference('average')    # average-reference → better ICA - Remove bias from a specific reference electrode
    # ------------------------------------------------------------------
    # Standard filtering
    raw.notch_filter(notch, verbose=False) #notch filter (e.g. 50 Hz) removes 50Hz power line noise
    raw.filter(l_freq, h_freq, fir_design='firwin', verbose=False) #bandpass filter keeps frequencies between 1-40Hz
    # ------------------------------------------------------------------
    # Optional cropping for ICA (reduces runtime, excludes noisy start/end)
    raw_ica = raw.copy()
    if crop_tmin is not None or crop_tmax is not None:
        raw_ica.crop(tmin=crop_tmin, tmax=crop_tmax)
    # ------------------------------------------------------------------
    # Fit ICA
    ica = ICA(n_components=ica_var, method='fastica', random_state=random_state)
    reject_dict = None
    if reject_uV is not None:
        reject_dict = dict(eeg=reject_uV )   # µV → V this tells ICA to ignore any  1-second segment where the peak-to-peak amplitude exceeds the threshold reject_uV
    ica.fit(raw_ica, reject=reject_dict, verbose=False)

    # --- automatic quick picks: extremely high-frequency components (muscle)
    # muscle_idx, _ = ica.find_bads_muscle(raw_ica, verbose=False)
    # ica.exclude = muscle_idx            # start with these; eye ICs absent (no EOG)

 
    raw_clean = raw.copy()
    ica.apply(raw_clean)                # remove excluded ICs in-place

    return raw_clean, ica



def compute_adaptive_threshold(raw, percentile=99, scale=1.5):
    """
    Compute adaptive peak-to-peak rejection threshold (µV).

    Parameters
    ----------
    raw : mne.io.Raw
        The EEG data (should be in µV already).
    percentile : float
        Percentile of the peak-to-peak distribution to base the threshold on.
    scale : float
        Multiplier to increase margin above the selected percentile.

    Returns
    -------
    float
        Adaptive threshold in µV.
    """
    ptp = raw.get_data(reject_by_annotation='omit').ptp(axis=1)  # µV
    return np.percentile(ptp, percentile) * scale



def scale_eeg_to(raw, target="uV"):  
    """
    Ensure EEG data is scaled to the desired unit ('V' or 'uV').

    Parameters
    ----------
    raw : mne.io.Raw
        EEG object to check and scale.
    target : str
        Desired unit ('V' or 'uV').

    Returns
    -------
    raw : mne.io.Raw
        Rescaled EEG with updated unit metadata.
    """
    target = target.lower()

    # Ensure metadata exists, defaulting to volts
    if not hasattr(raw, '_orig_units') or 'EEG' not in raw._orig_units:
        raw._orig_units = {'EEG': 'V'}

    orig = raw._orig_units.get('EEG', 'V').lower()
    if orig not in ('v', 'uv'):
        # fallback assume volts
        orig = 'v'
        raw._orig_units['EEG'] = 'V'

    # No conversion needed
    if orig == target:
        print(f"EEG already in {target}. No scaling applied.")
        return raw

    # Perform conversion
    if orig == 'v' and target == 'uv':
        raw._data *= 1e6
        raw._orig_units['EEG'] = 'uV'
        print("Scaling EEG from volts → microvolts.")
    elif orig == 'uv' and target == 'v':
        raw._data *= 1e-6
        raw._orig_units['EEG'] = 'V'
        print("Scaling EEG from microvolts → volts.")
    else:
        print(f"Unknown or mismatched units: {orig} → {target}. No scaling applied.")

    return raw



def plot_spike_template(template, title="Spike Template", fs=150):
    """Plot a 1D EEG template with time axis in seconds."""
    t = np.linspace(-0.075, 0.075, len(template))
    plt.plot(t, template)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    plt.show()




