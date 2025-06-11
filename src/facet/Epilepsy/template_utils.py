from mne.annotations import Annotations
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mne.io import Raw
from scipy.signal import find_peaks
from mne.preprocessing import ICA
import mne
from mne.time_frequency import psd_array_welch as psd_welch

def annotate_spike(raw, onset=0.70, duration=0.15, label="spike", plot=True):
    """
    Annotate a spike in the raw EEG data and optionally plot it.

    Parameters:
    - raw: mne.io.Raw
        The EEG recording.
    - onset: float
        Time in seconds where the spike starts.
    - duration: float
        Duration of the spike in seconds.
    - label: str
        Description for the annotation.
    - plot: bool
        Whether to show a plot of the annotated segment.

    Returns:
    - raw: mne.io.Raw
        The same raw object with the annotation added.
    """
    raw.set_annotations(Annotations(onset=[onset],
                                    duration=[duration],
                                    description=[label]))

    if plot:
        raw.plot(
            start=0,
            duration=5,
            scalings=dict(eeg=100),
            title=f'Spike Annotation: {label}',
            show=True
        )

    return raw



def extract_template(raw: Raw, channel: str, start_time: float, end_time: float, normalize: bool = True, plot: bool = True):
    """
    Extract a spike template from the EEG data.

    Parameters:
    - raw: mne.io.Raw object
    - channel: str, EEG channel name to extract from (e.g., "F3")
    - start_time: float, start time in seconds
    - end_time: float, end time in seconds
    - normalize: bool, whether to normalize the template
    - plot: bool, whether to plot the template

    Returns:
    - template (np.ndarray): the extracted (and optionally normalized) waveform
    """
    sfreq = raw.info['sfreq']
    start_sample = int(start_time * sfreq)
    end_sample = int(end_time * sfreq)

    channel_index = raw.ch_names.index(channel)
    template_data = raw.get_data(picks=[channel_index])[0, start_sample:end_sample]

    if normalize:
        template_data = (template_data - np.mean(template_data)) / np.std(template_data)

    if plot:
        times = np.linspace(start_time, end_time, len(template_data))
        plt.plot(times, template_data)
        plt.title(f"Template from {channel} ({start_time:.2f}–{end_time:.2f}s)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (\u00b5V)")
        plt.grid(True)
        plt.show()

    return template_data



def find_spike_template(raw, channel="F3", threshold=100, window=0.15):
    """
    Automatically find a spike in the given EEG channel and extract a template.
    
    Parameters:
        raw: MNE Raw object
        channel: channel name to search in
        threshold: amplitude threshold (in µV)
        window: total window duration in seconds (e.g. 0.15s -> 75ms before/after)

    Returns:
        template_normalized: the extracted spike template (1D np.array)
        template_times: the time axis (1D np.array)
        center_time: center of the spike in seconds
    """
    import numpy as np

    data = raw.get_data(picks=[channel])[0] * 1e6  # Convert V → µV
    times = raw.times
    sfreq = raw.info['sfreq']
    half_window_samples = int((window / 2) * sfreq)

    # Find a spike candidate (just the first one over threshold)
    spike_indices = np.where(np.abs(data) > threshold)[0]
    if len(spike_indices) == 0:
        raise ValueError("No spike candidate found above threshold.")

    center_idx = spike_indices[0]

    # Handle boundary safely
    start_idx = max(center_idx - half_window_samples, 0)
    end_idx = min(center_idx + half_window_samples, len(data))

    spike_segment = data[start_idx:end_idx]
    spike_times = times[start_idx:end_idx]

    # Normalize
    spike_normalized = (spike_segment - np.mean(spike_segment)) / np.std(spike_segment)

    return spike_normalized, spike_times, times[center_idx]



def normalize_signal(sig):
    """Zero-mean, unit-variance normalization."""
    return (sig - np.mean(sig)) / np.std(sig)

def detect_spike_matches(
    raw,
    channel,
    template,
    threshold_ratio=0.8,
    min_distance_sec=0.3,
    threshold_uV=None,
    normalize=True,
    verbose=False,
):
    """
    Detect spike-like events in a signal using normalized cross-correlation with a template.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Cleaned EEG data.
    channel : str
        Name of the EEG channel to search in.
    template : np.ndarray
        1D spike template in µV.
    threshold_ratio : float
        Minimum correlation to consider a match (typically 0.6–0.9).
    min_distance_sec : float
        Minimum time between matches in seconds.
    threshold_uV : float or None
        Minimum peak-to-peak amplitude to consider the segment (optional).
    normalize : bool
        Whether to z-score normalize both template and signal segments.
    verbose : bool
        Print match info and rejection reasons.

    Returns
    -------
    match_times : list of float
        Time points (in seconds) where a spike match was found.
    match_scores : list of float
        Corresponding match scores (normalized correlation).
    """
    signal = raw.get_data(picks=[channel])[0]
 
    sfreq = raw.info['sfreq']
    template_len = len(template)
    min_distance = int(min_distance_sec * sfreq)
    half_len = template_len // 2

    # Normalize template (once)
    template_pos = (template - np.mean(template)) / np.std(template) if normalize else template
    template_neg = -template_pos

    match_times = []
    match_scores = []

    # Slide window across signal
    for i in range(half_len, len(signal) - half_len, min_distance):
        segment = signal[i - half_len : i + half_len]
        if len(segment) != template_len:
            continue  # Skip bad edges

        # Reject low-amplitude segments if threshold is set
        if threshold_uV and np.ptp(segment) < threshold_uV:
            if verbose:
                print(f"Rejected at {i/sfreq:.2f}s due to low amplitude ({np.ptp(segment):.1f} µV)")
            continue

        # Normalize segment
        segment_norm = (segment - np.mean(segment)) / np.std(segment) if normalize else segment

        # Correlate with template and inverted template
        score = max(
            np.dot(segment_norm, template_pos),
            np.dot(segment_norm, template_neg)
        ) / template_len

        if score >= threshold_ratio:
            match_times.append(i / sfreq)
            match_scores.append(score)
            if verbose:
                print(f"Match at {i/sfreq:.2f}s (score = {score:.2f})")

    return match_times, match_scores



def plot_match_with_template(raw, channel, match_times, template, sfreq, window=0.3):
    """
    Plot signal segments around matches with spike template overlayed.
    """
    signal = raw.get_data(picks=[channel])[0]
    half_win = int((window / 2) * sfreq)
    template_len = len(template)

    for match_time in match_times[:5]:
        center = int(match_time * sfreq)
        start = center - half_win
        end = center + half_win

        if start < 0 or end > len(signal):
            continue

        segment = signal[start:end]
        seg_time = np.arange(start, end) / sfreq

        segment_norm = normalize_signal(segment)
        template_norm = normalize_signal(template)

        overlay = np.zeros_like(segment_norm)
        temp_start = (len(segment_norm) - template_len) // 2
        overlay[temp_start:temp_start + template_len] = template_norm

        plt.figure(figsize=(10, 3))
        plt.plot(seg_time, segment_norm, label=f"{channel} segment", alpha=0.7)
        plt.plot(seg_time, overlay, label="Template", linestyle='--')
        plt.title(f"Visual Match at {match_time:.2f}s in {channel}")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()



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
        reject_dict = dict(eeg=reject_uV * 1e-6)   # µV → V this tells ICA to ignore any  1-second segment where the peak-to-peak amplitude exceeds the threshold reject_uV
    ica.fit(raw_ica, reject=reject_dict, verbose=False)

    # --- automatic quick picks: extremely high-frequency components (muscle)
    # muscle_idx, _ = ica.find_bads_muscle(raw_ica, verbose=False)
    # ica.exclude = muscle_idx            # start with these; eye ICs absent (no EOG)

 
    raw_clean = raw.copy()
    ica.apply(raw_clean)                # remove excluded ICs in-place

    return raw_clean, ica




def flag_muscle_ics(ica_obj, raw_obj, hf_band=(30, 90), z_thresh=3.):
    """Return indices of ICs with unusually high 30–90 Hz power."""
    sfreq = raw_obj.info['sfreq']
    sources = ica_obj.get_sources(raw_obj).get_data()
    psds, freqs = psd_welch(
        sources,
        sfreq=sfreq,                
        fmin=hf_band[0],
        fmax=hf_band[1],
        n_fft=int(sfreq * 2),
        n_overlap=0,
        verbose=False
    )          
    hf_power = psds.mean(axis=1)
    zscores   = (hf_power - hf_power.mean()) / hf_power.std()
    return list(np.where(zscores > z_thresh)[0])



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
    ptp = raw.get_data(reject_by_annotation='omit').ptp(axis=1) * 1e6  # µV
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
        Rescaled EEG.
    """
    target = target.lower()
    if not hasattr(raw, "_orig_units"):
        print("Warning: Unit metadata not found. No scaling applied.")
        return raw

    # Normalize unit names
    orig = raw._orig_units.get("EEG", "").lower()

    if orig == target:
        print(f"EEG already in {target}. No scaling applied.")
        return raw

    if orig == "v" and target == "uv":
        print("Scaling EEG from volts → microvolts.")
        raw._data *= 1e6
    elif orig == "uv" and target == "v":
        print("Scaling EEG from microvolts → volts.")
        raw._data *= 1e-6
    else:
        print(f"Unknown or mismatched units: {orig} → {target}. No scaling.")
    return raw


def extract_median_spike(sig, sfreq, height_mult=4, min_dist=0.2, win_len=0.15):
    """Find spike-like peaks in a 1D IC signal and return a normalized median template."""
    

    peaks, _ = find_peaks(np.abs(sig), height=height_mult * sig.std(), distance=int(min_dist * sfreq))
    win = int(win_len * sfreq)
    snips = [sig[p - win//2 : p + win//2] for p in peaks if p - win//2 >= 0 and p + win//2 <= len(sig)]
    template = np.median(snips, axis=0)
    return (template - template.mean()) / template.std()

def backproject_ic_to_sensor(template_ic, ic_idx, raw, ica, sensor):
    """Project 1D IC template back to EEG space and return signal from one channel."""
    topo = ica.mixing_matrix_[:, ic_idx]
    projected = np.outer(topo, template_ic)  # shape: (n_channels, n_time)
    return projected[raw.ch_names.index(sensor)]

def plot_spike_template(template, title="Spike Template", fs=150):
    """Plot a 1D EEG template with time axis in seconds."""
    t = np.linspace(-0.075, 0.075, len(template))
    plt.plot(t, template)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    plt.show()

def find_spike_matches_all_channels(raw, template, threshold_ratio=0.8, verbose=False):
    """
    Detect spike matches across all EEG channels and find the best individual match.

    Parameters
    ----------
    raw : mne.io.Raw
        Scaled EEG data.
    template : np.ndarray
        1D template (in microvolts).
    threshold_ratio : float
        Correlation threshold for accepting matches.
    verbose : bool
        If True, print match info.

    Returns
    -------
    channel_scores : dict
        Channel → number of matches.
    top_channel : str
        Channel with best individual match.
    top_time : float
        Time of best match (in seconds).
    top_score : float
        Best individual match score.
    """
    from collections import defaultdict
    import numpy as np

    channel_scores = defaultdict(int)
    top_score = -np.inf
    top_channel = None
    top_time = None

    for ch in raw.info['ch_names']:
        match_times, match_scores = detect_spike_matches(
            raw, ch, template, threshold_ratio=threshold_ratio, verbose=verbose
        )
        channel_scores[ch] = len(match_times)

        if match_scores:
            max_idx = np.argmax(match_scores)
            if match_scores[max_idx] > top_score:
                top_score = match_scores[max_idx]
                top_channel = ch
                top_time = match_times[max_idx]

    return channel_scores, top_channel, top_time, top_score


def plot_top_matches_from_results(raw, channel, match_times, match_scores, template, top_n=5):
    """
    Plot the top N spike matches from already detected results.

    Parameters
    ----------
    raw : mne.io.Raw
        Cleaned EEG data.
    channel : str
        Channel name.
    match_times : list of float
        Detected spike match times (in seconds).
    match_scores : list of float
        Corresponding match scores.
    template : np.ndarray
        Template waveform in µV.
    top_n : int
        Number of top matches to plot.
    """
    if not match_scores:
        print(f"No matches to plot for channel {channel}.")
        return

    top_matches = sorted(zip(match_times, match_scores), key=lambda x: x[1], reverse=True)[:top_n]

    print(f"\nTop {len(top_matches)} matches in {channel}:")
    for t, score in top_matches:
        print(f"{channel} at {t:.2f}s (score = {score:.2f})")
        plot_match_with_template(raw, channel, [t], template, raw.info["sfreq"])
