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
from facet.Epilepsy.shared_utils import build_template

def pearson_r(signal, template_z):
    """
    Sample-wise Pearson correlation between 1-D `signal`
    and zero-mean/unit-std template `template_z` (length L).
    Returns r(t) in [-1, +1], same length as signal.
    """
    L = len(template_z)

    # Numerator = cross-corr
    num = correlate(signal, template_z[::-1], mode='same')

    # Local mean & std via convolution
    kernel = np.ones(L) / L
    mean   = np.convolve(signal,    kernel, mode='same')
    mean2  = np.convolve(signal**2, kernel, mode='same')
    std    = np.sqrt(np.maximum(mean2 - mean**2, 1e-12))

    r = num / (L * std)            # template std = 1
    r[~np.isfinite(r)] = 0
    return r

def find_best_ica(raw, template_z, half_shift,
                  n_components=25, random_state=97):
    """
    Compute ICA, then pick the component whose Pearson r(t) 
    has the highest peak |r| anywhere in the recording.

    Returns
    -------
    best_idx   : int       index of the chosen ICA component
    best_trace : ndarray   that component's time series (µV)
    best_r     : ndarray   its Pearson r(t) with the template
    """

    # 1) fit ICA on a 1+ Hz high-passed copy
    ica = ICA(n_components=n_components,
              random_state=random_state,
              max_iter="auto")
    ica.fit(raw.copy().filter(1., None))

    # 2) grab all component time-series
    sources = ica.get_sources(raw).get_data()  # (n_comp, n_times)

    best_idx, best_peak = None, -np.inf
    best_r,   best_trace = None, None
    
    
    # 3) slide-correlate each component against the template
    for comp_idx, comp in enumerate(sources):
        r = pearson_r(comp, template_z)
        r = np.roll(r, -half_shift)
        peak = np.max(np.abs(r))
        if peak > best_peak:
            best_peak, best_idx = peak, comp_idx
            best_r, best_trace = r, comp

    return best_idx, best_trace, best_r

        
def compute_spike_regressors(raw, spike_sec,
                             half_win_s=0.10,
                             th_raw=0.30, th_ica=0.85,
                             min_dist_s=0.3,
                             match_tol_s=0.2):
    """
    Build template, compute r_raw & r_ica, pick discrete spike events,
    and then match them back to your annotated times with a tolerance.

    Returns
    -------
    template_z   : ndarray   z-scored spike template
    r_raw        : ndarray   continuous r(t) on best scalp channel
    r_ica        : ndarray   continuous r(t) on best ICA component
    peaks_raw    : ndarray   sample-indices of raw-channel spikes
    peaks_ica    : ndarray   sample-indices of ICA-component spikes
    best_ch      : int       scalp channel index used for template
    best_ica_idx : int       ICA component index
    caught       : list      annotated times that were detected
    missed       : list      annotated times that were *not* detected
    """
    
  
    sfreq = raw.info['sfreq']
    # 1) build template…
    best_ch, template_z, half_shift = build_template(raw, spike_sec, half_win_s)

    # 2) raw‐channel correlation…
    sig   = raw.get_data(picks=[best_ch])[0]
    sig_z = (sig - sig.mean())/sig.std()
    r_raw = pearson_r(sig_z, template_z)
    r_raw = np.roll(r_raw, -half_shift)

    # 3) ICA correlation…
    best_ica_idx, ica_trace, r_ica = find_best_ica(raw, template_z, half_shift)

    # 4) pick peaks
    dist_s       = int(min_dist_s * sfreq)
    peaks_raw, _ = find_peaks(r_raw, height=th_raw, distance=dist_s)
    peaks_ica, _ = find_peaks(r_ica, height=th_ica, distance=dist_s)

    # 5) match back into caught / missed
    tol_s    = match_tol_s
    tol_samp = int(tol_s * sfreq)
    caught, missed = [], []
    for t0 in spike_sec:
        idx0 = int(round(t0 * sfreq))
        if np.any(np.abs(peaks_ica - idx0) <= tol_samp):
            caught.append(t0)
        else:
            missed.append(t0)

    print(f"Caught  {len(caught)}/{len(spike_sec)} spikes within ±{tol_s}s:")
    print("  → caught:", caught)
    print("  → missed:", missed)

    # 6) final summary plot
    times = np.arange(raw.n_times)/sfreq
    plt.figure(figsize=(10,3))
    plt.plot(times, ica_trace, label=f"ICA {best_ica_idx} (µV)")
    plt.plot(times, r_ica*20, label="r(t)×20")
    for s in spike_sec:
        plt.axvline(s, ls='--', c='C3', alpha=0.7)
    plt.scatter(peaks_ica/sfreq,
                np.full_like(peaks_ica, th_ica*20),
                marker='v', c='C1', label='detected')
    plt.legend(loc='upper right')
    plt.xlabel("Time (s)")
    plt.title("ICA vs annotations & detections")
    plt.tight_layout()
    plt.show()

    return (template_z, r_raw, r_ica,
            peaks_raw, peaks_ica,
            best_ch, best_ica_idx,
            caught, missed)


def plot_ica_with_annotations(raw, ica_trace, peaks_ica, spike_sec,
                              match_tol_s=0.1, tmin=0, tmax=None ):
    """
    raw         : mne.io.Raw    (just to get sfreq and times)
    ica_trace   : 1D array (n_times,) your chosen ICA component (µV)
    peaks_ica   : 1D int array    sample indices where you ran find_peaks()
    spike_sec   : list of floats  manual annotation times in seconds
    match_tol_s : float           tolerance window for a “hit”
    tmin,tmax   : floats          time range to plot (in seconds)
    """
    sf = raw.info['sfreq']
    times = np.arange(ica_trace.size) / sf
    if tmax is None:
        tmax = times[-1]
    # convert peaks to times
    peaks_t = peaks_ica / sf

    # full overview
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(times, ica_trace, color='C0', lw=0.6, label='ICA trace (µV)')
    ax.scatter(peaks_t, np.interp(peaks_t, times, ica_trace),
               marker='v', color='C1', label='Detected peaks')
    ax.vlines(spike_sec, ica_trace.min(), ica_trace.max(),
              color='C3', linestyle='--', alpha=0.7, label='Annotations')
    ax.set_xlim(tmin, tmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('ICA component vs annotated spikes & detected events')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # now zoom in on the first few annotations
    for s in spike_sec[:5]:
        window = match_tol_s * 5  # e.g. 5× your tolerance
        sel = (times >= s-window) & (times <= s+window)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(times[sel], ica_trace[sel], color='C0', lw=1)
        # detected peaks in window
        sel_peaks = peaks_t[(peaks_t >= s-window) & (peaks_t <= s+window)]
        ax.scatter(sel_peaks,
                   np.interp(sel_peaks, times, ica_trace),
                   marker='v', color='C1')
        ax.axvline(s, color='C3', linestyle='--', label='Annotation')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('µV')
        ax.set_title(f'Zoom around manual spike @ {s:.2f}s')
        ax.legend()
        plt.tight_layout()
        plt.show()




def quick_spike_overview(raw, spike_sec, highlight_ch, window_s=0.4, n_spikes=3):
    """
    Plot all channels ±window_s/2 around the first n_spikes annotated events,
    highlighting a single channel in red with its name in the legend.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Your EEG data (µV).
    spike_sec : list of float
        Annotated spike times in seconds.
    highlight_ch : int
        Index of the channel to highlight/label.
    window_s : float
        Length of window in seconds (default 0.4 s).
    n_spikes : int
        How many spikes (rows) to plot.
    """
    sf = raw.info['sfreq']
    half = int(window_s * sf / 2)
    data = raw.get_data()  # shape (n_ch, n_times)
    ch_names = raw.ch_names
    
    t_axis = np.arange(-half, half) / sf * 1e3  # milliseconds

    fig, axes = plt.subplots(n_spikes, 1, figsize=(8, 2.5*n_spikes), sharex=True)
    if n_spikes == 1:
        axes = [axes]

    for ax, sec in zip(axes, spike_sec[:n_spikes]):
        idx = int(round(sec * sf))
        start, stop = idx-half, idx+half
        if start < 0 or stop > data.shape[1]:
            ax.text(0.5, 0.5, "Spike too close to edge", ha='center')
            continue

        # Plot all channels in light grey
        for ch in range(data.shape[0]):
            ax.plot(t_axis, data[ch, start:stop],
                    color='grey', alpha=0.4, linewidth=0.5)

        # Overplot the highlighted channel in red
        ax.plot(t_axis, data[highlight_ch, start:stop],
                color='red', linewidth=1.5,
                label=f"{ch_names[highlight_ch]}")

        ax.axvline(0, color='k', ls='--')
        ax.set_ylabel(f"Spike @ {sec:.2f}s")
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    plt.show()
    plt.close("all")


def list_top_spike_channels(raw, spike_sec, half_win_s=0.10, top_n=5):
    """
    Identify the top N EEG channels by average peak-to-peak amplitude around annotated spikes.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG data, assumed in microvolts (µV).
    spike_sec : list of float
        Spike annotation times in seconds.
    half_win_s : float
        Half-window length in seconds for P–P calculation (default ±0.10 s).
    top_n : int
        Number of top channels to return (default 5).

    Returns
    -------
    top_channels : list of tuples
        List of (index, name, mean_pp) for the top N channels, sorted descending.
    """
    sfreq = raw.info['sfreq']
    half_samples = int(round(half_win_s * sfreq))
    idxs = (np.array(spike_sec) * sfreq).astype(int)

    n_ch = raw.info['nchan']
    data = raw.get_data()  # shape (n_ch, n_times), µV

    mean_pp = np.zeros(n_ch)
    for ch in range(n_ch):
        pps = []
        for idx in idxs:
            start, stop = idx - half_samples, idx + half_samples
            if 0 <= start and stop <= data.shape[1]:
                segment = data[ch, start:stop]
                pps.append(np.ptp(segment))
        mean_pp[ch] = np.mean(pps) if pps else 0

    order = np.argsort(mean_pp)[::-1]
    top = [(int(ch), raw.ch_names[ch], float(mean_pp[ch])) for ch in order[:top_n]]
    return top


def zoom_detected_events(raw, trace, r, peaks, half_win_s=0.4, th_ica=0.85):
    """
    Loop over detected peaks and plot each event in a zoomed-in window.

    Parameters
    ----------
    raw : mne.io.Raw
        EEG object (used for sampling frequency).
    trace : ndarray, shape (n_times,)
        The time series (e.g., ICA component) in µV.
    r : ndarray, shape (n_times,)
        Continuous correlation trace r(t).
    peaks : array-like of int
        Sample indices of detected events.
    half_win_s : float
        Total window length in seconds (default 0.4s).
    th_ica : float
        Threshold used for detection (for plotting reference lines).
    """
    import matplotlib.pyplot as plt
    sfreq = raw.info['sfreq']
    t = np.arange(trace.size) / sfreq
    half_samples = int(round(half_win_s/2 * sfreq))
    
    for idx in peaks:
        center = idx / sfreq
        start = idx - half_samples
        stop = idx + half_samples
        if start < 0 or stop > trace.size:
            continue
        sel = slice(start, stop)
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(t[sel], trace[sel], label='ICA trace (µV)')
        ax.plot(t[sel], r[sel]*20, label='r(t) × 20', lw=1)
        ax.axvline(center, color='red', linestyle=':')
        ax.axhline(th_ica*20, ls='--', color='gray')
        ax.axhline(-th_ica*20, ls='--', color='gray')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Spike at {center:.2f}s')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()



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


def normalize_signal(sig):
    """Zero-mean, unit-variance normalization."""
    return (sig - np.mean(sig)) / np.std(sig)



def backproject_ic_to_sensor(template_ic, ic_idx, raw, ica, sensor):
    """Project 1D IC template back to EEG space and return signal from one channel."""
    topo = ica.mixing_matrix_[:, ic_idx]
    projected = np.outer(topo, template_ic)  # shape: (n_channels, n_time)
    return projected[raw.ch_names.index(sensor)]



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



