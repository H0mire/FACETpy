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
from mne.filter import filter_data
from scipy.stats import kurtosis

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
                  spike_sec, half_win_s=0.10):
    """
    Ebrahimzade 2021 version:
        • fit ICA  (1–40 Hz copy)
        • keep first 3 ICs with kurtosis ≥ 3
        • sum them, band-pass 3–25 Hz
        • correlate (|r|) with the template
    Returns
    -------
    keep_idx : list[int]      indices of ICs used
    comp_bp  : 1-D ndarray    composite IC trace (µV, 3–25 Hz)
    r_abs    : 1-D ndarray    |r|(t) aligned with raw
    """
    sf   = raw.info['sfreq']
    hw   = int(round(half_win_s * sf))

    # 1) ICA on 1-40 Hz copy
    ica = mne.preprocessing.ICA(
            n_components=min(20, raw.info['nchan']),
            random_state=97, method='fastica', max_iter='auto')
    ica.fit(raw.copy().filter(1., 40., picks='eeg'))
    S = ica.get_sources(raw).get_data()          # shape (n_comp, n_times)

    # 2) keep first 3 peaky ICs
    keep_idx = np.where(kurtosis(S, axis=1, fisher=False) >= 3)[0]
    if keep_idx.size == 0:                       # fallback: most-kurtotic IC
        keep_idx = np.array([np.argmax(kurtosis(S, axis=1, fisher=False))])

    # 3) composite trace  → 3–25 Hz
    comp      = S[keep_idx].sum(axis=0)
    comp_bp   = filter_data(comp, sf, 3., 25., verbose=False)

    # 4) correlation |r|
    r = np.roll(pearson_r(comp_bp, template_z), -half_shift)
    r_abs = np.abs(r)                            # 0 … 1

    return keep_idx.tolist(), comp_bp.ravel(), r_abs.ravel()


def compute_spike_regressors(raw, spike_sec,
                             half_win_s=0.15,
                             th_raw=0.35,
                             match_tol_s=0.1):
    """
    Ebrahimzade 2021 implementation:
      • build spike template on best scalp channel
      • corr(template, best channel) → r_raw   (aux plot)
      • find_best_ica() returns:
            keep_idx   : list of IC indices used
            ica_trace  : composite IC (band-passed 3–25 Hz)
            r_ica      : |r|(t) on that trace, 0…1
      • adaptive gate  th_ica = 0.75 * r_ica.max()
      • peaks: find local maxima ≥ th_ica, refractory 0.2 s
      • match detections to annotations within ±match_tol_s
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    sfreq = raw.info['sfreq']

    # 1  template
    best_ch, template_z, half_shift, spike_sec_refined = \
    build_template(raw, spike_sec, half_win_s=0.15, return_refined=True)

    # 2  raw-channel correlation (auxiliary)
    sig   = raw.get_data(picks=[best_ch])[0]
    sig_z = (sig - sig.mean()) / sig.std()
    r_raw = pearson_r(sig_z, template_z)        # centre already aligned

    # 3  ICA composite + correlation (|r|)
    keep_idx, ica_trace, r_ica = find_best_ica(
        raw, template_z, half_shift, spike_sec_refined, half_win_s)

    # 4  adaptive threshold & peak picking 
    th_ica   = 0.75 * r_ica.max()
    min_dist = int(0.15 * sfreq)
    peaks_raw, _ = find_peaks(r_raw, height=th_raw, distance=min_dist)
    peaks_ica, _ = find_peaks(r_ica, height=th_ica, distance=min_dist)

    refractory = int(0.15 * sfreq)           # 150 ms
    merged = []
    for p in peaks_ica:
        if not merged or p - merged[-1] > refractory:
            merged.append(p)
    peaks_ica = np.array(merged, int)
    # 5  match detections to manual spikes
    tol_samp = int(match_tol_s * sfreq)
    caught   = []
    missed   = []
    for t0 in spike_sec_refined:
        i0 = int(round(t0 * sfreq))
        if np.any(np.abs(peaks_ica - i0) <= tol_samp):
            caught.append(t0)
        else:
            missed.append(t0)

    print(f"r_ica max = {r_ica.max():.3f}   th_ica = {th_ica:.3f}")
    print(f"Caught {len(caught)}/{len(spike_sec_refined)} spikes within ±{match_tol_s}s")


    return (template_z, r_raw, r_ica,
            peaks_raw, peaks_ica,
            best_ch, keep_idx,
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




