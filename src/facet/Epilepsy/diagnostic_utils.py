import matplotlib.pyplot as plt
import numpy as np

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


