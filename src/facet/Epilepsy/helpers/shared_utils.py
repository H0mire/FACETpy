import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.signal import detrend

def build_template(raw, spike_sec, half_win_s=0.15, baseline_ms=(-120, -20),
                   smooth=False, return_refined=False, visualize=True,
                   max_spikes=None, return_segments=False):
    """
    Builds a peak-aligned, polarity-standardised template.
    Any annotation lag is removed automatically by re-centering each
    snippet on its own largest absolute deflection.

    Parameters
    ----------
    raw : mne.io.Raw
        The EEG raw data.
    spike_sec : list of float
        Spike times in seconds.
    half_win_s : float
        Half-window length in seconds.
    baseline_ms : tuple of float
        Baseline window in ms (start, end) for mean subtraction.
    smooth : bool
        Whether to apply light smoothing (5-point moving average).
    return_refined : bool
        Whether to return refined spike times.
    visualize : bool
        Whether to plot the template.
    max_spikes : int | None
        If set and more than ``max_spikes`` valid segments are available,
        only the ``max_spikes`` segments most correlated with a
        leave-one-out grand-average (i.e. each segment is scored against
        the average of every OTHER segment, never against itself) are kept
        for the final template. This automates the "hand-select the best
        10-20 spikes" step some IED-template protocols specify, using
        template correlation as the objective proxy for waveform quality
        instead of visual review. ``None`` (default) keeps every valid
        segment, matching prior behaviour.
    return_segments : bool
        If True, also return the aligned/polarity-standardised/baseline-
        corrected individual epochs (in physical signal units) and their
        pre-z-score mean ``T``. Opt-in and off by default; it exposes the
        intermediate template stages for diagnostics/plotting without
        changing any pipeline behaviour.

    Returns
    -------
    best_ch : int
        Index of the best channel.
    template_z : ndarray
        Z-scored template.
    shift : int
        Shift applied (0 since already centered).
    refined_times : list, optional
        Refined spike times if return_refined=True.
    segments : ndarray, optional
        Aligned per-spike epochs (n_kept, n_samples), physical units, if
        return_segments=True.
    template_phys : ndarray, optional
        Pre-z-score average template (physical units) if
        return_segments=True.
    """
    sf = raw.info['sfreq']
    hw = int(round(half_win_s * sf))
    idxs = (np.asarray(spike_sec) * sf).astype(int)

    # 1) Choose channel with largest cumulative P-P (fixed condition).
    #    Restrict to EEG channels only — otherwise high-amplitude non-brain
    #    channels (ECG/EMG/ear) win on peak-to-peak and the template would be
    #    built from a cardiac/muscle artefact instead of the IED.
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads')
    if len(eeg_picks) == 0:
        eeg_picks = np.arange(raw.info['nchan'])  # fallback: no typed EEG
    pp = []
    for ch in eeg_picks:
        accum = 0.0
        for i in idxs:
            start = i - hw
            stop = i + hw
            if 0 <= start and stop <= raw.n_times:
                seg = raw._data[ch, start:stop]
                accum += seg.ptp()
        pp.append(accum)
    best_ch = int(eeg_picks[int(np.argmax(pp))])

    segs, refined_times = [], []
    for i in idxs:
        start = i - hw
        stop = i + hw
        if 0 <= start and stop <= raw.n_times:
            seg = raw._data[best_ch, start:stop].astype(float, copy=True)

            # Fixed baseline subtraction (pre-spike window mean)
            times_ms = np.linspace(-half_win_s*1e3, half_win_s*1e3, seg.size)
            bmask = (times_ms >= baseline_ms[0]) & (times_ms <= baseline_ms[1])
            if bmask.any():
                seg -= seg[bmask].mean()

            # Flip so main spike is positive
            if np.abs(seg.min()) > np.abs(seg.max()):
                seg *= -1

            # Roll so largest |deflection| is at center
            peak_idx = np.abs(seg).argmax()
            centre = len(seg) // 2
            shift = centre - peak_idx
            seg = np.roll(seg, shift)
            refined_times.append((i + shift) / sf)

            segs.append(seg)

    # Assert enough segments
    if len(segs) < 5:
        raise ValueError("Not enough spike segments to build a stable template.")

    # Optional quality selection: keep only the ``max_spikes`` segments most
    # correlated with a LEAVE-ONE-OUT grand-average, discarding the rest
    # before the final average is computed. Segments/refined_times stay in
    # sync.
    #
    # Each segment is scored against the average of every OTHER segment
    # (never against itself). Scoring against the average of ALL segments
    # (including itself) is circular: a segment always pulls its own
    # reference toward itself, so noisy-but-similar segments can inflate
    # their own score and get kept. Excluding the segment being scored
    # removes that self-inclusion bias.
    if max_spikes is not None and len(segs) > max_spikes:
        segs_arr = np.stack(segs)  # (n_spikes, n_samples)
        n = segs_arr.shape[0]
        total = segs_arr.sum(axis=0)
        scores = np.empty(n)
        for i in range(n):
            loo_mean = (total - segs_arr[i]) / (n - 1)
            seg = segs_arr[i]
            if np.std(seg) > 0 and np.std(loo_mean) > 0:
                scores[i] = np.corrcoef(seg, loo_mean)[0, 1]
            else:
                scores[i] = -np.inf
        keep = np.argsort(scores)[::-1][:max_spikes]
        keep = np.sort(keep)  # preserve chronological order
        segs = [segs[i] for i in keep]
        refined_times = [refined_times[i] for i in keep]

    T = np.mean(segs, axis=0)

    # Optional smoothing
    if smooth:
        kernel = np.ones(5) / 5.0
        T = np.convolve(T, kernel, mode='same')

    # Z-score
    template_z = (T - T.mean()) / (T.std() + 1e-12)

    if visualize:
        times = np.linspace(-half_win_s*1e3, half_win_s*1e3, len(T))
        plt.figure(figsize=(4.5,3))
        plt.plot(times, template_z)
        plt.axvline(0, ls='--', c='k')
        plt.title('IED template')
        plt.xlabel('Time (ms)')
        plt.tight_layout()
        plt.show()

    shift = 0  # Already centered
    out = [best_ch, template_z, shift]
    if return_refined:
        out.append(refined_times)
    if return_segments:
        out.extend([np.stack(segs), T])
    return tuple(out)
