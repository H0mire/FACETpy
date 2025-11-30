import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

def build_template(raw, spike_sec, half_win_s=0.15, baseline_ms=(-120, -20),
                   smooth=False, return_refined=False):
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
    """
    sf = raw.info['sfreq']
    hw = int(round(half_win_s * sf))
    idxs = (np.asarray(spike_sec) * sf).astype(int)

    # 1) Choose channel with largest cumulative P-P (fixed condition)
    pp = []
    for ch in range(raw.info['nchan']):
        accum = 0.0
        for i in idxs:
            start = i - hw
            stop = i + hw
            if 0 <= start and stop <= raw.n_times:
                seg = raw._data[ch, start:stop]
                accum += seg.ptp()
        pp.append(accum)
    best_ch = int(np.argmax(pp))

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

    T = np.mean(segs, axis=0)

    # Optional smoothing
    if smooth:
        kernel = np.ones(5) / 5.0
        T = np.convolve(T, kernel, mode='same')

    # Z-score
    template_z = (T - T.mean()) / (T.std() + 1e-12)

    times = np.linspace(-half_win_s*1e3, half_win_s*1e3, len(T))
    plt.figure(figsize=(4.5,3))
    plt.plot(times, template_z)
    plt.axvline(0, ls='--', c='k')
    plt.title('IED template')
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()

    shift = 0  # Already centered
    if return_refined:
        return best_ch, template_z, shift, refined_times
    else:
        return best_ch, template_z, shift
