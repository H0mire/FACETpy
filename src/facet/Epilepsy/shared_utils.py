import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

def build_template(raw, spike_sec, half_win_s=0.15, return_refined=False):
    """
    Builds a peak-aligned, polarity-standardised template.
    Any annotation lag is removed automatically by re-centering each
    snippet on its own largest absolute deflection.
    """
    sf   = raw.info['sfreq']
    hw   = int(round(half_win_s * sf))
    idxs = (np.asarray(spike_sec) * sf).astype(int)

    # 1)   choose channel with largest cumulative P-P
    pp = [sum(raw._data[ch, i-hw:i+hw].ptp()
              for i in idxs if 0 <= i-hw < raw.n_times <= i+hw)
          for ch in range(raw.info['nchan'])]
    best_ch = int(np.argmax(pp))

    segs, refined_times = [], []
    for i in idxs:
        if 0 <= i-hw and i+hw <= raw.n_times:
            seg = raw._data[best_ch, i-hw:i+hw].astype(float, copy=True)
            seg = detrend(seg, type='linear')            # baseline
            # flip so main spike is positive
            if np.abs(seg.min()) > np.abs(seg.max()):
                seg *= -1

            # ---- NEW: roll so largest |deflection| is at centre ----
            peak_idx = np.abs(seg).argmax()
            centre   = len(seg)//2
            shift    = centre - peak_idx
            seg      = np.roll(seg, shift)
            refined_times.append((i + shift) / sf)       # optional

            segs.append(seg)

    T = np.mean(segs, axis=0)

    # z-score
    template_z = (T - T.mean()) / T.std()

    # quick look
    times = np.linspace(-half_win_s*1e3, half_win_s*1e3, len(T))
    plt.figure(figsize=(4.5,3))
    plt.plot(times, template_z); plt.axvline(0, ls='--', c='k')
    plt.title('IED template'); plt.xlabel('Time (ms)'); plt.tight_layout(); plt.show()

    shift = 0                       # already centred
    if return_refined:
        return best_ch, template_z, shift, refined_times
    else:
        return best_ch, template_z, shift
