import matplotlib.pyplot as plt
import numpy as np

def build_template(raw, spike_sec, half_win_s=0.15):
    sf   = raw.info['sfreq']
    hw   = int(round(half_win_s * sf))
    idxs = (np.array(spike_sec) * sf).astype(int)

    # 1) pick channel with largest total pp
    n_ch = raw.info['nchan']
    pp  = np.zeros(n_ch)
    for ch in range(n_ch):
        for i in idxs:
            if 0 <= i-hw < raw.n_times and i+hw <= raw.n_times:
                pp[ch] += raw._data[ch, i-hw:i+hw].ptp()
    best_ch = pp.argmax()

    # 2) average all ±hw segments
    segs = [raw._data[best_ch, i-hw:i+hw] for i in idxs
            if 0 <= i-hw < raw.n_times and i+hw <= raw.n_times]
    T    = np.stack(segs).mean(axis=0)

    # 3) peak-align so the largest absolute deflection is at center
    half_len = len(T)//2
    peak_idx = np.abs(T).argmax()
    shift    = peak_idx - half_len
    T        = np.roll(T, -shift)

    # 4) z-score
    template_z = (T - T.mean()) / T.std()

    # 5) sanity‐check plot
    times = np.linspace(-half_win_s*1000, half_win_s*1000, len(T))
    plt.figure(figsize=(5,4))
    plt.plot(times, template_z, lw=1.4)
    plt.axvline(0, ls='--', c='C0')
    plt.title("Template (z-scored & peak-aligned)")
    plt.xlabel("Time (ms)"); plt.tight_layout(); plt.show()

    return best_ch, template_z, shift