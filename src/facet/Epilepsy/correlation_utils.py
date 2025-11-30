import matplotlib.pyplot as plt
import numpy as np
from mne.io import Raw
from scipy.signal import find_peaks
from mne.preprocessing import ICA
import mne
from scipy.signal import correlate
from facet.Epilepsy.shared_utils import build_template
from mne.filter import filter_data
from scipy.stats import kurtosis
from facet.Epilepsy.Models.pipeline_results import TemplateICADetection
from facet.Epilepsy.regressors import generate_hrf_regressors

# ======================= ICA composite (Ebrahimzadeh) =======================
def build_ica_composite(raw, template_z, band_ica=(1., 40.), band_comp=(3., 25.),
                        kurtosis_min=3.0, max_keep=3, random_state=97):
    """Fit ICA, select peaky components, sum, band-pass, correlate with template."""
    sf = raw.info['sfreq']
    ica = ICA(n_components=min(20, raw.info['nchan']),
              random_state=random_state, method='fastica', max_iter='auto')
    ica.fit(raw.copy().filter(*band_ica, picks='eeg'))
    S = ica.get_sources(raw).get_data()  # (n_comp, n_times)
    k = kurtosis(S, axis=1, fisher=False)
    keep = np.where(k >= kurtosis_min)[0]
    if keep.size == 0:
        keep = np.array([int(np.argmax(k))])
    keep = keep[:max_keep]
    comp = S[keep].sum(axis=0)
    comp_bp = filter_data(comp, sf, band_comp[0], band_comp[1], verbose=False)
    r = sliding_template_correlation(normalize_signal(comp_bp), template_z)
    return keep.tolist(), comp_bp, np.abs(r)


# ======================= Main component selection (Ebrahimzadeh) =======================
def select_components_template_ica(raw, spike_sec, half_win_s=0.15, band_comp=(3., 25.)):
    """Select ICA components that correlate with template at annotated IED windows."""
    from loguru import logger
    sf = raw.info['sfreq']
    best_ch, template_z, _, refined = build_template(
        raw, spike_sec, half_win_s=half_win_s, return_refined=True)

    # Augment template if small set
    augmented_spikes = augment_template(raw, spike_sec, template_z, best_ch)
    logger.info(f"Augmented spikes: {len(augmented_spikes)} (original: {len(spike_sec)})")

    # Multi-run ICA for stable candidates
    stable_indices = multi_run_ica(raw)
    logger.info(f"Stable candidate components: {stable_indices}")

    # Fit ICA once to get sources
    ica = ICA(n_components=min(20, raw.info['nchan']),  # Reduced for speed
              random_state=97, method='infomax', max_iter='auto')
    ica.fit(raw.copy())  # Raw is already filtered to 1-100
    S = ica.get_sources(raw).get_data()

    accepted = []
    timecourses = []
    hrf_regs = {}

    for idx in stable_indices:
        comp_tc = S[idx]
        comp_bp = filter_data(comp_tc, sf, band_comp[0], band_comp[1], verbose=False)
        logger.info(f"Checking component {idx}")
        if check_component_acceptance(comp_bp, template_z, augmented_spikes, sf):
            logger.info(f"Accepted component {idx}")
            accepted.append(idx)
            timecourses.append(comp_bp)
            hrf_regs[idx] = generate_hrf_regressors(comp_bp, sf)
        else:
            logger.info(f"Rejected component {idx}")

    return TemplateICADetection(
        template_z=template_z,
        best_channel=best_ch,
        refined_times=augmented_spikes,
        accepted_components=accepted,
        component_timecourses=timecourses,
        hrf_regressors=hrf_regs,
        ica=ica
    )







# ======================= Generic helpers =======================
def sliding_template_correlation(signal_z, template_z):
    """Temporal cross-correlation r(t) for template detection (Ebrahimzadeh 2021)."""
    L = len(template_z)
    num = correlate(signal_z, template_z, mode='same')
    kernel = np.ones(L) / L
    mean = np.convolve(signal_z, kernel, 'same')
    mean2 = np.convolve(signal_z**2, kernel, 'same')
    std = np.sqrt(np.maximum(mean2 - mean**2, 1e-12))
    r = num / (L * std)
    r[~np.isfinite(r)] = 0
    return r

def detect_peaks(r_trace, threshold, min_distance_samples):
    """Peak indices where r ≥ threshold (with refractory)."""
    peaks, _ = find_peaks(r_trace, height=threshold, distance=min_distance_samples)
    return peaks

def match_annotations(peaks, ann_times_s, sfreq, tol_s):
    """Match detected peaks to annotated times within ±tol_s."""
    tol = int(round(tol_s * sfreq))
    caught, missed = [], []
    for t in ann_times_s:
        samp = int(round(t * sfreq))
        if np.any(np.abs(peaks - samp) <= tol):
            caught.append(t)
        else:
            missed.append(t)
    return caught, missed

def normalize_signal(x, eps=1e-12):
    return (x - x.mean()) / (x.std() + eps)

# ======================= Multi-run ICA for stability =======================
def multi_run_ica(raw, n_runs=10, band_ica=(1., 100.), kurtosis_min=2.0, max_keep=3):
    """Run ICA multiple times to select stable candidate components."""
    from loguru import logger
    from collections import Counter
    component_counts = Counter()
    component_vars = {}  # index - list of explained vars

    for run in range(n_runs):
        random_state = run  # different seed each time
        ica = ICA(n_components=min(20, raw.info['nchan']),
                  random_state=random_state, method='infomax', max_iter='auto')
        ica.fit(raw.copy())  # Raw is already filtered to 1-100
        S = ica.get_sources(raw).get_data()
        k = kurtosis(S, axis=1, fisher=False)
        keep = np.where(k >= kurtosis_min)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(k))])
        for idx in keep:
            component_counts[idx] += 1
            if idx not in component_vars:
                component_vars[idx] = []
            component_vars[idx].append(ica.explained_var_[idx] if hasattr(ica, 'explained_var_') else 0)

    # Select top by frequency, then by average explained var
    candidates = sorted(component_counts.items(), key=lambda x: (x[1], np.mean(component_vars.get(x[0], [0]))), reverse=True)
    stable_indices = [idx for idx, _ in candidates[:max_keep]]
    logger.info(f"Component counts: {dict(component_counts)}")
    logger.info(f"Selected stable: {stable_indices}")
    return stable_indices

# ======================= Template augmentation =======================
def augment_template(raw, spike_sec, template_z, best_ch, high_r_min=0.96, high_r_max=0.98, refractory_s=0.15):
    """Augment spike times with high-correlation detections if initial set is small."""
    if len(spike_sec) >= 10:  # threshold for small set
        return spike_sec
    sf = raw.info['sfreq']
    chan_sig = raw.get_data(picks=[best_ch])[0]
    r = sliding_template_correlation(normalize_signal(chan_sig), template_z)
    th_high = high_r_min  # use min for threshold
    min_dist = int(round(refractory_s * sf))
    peaks = detect_peaks(r, th_high, min_dist)
    new_times = peaks / sf
    # Filter to high_r_max if needed, but for now add all >= min
    augmented = list(spike_sec) + [t for t in new_times if t not in spike_sec]
    return sorted(augmented)

# ======================= Windowed correlation at IEDs =======================
def check_component_acceptance(component_tc, template_z, spike_times, sfreq, window_s=1.0, min_corr=0.85):
    """Check if component correlates >= min_corr with template at all annotated IED windows."""
    from loguru import logger
    half_win = int(round(window_s / 2 * sfreq))
    for t in spike_times:
        samp = int(round(t * sfreq))
        start = max(0, samp - half_win)
        end = min(len(component_tc), samp + half_win)
        window_sig = component_tc[start:end]
        if len(window_sig) < len(template_z):
            continue  # skip if window too small
        r = sliding_template_correlation(normalize_signal(window_sig), template_z)
        max_r = np.max(np.abs(r))
        logger.info(f"Spike at {t:.2f}s: max correlation {max_r:.3f}")
        if max_r < min_corr:
            return False
    return True


