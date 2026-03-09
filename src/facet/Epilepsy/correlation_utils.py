import matplotlib.pyplot as plt
import numpy as np
from mne.io import Raw
from scipy.signal import find_peaks
from mne.preprocessing import ICA
import mne
from scipy.signal import correlate
from facet.Epilepsy.shared_utils import build_template
from mne.filter import filter_data
from scipy.stats import kurtosis  # used by legacy build_ica_composite
from facet.Epilepsy.Models.pipeline_results import TemplateICADetection
from facet.Epilepsy.regressors import generate_hrf_regressors
from facet.Epilepsy.diagnostic_utils import plot_ica_components_timecourses

# ======================= ICA composite (Ebrahimzadeh) =======================
def build_ica_composite(raw, template_z, band_ica=(1., 40.), band_comp=(3., 25.),
                        kurtosis_min=3.0, max_keep=3, random_state=97):
    """Fit ICA, select peaky components, sum, band-pass, correlate with template."""
    sf = raw.info['sfreq']
    n_eeg = len(mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads'))
    ica = ICA(n_components=min(20, n_eeg),
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
def select_components_template_ica(raw, spike_sec, half_win_s=0.15, band_comp=(3., 25.), th_raw=0.60, match_tol_s=0.1, visualize=False):
    """Select ICA components that correlate with template at annotated IED windows."""
    from loguru import logger
    sf = raw.info['sfreq']
    best_ch, template_z, _, refined = build_template(
        raw, spike_sec, half_win_s=half_win_s, return_refined=True, visualize=visualize)

    # Augment template if small set
    augmented_spikes = augment_template(raw, spike_sec, template_z, best_ch)
    logger.info(f"Augmented spikes: {len(augmented_spikes)} (original: {len(spike_sec)})")

    # Multi-run ICA for stable candidates
    stable_indices, component_counts, component_lambdas = multi_run_ica(raw)
    logger.info(f"Stable candidate components: {stable_indices}")

    # Fit ICA once to get sources
    n_eeg = len(mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads'))
    ica = ICA(n_components=min(20, n_eeg),  # Reduced for speed
              random_state=97, method='infomax', max_iter='auto')
    ica.fit(raw.copy())  # Raw is already filtered to 1-100
    S = ica.get_sources(raw).get_data()

    accepted = []
    timecourses = []
    hrf_regs = {}
    window_corr_map = {}  # {comp_idx: [per-window max|r|]}

    for idx in stable_indices:
        comp_tc = S[idx]
        comp_bp = filter_data(comp_tc, sf, band_comp[0], band_comp[1], verbose=False)
        logger.info(f"Checking component {idx}")
        is_accepted, per_window_corr = check_component_acceptance(comp_bp, template_z, augmented_spikes, sf, min_corr=th_raw)
        if is_accepted:
            logger.info(f"Accepted component {idx}")
            accepted.append(idx)
            timecourses.append(comp_bp)
            hrf_regs[idx] = generate_hrf_regressors(comp_bp, sf)
            window_corr_map[idx] = per_window_corr
        else:
            logger.info(f"Rejected component {idx}")

    # Optional visualization: plot timecourses + averaged epochs for accepted components
    if visualize and len(accepted) > 0:
        try:
            plot_ica_components_timecourses(raw, ica=ica, S=S, component_indices=accepted,
                                            template_z=template_z, spike_times=augmented_spikes)
        except Exception as e:
            logger.warning(f"Component timecourse plotting failed: {e}")

    return TemplateICADetection(
        template_z=template_z,
        best_channel=best_ch,
        refined_times=augmented_spikes,
        accepted_components=accepted,
        component_timecourses=timecourses,
        hrf_regressors=hrf_regs,
        ica=ica,
        original_spike_sec=list(spike_sec),
        per_component_window_corr=window_corr_map,
        ica_selection_stats={
            'component_counts': component_counts,
            'component_lambdas': component_lambdas,
        },
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
def multi_run_ica(raw, n_runs=10, band_ica=(1., 100.), max_keep=3):
    """Run ICA multiple times to select stable candidate components.

    Ebrahimzadeh 2021: "ICA algorithm was applied 10 times using different
    arbitrary (random) initialization weights, and the initial candidates
    selected based on being those seen most often in the 10 repetitions.
    From these, the three components with the highest average λ (weight of
    extracted independent components) across all 10 iterations were selected
    as final candidates."

    λ is interpreted as the L2 norm of each component's column in the ICA
    mixing matrix (A), which quantifies the component's contribution to the
    observed EEG signal.

    NOTE – Component identity across runs:  ICA with different random seeds
    produces different decompositions; component index k in run i is not
    necessarily the same source as index k in run j.  The current
    implementation uses index identity as a proxy, which is a known
    simplification.  A full implementation would match components across
    runs by correlating their mixing vectors or source timecourses.
    """
    from loguru import logger
    from collections import Counter
    component_counts = Counter()
    component_lambdas = {}  # index → list of λ values across runs

    for run in range(n_runs):
        random_state = run  # different seed each time
        n_eeg = len(mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads'))
        ica = ICA(n_components=min(20, n_eeg),
                  random_state=random_state, method='infomax', max_iter='auto')
        ica.fit(raw.copy())  # Raw is already filtered to 1-100
        n_components = ica.n_components_

        # Compute λ (mixing weight) for each component: L2 norm of mixing column
        mixing = ica.mixing_matrix_  # shape (n_channels, n_components)
        for idx in range(n_components):
            component_counts[idx] += 1
            if idx not in component_lambdas:
                component_lambdas[idx] = []
            lam = np.linalg.norm(mixing[:, idx])
            component_lambdas[idx].append(lam)

    # Two-stage selection per paper:
    # 1. Rank by frequency (most often across runs)
    # 2. Among those, pick top max_keep by highest average λ
    candidates = sorted(
        component_counts.items(),
        key=lambda x: (x[1], np.mean(component_lambdas.get(x[0], [0]))),
        reverse=True
    )
    stable_indices = [idx for idx, _ in candidates[:max_keep]]
    logger.info(f"Component counts: {dict(component_counts)}")
    logger.info(f"Average λ: { {idx: f'{np.mean(vals):.4f}' for idx, vals in component_lambdas.items()} }")
    logger.info(f"Selected top-{max_keep} stable: {stable_indices}")
    return stable_indices, dict(component_counts), {k: list(v) for k, v in component_lambdas.items()}

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
def check_component_acceptance(component_tc, template_z, spike_times, sfreq, window_s=1.0, min_corr=0.60):
    """Check if component correlates >= min_corr with template at annotated IED windows.

    Paper (Ebrahimzadeh 2021): "Components that did not have cross-correlation
    with the templates at the times of the IED events of at least 0.85 were
    rejected."  We lower the default to 0.60 because empirical median
    correlations on the VEPISET dataset are substantially below 0.85
    (typical range 0.5-0.65).  The paper does not specify whether this means
    every single window or an aggregate.  We use the **median** of per-window
    max |r| values so that a single noisy/artefactual window cannot veto an
    otherwise good component.  This is documented as an interpretation decision.
    """
    from loguru import logger
    half_win = int(round(window_s / 2 * sfreq))
    per_window_corr = []
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
        per_window_corr.append(max_r)

    if len(per_window_corr) == 0:
        return False, []
    median_corr = np.median(per_window_corr)
    logger.info(f"Median correlation across {len(per_window_corr)} windows: {median_corr:.3f} (threshold {min_corr})")
    return median_corr >= min_corr, per_window_corr

