import matplotlib.pyplot as plt
import numpy as np
from mne.io import Raw
from scipy.signal import find_peaks
from mne.preprocessing import ICA
import mne
from scipy.signal import correlate
from facet.Epilepsy.helpers.shared_utils import build_template
from mne.filter import filter_data
from scipy.stats import kurtosis  # used by legacy build_ica_composite
from facet.Epilepsy.Models.pipeline_results import TemplateICADetection
from facet.Epilepsy.helpers.regressors import generate_hrf_regressors
from facet.Epilepsy.helpers.diagnostic_utils import plot_ica_components_timecourses

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
def select_components_template_ica(raw, spike_sec, half_win_s=0.15, band_comp=(1., 30.), th_raw=0.85, match_tol_s=0.1, visualize=False):
    """Select ICA components that correlate with the IED template (Ebrahimzadeh 2021).

    Each candidate component is accepted if a high quantile of its single-trial
    cross-correlation with the template at the IED times reaches ``th_raw``
    (paper-faithful Template Component Cross-Correlation).
    """
    from loguru import logger
    sf = raw.info['sfreq']
    best_ch, template_z, _, refined = build_template(
        raw, spike_sec, half_win_s=half_win_s, return_refined=True, visualize=visualize)

    # Augment template if small set
    augmented_spikes = augment_template(raw, spike_sec, template_z, best_ch)
    logger.info(f"Augmented spikes: {len(augmented_spikes)} (original: {len(spike_sec)})")

    # Multi-run ICA: cluster components across runs and return cluster centroids
    # (mixing-vector templates of sources that appear most often / with largest λ).
    n_ica_runs = 10
    cluster_centroids, component_counts, component_lambdas = multi_run_ica(
        raw, n_runs=n_ica_runs)
    logger.info(f"Discovered {len(cluster_centroids)} stable component clusters")

    # Fit final ICA once to get sources, then map each cluster centroid to the
    # best-matching component of this final fit (greedy |Pearson| on mixing cols).
    n_eeg = len(mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads'))
    ica = ICA(n_components=min(20, n_eeg),  # Reduced for speed
              random_state=97, method='infomax', max_iter='auto')
    ica.fit(raw.copy())  # Raw is already filtered to 1-100
    S = ica.get_sources(raw).get_data()

    stable_indices = match_clusters_to_ica(cluster_centroids, ica)
    logger.info(f"Stable candidate components (final ICA indices): {stable_indices}")

    # Remove artifact ICs (ECG/EOG/muscle) from the candidate pool so a cardiac
    # or muscular source cannot be mistaken for an epileptic component
    # (Ebrahimzadeh 2021: artifact components are identified and removed first).
    artifact_ics = find_artifact_components(ica, raw)
    if artifact_ics:
        kept = [i for i in stable_indices if i not in artifact_ics]
        removed = [i for i in stable_indices if i in artifact_ics]
        # Only apply the exclusion if at least one candidate remains, so a
        # subject is never left with an empty candidate pool.
        if removed and kept:
            logger.info(f"Excluding artifact ICs from candidates: {removed}")
            stable_indices = kept
        elif removed and not kept:
            logger.warning(
                f"All stable candidates flagged as artifacts {removed}; "
                f"keeping them to avoid an empty candidate pool."
            )

    # Map cluster-level run-frequencies onto the final ICA component indices so
    # reproducibility can be reported per accepted component (run frequency = how
    # many of the ``n_ica_runs`` ICA repetitions that source appeared in).
    component_run_counts = {
        stable_indices[i]: int(component_counts.get(i, 0))
        for i in range(len(stable_indices))
    }

    # Evaluate each candidate component against the template at the IED times.
    accepted = []
    all_data = {}  # idx -> (comp_bp, per_window_corr, score)
    for idx in stable_indices:
        comp_tc = S[idx]
        comp_bp = filter_data(comp_tc, sf, band_comp[0], band_comp[1], verbose=False)
        logger.info(f"Checking component {idx}")
        is_accepted, per_window_corr, score = check_component_acceptance(
            comp_bp, template_z, augmented_spikes, sf,
            min_corr=th_raw, half_win_s=half_win_s)
        all_data[idx] = (comp_bp, per_window_corr, score)
        if is_accepted:
            accepted.append(idx)
        logger.info(
            f"Component {idx}: score={score:.3f} "
            f"({'accepted' if is_accepted else 'rejected'})"
        )

    # Guarantee at least one component per subject: if none reached the
    # threshold, fall back to the best-scoring candidate.
    fallback_used = False
    if not accepted and all_data:
        best_idx = max(all_data, key=lambda i: all_data[i][2])
        logger.warning(
            f"No component reached threshold {th_raw}; falling back to "
            f"best-scoring component {best_idx} (score={all_data[best_idx][2]:.3f})"
        )
        accepted = [best_idx]
        fallback_used = True
    logger.info(f"Accepted components: {accepted} (fallback={fallback_used})")

    # Build outputs for the accepted (or fallback) components
    timecourses = []
    hrf_regs = {}
    window_corr_map = {}
    component_scores = {idx: all_data[idx][2] for idx in all_data}
    for idx in accepted:
        comp_bp, per_window_corr, _ = all_data[idx]
        timecourses.append(comp_bp)
        hrf_regs[idx] = generate_hrf_regressors(comp_bp, sf)
        window_corr_map[idx] = per_window_corr

    # Optional visualization: plot timecourses + averaged epochs for accepted components
    if visualize and len(accepted) > 0:
        try:
            plot_ica_components_timecourses(raw, ica=ica, S=S, component_indices=accepted,
                                            template_z=template_z, spike_times=augmented_spikes)
        except Exception as e:
            logger.warning(f"Component timecourse plotting failed: {e}")

    return TemplateICADetection(
        template_z=template_z,
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
            'component_run_counts': component_run_counts,
            'component_scores': component_scores,
            'fallback_used': fallback_used,
            'threshold': th_raw,
            'template_channel': int(best_ch),
            'artifact_ics': artifact_ics,
            'n_runs': n_ica_runs,
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
def multi_run_ica(raw, n_runs=10, band_ica=(1., 100.), max_keep=3,
                  cluster_threshold=0.8):
    """Run ICA multiple times and cluster components across runs.

    Ebrahimzadeh 2021: "ICA algorithm was applied 10 times using different
    arbitrary (random) initialization weights, and the initial candidates
    selected based on being those seen most often in the 10 repetitions.
    From these, the three components with the highest average λ (weight of
    extracted independent components) across all 10 iterations were selected
    as final candidates."

    λ is the L2 norm of each component's column in the ICA mixing matrix (A),
    which quantifies the component's contribution to the observed EEG signal.

    Component-identity matching across runs
    ---------------------------------------
    ICA decompositions are permutation- and sign-invariant, so raw component
    indices are not comparable across runs. We cluster all (n_runs ×
    n_components) mixing vectors by absolute Pearson correlation of unit-norm
    columns. Each resulting cluster represents one source seen across runs.
    Clusters are ranked by the number of distinct runs they appear in
    (frequency, paper's "most often" criterion) and then by mean λ. The top
    ``max_keep`` cluster centroids are returned and can be matched to the
    components of any subsequent ICA fit via ``match_clusters_to_ica``.

    Parameters
    ----------
    raw : mne.io.Raw
        Pre-filtered EEG data.
    n_runs : int
        Number of ICA repetitions with different random seeds.
    max_keep : int
        Number of stable clusters to return (paper: 3).
    cluster_threshold : float
        Minimum |correlation| of mixing vectors for two components to be
        grouped into the same cluster.

    Returns
    -------
    cluster_centroids : list of ndarray
        Sign-aligned, unit-norm mixing-vector centroid for each selected
        cluster (shape: (n_channels,)). Use ``match_clusters_to_ica`` to
        map these to components of a final ICA fit.
    component_counts : dict[int, int]
        Selected-cluster index → number of distinct runs it appeared in.
    component_lambdas : dict[int, list[float]]
        Selected-cluster index → list of λ values from all member components.
    """
    from loguru import logger

    n_eeg = len(mne.pick_types(raw.info, eeg=True, meg=False, exclude='bads'))
    n_comp = min(20, n_eeg)

    # Collect every component from every run: (unit-norm mixing vector, λ, run, idx)
    all_vecs = []
    all_lambdas = []
    all_runs = []
    for run in range(n_runs):
        ica = ICA(n_components=n_comp, random_state=run,
                  method='infomax', max_iter='auto')
        ica.fit(raw.copy())
        mixing = ica.mixing_matrix_  # (n_channels, n_components)
        for k in range(ica.n_components_):
            vec = mixing[:, k]
            lam = float(np.linalg.norm(vec))
            if lam > 0:
                all_vecs.append(vec / lam)
                all_lambdas.append(lam)
                all_runs.append(run)

    if not all_vecs:
        logger.warning("multi_run_ica: no components collected")
        return [], {}, {}

    M = np.asarray(all_vecs)              # (n_total, n_channels), unit-norm rows
    lambdas_all = np.asarray(all_lambdas) # (n_total,)
    runs_all = np.asarray(all_runs)       # (n_total,)

    # Sign-invariant similarity (vectors are unit-norm → |dot| == |Pearson|)
    sim = np.abs(M @ M.T)

    # Greedy clustering: seed each cluster with the highest-λ unassigned vector,
    # absorb all unassigned vectors above the similarity threshold.
    n_total = M.shape[0]
    assigned = np.zeros(n_total, dtype=bool)
    order = np.argsort(-lambdas_all)
    clusters = []
    for i in order:
        if assigned[i]:
            continue
        mask = (sim[i] >= cluster_threshold) & (~assigned)
        members = np.where(mask)[0]
        if members.size == 0:
            continue
        clusters.append(members)
        assigned[members] = True

    # Build per-cluster statistics
    cluster_info = []
    for members in clusters:
        runs_in = np.unique(runs_all[members])
        member_lambdas = lambdas_all[members].tolist()
        seed_vec = M[members[0]]
        signs = np.sign(M[members] @ seed_vec)
        signs[signs == 0] = 1.0
        aligned = M[members] * signs[:, None]
        centroid = aligned.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        cluster_info.append({
            'centroid': centroid,
            'run_count': int(len(runs_in)),
            'mean_lambda': float(np.mean(member_lambdas)),
            'lambdas': member_lambdas,
        })

    # Paper-aligned ranking: primary = frequency across runs, secondary = mean λ
    cluster_info.sort(
        key=lambda c: (c['run_count'], c['mean_lambda']),
        reverse=True,
    )
    selected = cluster_info[:max_keep]

    cluster_centroids = [c['centroid'] for c in selected]
    component_counts = {i: c['run_count'] for i, c in enumerate(selected)}
    component_lambdas = {i: c['lambdas'] for i, c in enumerate(selected)}

    mean_lambda_str = [f"{c['mean_lambda']:.4f}" for c in selected]
    logger.info(
        f"multi_run_ica: {len(cluster_info)} clusters from {n_runs} runs "
        f"({n_total} total components, threshold={cluster_threshold})"
    )
    logger.info(
        f"Top-{max_keep} cluster run-frequencies: "
        f"{[c['run_count'] for c in selected]}"
    )
    logger.info(f"Top-{max_keep} cluster mean λ: {mean_lambda_str}")

    return cluster_centroids, component_counts, component_lambdas


def match_clusters_to_ica(cluster_centroids, ica):
    """Map cluster-centroid mixing vectors to components of a fitted ICA.

    Greedy matching by absolute Pearson correlation of mixing columns
    (sign-invariant). Each centroid is paired with a distinct ICA component.

    Parameters
    ----------
    cluster_centroids : list of ndarray
        Unit-norm mixing-vector centroids (output of ``multi_run_ica``).
    ica : mne.preprocessing.ICA
        Fitted ICA whose components will be matched.

    Returns
    -------
    matched_indices : list[int]
        Component index in ``ica`` for each centroid, in the same order.
    """
    from loguru import logger

    if not cluster_centroids:
        return []

    mixing = ica.mixing_matrix_  # (n_channels, n_components)
    norms = np.linalg.norm(mixing, axis=0)
    norms[norms == 0] = 1.0
    mixing_norm = mixing / norms[None, :]

    matched = []
    used = set()
    for ci, centroid in enumerate(cluster_centroids):
        sims = np.abs(mixing_norm.T @ centroid)  # (n_components,)
        for k in np.argsort(-sims):
            k_int = int(k)
            if k_int not in used:
                matched.append(k_int)
                used.add(k_int)
                logger.info(
                    f"Cluster {ci} → ICA component {k_int} "
                    f"(|r|={sims[k_int]:.3f})"
                )
                break
    return matched

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

# ======================= Artifact IC detection =======================
def find_artifact_components(ica, raw):
    """Identify ECG / EOG / muscle ICA components to exclude from candidates.

    Uses MNE's correlation/template detectors against the corresponding
    physiological channels (ECG/EOG) plus a spectral muscle heuristic.  Each
    detector is optional: if the required channel type is absent, that detector
    is skipped.  Returns a sorted list of unique component indices.

    Ebrahimzadeh 2021 removes eye-blink, eye-movement, cardiac, muscular,
    swallowing and machine-vibration components before selecting epileptic
    candidates; this is the automated analogue.
    """
    from loguru import logger
    bad = set()

    # Cardiac (needs an ECG channel — now typed correctly at load time)
    if mne.pick_types(ica.info, ecg=True, meg=False, eeg=False).size:
        try:
            inds, _ = ica.find_bads_ecg(raw, method="correlation",
                                        threshold="auto", verbose=False)
            bad.update(inds)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"find_bads_ecg skipped: {e}")

    # Ocular (needs an EOG channel)
    if mne.pick_types(ica.info, eog=True, meg=False, eeg=False).size:
        try:
            inds, _ = ica.find_bads_eog(raw, verbose=False)
            bad.update(inds)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"find_bads_eog skipped: {e}")

    # Muscle (spectral heuristic — no extra channel required)
    try:
        inds, _ = ica.find_bads_muscle(raw, verbose=False)
        bad.update(inds)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"find_bads_muscle skipped: {e}")

    return sorted(bad)


# ======================= Windowed correlation at IEDs =======================
def check_component_acceptance(component_tc, template_z, spike_times, sfreq,
                               window_s=0.3, min_corr=0.85, half_win_s=0.15,
                               single_trial_quantile=0.90):
    """Accept a component if its IED response matches the template (Ebrahimzadeh 2021).

    Paper: "Components that did not have cross-correlation with the templates at
    the times of the IED events of at least 0.85 were rejected", using a sliding
    window of width 0.3 s at the IED times (Eq. 1).

    A 0.3 s window (``window_s``) is slid at each IED and the max |r| with the
    template is taken per IED.  The component is accepted if a high quantile
    (``single_trial_quantile``, default 0.90) of those per-IED correlations
    reaches ``min_corr``.  The paper's wording ("did not have cross-correlation
    … of at least 0.85") describes the component having such correlation at the
    IEDs, so a high quantile is used rather than the median (which would be
    overly pessimistic on noisy single trials).

    Returns
    -------
    accepted : bool
    per_window_corr : list[float]
        Single-trial max |r| at each IED window (diagnostics).
    score : float
        The quantile of per-IED correlations used for the decision.
    """
    from loguru import logger

    # Slide a 0.3 s window at each IED (paper Eq. 1) and take the best alignment.
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
        per_window_corr.append(np.max(np.abs(r)))

    if per_window_corr:
        score = float(np.quantile(per_window_corr, single_trial_quantile))
    else:
        score = 0.0
    logger.info(
        f"decision score {score:.3f} "
        f"(q{single_trial_quantile:.2f} of {len(per_window_corr)} IED windows, "
        f"threshold {min_corr})"
    )
    return score >= min_corr, per_window_corr, score

