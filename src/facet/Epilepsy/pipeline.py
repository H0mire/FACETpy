import sys
from loguru import logger
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter1d, uniform_filter1d

sys.path.append("../../src")

from facet.Epilepsy.helpers.preprocessing import prepare_eeg_data
from facet.Epilepsy.helpers.correlation_utils import (
    select_components_template_ica, sliding_template_correlation,
    normalize_signal, detect_peaks,
)
from facet.Epilepsy.Models.pipeline_results import TemplateICADetection
from facet.Epilepsy.helpers.regressors import (
    build_grouiller_regressor, compute_and_attach_ica_regressors,
    _build_epileptic_map, _compute_spatial_correlation_timecourse, _double_gamma_hrf,
)

def run_ebrahimzadeh_pipeline(
    mat_path: str,
    sfreq: float = 500.0,
    half_win_s: float = 0.15,
    th_raw: float = 0.85,
    match_tol_s: float = 0.1,
    visualize: bool = False,
    tr: float = None,
    template_band: tuple = (1., 30.),
    max_template_spikes: int | None = 20,
) -> TemplateICADetection | None:
    """Run the Ebrahimzadeh (2021) EEG component selection pipeline.

    Parameters
    ----------
    mat_path : str
        Path to .mat file (expects 'eeg_data' and 'events').
    sfreq : float
        Sampling frequency (Hz).
    half_win_s : float
        Half window around each spike for template (s).
    th_raw : float
        Correlation threshold for component selection.
    match_tol_s : float
        Tolerance for matching spikes (s).
    visualize : bool
        Whether to visualize results.
    tr : float, optional
        fMRI Repetition Time (s). If provided, generates continuous ICA regressor.
    template_band : tuple
        Band-pass used strictly for template-stage operations
        (template build + augmentation correlation pass). Default: 1-30 Hz.
    max_template_spikes : int | None
        Maximum number of best-correlated spikes to average for the template.
        ``20`` approximates the paper's 10-20 spike hand-selection cap.

    Returns
    -------
    TemplateICADetection
        Structured results including template, accepted components, timecourses, HRF regressors, and the raw object.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("facet.log", level="DEBUG")

    logger.info(f"Loading and preparing EEG from {mat_path}")
    raw, raw_ica, spike_sec = prepare_eeg_data(mat_path, sfreq=sfreq)

    logger.info(f"Parsed {len(spike_sec)} spike annotations (raw)")
    if len(spike_sec) == 0:
        logger.warning("No spikes found; aborting template/detector stage.")
        return None

    raw_template = raw_ica.copy().filter(
        template_band[0], template_band[1], picks='eeg', verbose=False
    )

    detection = select_components_template_ica(
        raw_ica,
        spike_sec,
        half_win_s=half_win_s,
        th_raw=th_raw,
        match_tol_s=match_tol_s,
        visualize=visualize,
        max_template_spikes=max_template_spikes,
        template_raw=raw_template,
    )

    logger.info(f"Refined/augmented spike times: {len(detection.refined_times)}")
    logger.info(f"Selected {len(detection.accepted_components)} accepted ICA components")

    if tr is not None:
        logger.info(f"Generating continuous ICA regressor (TR={tr}s)")
        detection = compute_and_attach_ica_regressors(detection, sfreq, tr)

    if detection:
        detection.raw = raw
        if detection.original_spike_sec is None:
            detection.original_spike_sec = list(spike_sec)

    return detection


def run_grouiller_pipeline(
    raw,
    spike_sec: list,
    half_win_s: float = 0.15,
    sfreq: float = 500.0,
    tr: float = 2.5,
    band: tuple = (1., 30.),
):
    """Grouiller 2011 topography-based regressor pipeline.

    Parameters
    ----------
    raw : mne.io.Raw
        In-scanner EEG data.
    spike_sec : list
        Spike times in seconds.
    half_win_s : float
        Half-window for spike epoching (seconds).
    sfreq : float
        EEG sampling frequency (Hz).
    tr : float
        fMRI TR for regressor resampling.
    band : tuple
        Band-pass filter range (paper: 1–30 Hz).

    Returns
    -------
    dict
        {"regressor_hrf": regressor array, "epileptic_map": voltage map,
         "spatial_corr_timecourse": pre-HRF EEG-rate squared spatial
         correlation (for EEG-resolution temporal comparisons)}
    """
    regressor, epileptic_map, corr_sq = build_grouiller_regressor(
        raw=raw,
        spike_sec=spike_sec,
        half_win_s=half_win_s,
        tr=tr,
        band=band,
    )

    return {
        "regressor_hrf": regressor,
        "epileptic_map": epileptic_map,
        "spatial_corr_timecourse": corr_sq,
    }


def run_fused_pipeline(
    mat_path: str,
    sfreq: float = 500.0,
    half_win_s: float = 0.15,
    th_raw: float = 0.85,
    match_tol_s: float = 0.1,
    spatial_th: float = 0.85,
    fused_k: float = 6.0,
    band: tuple = (1., 30.),
    visualize: bool = False,
    has_fmri: bool = False,
    tr: float = 2.5,
):
    """Fused Ebrahimzadeh (TCCC) + Grouiller (EVMC) epilepsy pipeline.

    Parameters
    ----------
    mat_path : str
        Path to the subject .mat file.
    sfreq : float
        Sampling frequency (Hz).
    half_win_s : float
        Half-window for spike epoching / template (s).
    th_raw : float
        Temporal (template) correlation threshold (Ebrahimzadeh default 0.85).
    match_tol_s : float
        Tolerance (s) forwarded to the detection stage.
    spatial_th : float
        Spatial (topographic) correlation threshold (default 0.85).
    fused_k : float
        Detection sensitivity knob. Peaks are kept when the fused
        relative-elevation trace exceeds ``median + fused_k * MAD`` of itself.
        Higher -> fewer, higher-confidence detections; lower -> more. The
        threshold is self-calibrating per recording, so this single value
        should generalise across subjects (default 6.0).
    band : tuple
        Band-pass for map / spatial correlation (Grouiller: 1-30 Hz).
    visualize : bool
        Forwarded to the Ebrahimzadeh stage.
    has_fmri : bool
        If True, also build the standalone Grouiller/Ebrahimzadeh regressors
        and the fused fMRI regressor.
    tr : float
        fMRI repetition time (s), used only when ``has_fmri``.

    Returns
    -------
    dict
        ``{"detection", "regressor_grouiller", "regressor_ebrahimzadeh"
        (both only when ``has_fmri``), "fused"}``, where ``"fused"`` is
        ``{"epileptic_map", "r_temporal", "r_spatial", "detections_sec",
        "regressor_fused" (optional)}`` or ``None`` when no epileptic
        component was accepted.
    """
    detection_result = run_ebrahimzadeh_pipeline(
        mat_path=mat_path,
        sfreq=sfreq,
        half_win_s=half_win_s,
        th_raw=th_raw,
        match_tol_s=match_tol_s,
        visualize=visualize,
        tr=tr if has_fmri else None,
    )

    results = {"detection": detection_result}

    if has_fmri and detection_result is not None:
        spikes_to_use = detection_result.refined_times if hasattr(detection_result, 'refined_times') else []

        regressor_result = run_grouiller_pipeline(
            raw=detection_result.raw,
            spike_sec=spikes_to_use,
            half_win_s=half_win_s,
            sfreq=sfreq,
            tr=tr,
        )
        results["regressor_grouiller"] = regressor_result

    detection = results.get("detection")
    if detection is None or not detection.accepted_components:
        results["fused"] = None
        return results

    from facet.Epilepsy.helpers.spatial_validation import apply_spatial_validation
    detection, spatial_summary = apply_spatial_validation(
        detection, half_win_s=half_win_s, band=band,
        sfreq=detection.raw.info["sfreq"],
        tr=tr if has_fmri else None)
    results["spatial_gate"] = spatial_summary
    results["detection"] = detection
    if has_fmri and hasattr(detection, "regressor_ica"):
        results["regressor_ebrahimzadeh"] = detection.regressor_ica
    fused_components = list(detection.accepted_components)

    raw = detection.raw
    sf = raw.info["sfreq"]

    tc_by_idx = dict(
        zip(detection.accepted_components, detection.component_timecourses))
    fused_timecourses = [tc_by_idx[i] for i in fused_components if i in tc_by_idx]

    recon = detection.ica.apply(
        raw.copy(), include=list(fused_components), verbose=False)
    epileptic_map = _build_epileptic_map(
        recon, detection.refined_times, half_win_s=half_win_s, band=band)

    aligned = []
    for tc in fused_timecourses:
        r_signed = sliding_template_correlation(normalize_signal(tc), detection.template_z)
        sign = np.sign(r_signed[np.argmax(np.abs(r_signed))]) or 1.0
        aligned.append(sign * np.asarray(tc))
    composite = np.sum(aligned, axis=0)
    r_temporal = np.abs(
        sliding_template_correlation(normalize_signal(composite), detection.template_z))
    r_spatial = np.sqrt(np.clip(
        _compute_spatial_correlation_timecourse(raw, epileptic_map, band=band), 0.0, None))

    n = min(len(r_temporal), len(r_spatial))
    r_temporal, r_spatial = r_temporal[:n], r_spatial[:n]

    base_win = int(round(10.0 * sf))

    def _relative_elevation(x):
        mu = uniform_filter1d(x, size=base_win, mode="nearest")
        ex = np.clip(x - mu, 0.0, None)
        mad = uniform_filter1d(np.abs(x - mu), size=base_win, mode="nearest") + 1e-9
        return ex / mad

    z_t = _relative_elevation(r_temporal)
    z_s = _relative_elevation(r_spatial)
    tol = int(round(match_tol_s * sf))
    z_s = maximum_filter1d(z_s, size=2 * tol + 1)
    r_fused = np.sqrt(z_t * z_s)
    med = float(np.median(r_fused))
    mad = float(np.median(np.abs(r_fused - med))) + 1e-9
    height = med + fused_k * mad
    refractory = int(round(0.5 * sf))
    peaks = detect_peaks(r_fused, height, refractory)
    detections_sec = (peaks / sf).tolist()

    fused = {
        "epileptic_map": epileptic_map,
        "r_temporal": r_temporal,
        "r_spatial": r_spatial,
        "detections_sec": detections_sec,
    }

    if has_fmri and tr is not None:
        graded = (r_temporal ** 2) * (r_spatial ** 2)
        hrf_len = int(20 * sf)
        hrf = _double_gamma_hrf(np.arange(hrf_len) / sf)
        convolved = fftconvolve(graded, hrf)[:len(graded)]
        n_tr = int(np.floor((raw.n_times / sf) / tr))
        tr_times = np.arange(n_tr) * tr
        t_eeg = np.arange(len(convolved)) / sf
        fused["regressor_fused"] = np.interp(tr_times, t_eeg, convolved)

    results["fused"] = fused
    return results


