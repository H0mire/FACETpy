import os
import sys
import mne
from loguru import logger
import numpy as np
from scipy.io import loadmat

sys.path.append("../../src")

from facet.Epilepsy.preprocessing import load_mat_to_mne, filter_eeg, parse_spike_times, prepare_eeg_data
from facet.Epilepsy.correlation_utils import select_components_template_ica
from facet.Epilepsy.Models.pipeline_results import TemplateICADetection
from facet.Epilepsy.regressors import build_grouiller_regressor, generate_hrf_regressors, compute_and_attach_ica_regressors  

def run_ebrahimzadeh_pipeline(
    mat_path: str,
    sfreq: float = 500.0,
    half_win_s: float = 0.15,
    th_raw: float = 0.60,
    match_tol_s: float = 0.1,
    visualize: bool = False,
    tr: float = None,
) -> TemplateICADetection | None:
    """Run the Ebrahimzadeh (2021) EEG component selection pipeline.

    Core steps:
    1. Load EEG & filter (notch + band-pass)
    2. Parse annotated spike times
    3. Build peak‐aligned spike template (best channel auto selected)
    4. Augment template if small set
    5. Multi-run ICA to select stable candidates
    6. Check acceptance at annotated IED windows
    7. Generate HRF regressors for accepted components

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

    Returns
    -------
    TemplateICADetection
        Structured results including template, accepted components, timecourses, HRF regressors, and the raw object.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("facet.log", level="DEBUG")

    # Load, filter & parse
    logger.info(f"Loading and preparing EEG from {mat_path}")
    raw, raw_ica, spike_sec = prepare_eeg_data(mat_path, sfreq=sfreq)

    logger.info(f"Parsed {len(spike_sec)} spike annotations (raw)")
    if len(spike_sec) == 0:
        logger.warning("No spikes found; aborting template/detector stage.")
        return None

    # Core component selection (template build + multi-run ICA + acceptance check)
    detection = select_components_template_ica(
        raw_ica,
        spike_sec,
        half_win_s=half_win_s,
        th_raw=th_raw,
        match_tol_s=match_tol_s,
        visualize=visualize
    )

    logger.info(f"Refined/augmented spike times: {len(detection.refined_times)}")
    logger.info(f"Selected {len(detection.accepted_components)} accepted ICA components")

    # --- Generate Ebrahimzadeh (Continuous) Regressor ---
    if tr is not None:
        logger.info(f"Generating continuous ICA regressor (TR={tr}s)")
        detection = compute_and_attach_ica_regressors(detection, sfreq, tr)

    # Attach raw object and original spike annotations to result
    if detection:
        detection.raw = raw
        if detection.original_spike_sec is None:
            detection.original_spike_sec = list(spike_sec)

    return detection


def run_grouiller_pipeline(
    raw,                        # MNE Raw object (in-scanner EEG)
    spike_sec: list,            # Spike times in seconds (from long-term EEG annotations)
    half_win_s: float = 0.15,
    sfreq: float = 500.0,
    tr: float = 2.5,
    band: tuple = (1., 30.),
):
    """Grouiller 2011 topography-based regressor pipeline.

    Grouiller 2011: builds an epileptic voltage map from averaged spikes,
    computes its continuous spatial correlation with in-scanner EEG, squares
    the result, and convolves with a canonical HRF.

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
        {"regressor_hrf": regressor array, "epileptic_map": voltage map}
    """
    regressor, epileptic_map = build_grouiller_regressor(
        raw=raw,
        spike_sec=spike_sec,
        half_win_s=half_win_s,
        tr=tr,
        band=band,
    )

    return {"regressor_hrf": regressor, "epileptic_map": epileptic_map}


def run_combined_pipeline(
    mat_path: str,
    sfreq: float = 500.0,
    half_win_s: float = 0.15,
    th_raw: float = 0.60,
    match_tol_s: float = 0.1,
    visualize: bool = False,
    has_fmri: bool = False,
    tr: float = 2.5,
):
    """Run combined Ebrahimzadeh + Grouiller pipeline.

    Parameters
    ----------
    mat_path : str
        Path to .mat file.
    sfreq : float
        Sampling frequency (Hz).
    half_win_s : float
        Half window around each spike for template (s).
    th_raw : float
        Correlation threshold.
    match_tol_s : float
        Tolerance (s) for matching.
    visualize : bool
        If True, visualize ICA.
    has_fmri : bool
        If True, run Grouiller regressor step.
    tr : float
        fMRI Repetition Time (s).

    Returns
    -------
    dict
        Combined results.
    """
    # Run Ebrahimzadeh detection
    detection_result = run_ebrahimzadeh_pipeline(
        mat_path=mat_path,
        sfreq=sfreq,
        half_win_s=half_win_s,
        th_raw=th_raw,
        match_tol_s=match_tol_s,
        visualize=visualize,
        tr=tr if has_fmri else None
    )

    results = {"detection": detection_result}

    if has_fmri and detection_result is not None:
        # Run Grouiller topography-based regressor
        # Uses the raw EEG and spike annotations to build a spatial
        # correlation regressor (Grouiller 2011), NOT a stick-function.
        spikes_to_use = detection_result.refined_times if hasattr(detection_result, 'refined_times') else []
        
        regressor_result = run_grouiller_pipeline(
            raw=detection_result.raw,
            spike_sec=spikes_to_use,
            half_win_s=half_win_s,
            sfreq=sfreq,
            tr=tr,
        )
        results["regressor_grouiller"] = regressor_result
        
        # Add the ICA regressor to the top level results for easy access
        if hasattr(detection_result, 'regressor_ica'):
             results["regressor_ebrahimzadeh"] = detection_result.regressor_ica

    return results


