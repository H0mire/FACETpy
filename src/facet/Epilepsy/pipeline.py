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
    th_raw: float = 0.35,
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

    # Attach raw object to result
    if detection:
        detection.raw = raw

    return detection


def run_grouiller_pipeline(
    spike_times_sec: list,  # List of spike times (from annotations)
    total_duration_sec: float,  # Total EEG duration
    sfreq: float = 500.0,
    tr: float = 2.5,  # fMRI TR for resampling
    hrf_model: str = "canonical",
):
    """Standalone Grouiller pipeline: Build event-based regressor from spike times.

    Parameters
    ----------
    spike_times_sec : list
        Spike times in seconds.
    total_duration_sec : float
        Total recording duration (s).
    sfreq : float
        EEG sampling frequency (Hz).
    tr : float
        fMRI TR for regressor resampling.
    hrf_model : str
        HRF model ('canonical').

    Returns
    -------
    dict
        {"regressor_hrf": resampled HRF regressor array}
    """
    # For standalone, use provided spike times directly 
    final_spike_times = spike_times_sec
    
    # Build and return the HRF regressor
    regressor_hrf = build_grouiller_regressor(
        spike_times_sec=final_spike_times,
        total_duration_sec=total_duration_sec,
        tr=tr,
        hrf_model=hrf_model,
    )
    
    return {"regressor_hrf": regressor_hrf}


def run_combined_pipeline(
    mat_path: str,
    sfreq: float = 500.0,
    half_win_s: float = 0.15,
    th_raw: float = 0.35,
    match_tol_s: float = 0.1,
    visualize: bool = False,
    has_fmri: bool = False,
    tr: float = 2.5,
    hrf_model: str = "canonical",
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
    hrf_model : str
        HRF model.

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
        # Run Grouiller regressor
        # Use refined times if available, else original parsed times
        spikes_to_use = detection_result.refined_times if hasattr(detection_result, 'refined_times') else []
        
        regressor_result = run_grouiller_pipeline(
            spike_times_sec=spikes_to_use,
            total_duration_sec=detection_result.raw.n_times / sfreq,
            sfreq=sfreq,
            tr=tr,
            hrf_model=hrf_model,
        )
        results["regressor_grouiller"] = regressor_result
        
        # Add the ICA regressor to the top level results for easy access
        if hasattr(detection_result, 'regressor_ica'):
             results["regressor_ebrahimzadeh"] = detection_result.regressor_ica

    return results


