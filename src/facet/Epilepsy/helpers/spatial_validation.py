"""Spatial validation of TCCC components (scalp-space whole-map correlation).

Scalp-space whole-map correlation surrogate
-------------------------------------------
This module implements an OPTIONAL, additive spatial-plausibility step that is
applied *after* the existing temporal TCCC acceptance stage.  It is inspired by
the spatial-plausibility step of Ebrahimzadeh et al. (2021), in which candidate
components judged spatially discordant with the observed IED field (by more than
50 mm in source/dipole space) were rejected.

IMPORTANT SCIENTIFIC FRAMING (do not overstate):
- This is a *scalp-space adaptation*, NOT an exact reproduction of the original
  source/dipole-space > 50 mm criterion.  It must never be described as
  equivalent to the 50 mm rule.  The pipeline has no forward model, inverse
  solution, or dipole localisation, so the paper's source-space test cannot be
  reproduced here.
- The gate builds ONE patient-specific IED-reference scalp map (from the raw EEG
  and the refined IED times, via the existing ``_build_epileptic_map``) and, for
  each temporally accepted component, reconstructs that component alone, builds
  its scalp map the same way, and scores it by the absolute whole-map Pearson
  correlation |r| with the reference (absolute because ICA polarity is
  arbitrary).  A component is spatially accepted iff |r| >= 0.50.  The
  representative (for single-map reporting) is the surviving component with the
  highest |r|.  No adjacency rule, regional grouping, permutation null, or
  outcome-based tuning is used - the only spatial rule is the fixed |r| >= 0.50
  threshold (a pragmatic scalp-space adaptation, NOT the paper's 50 mm rule).
- The gate is a REAL SPATIAL FILTER: the surviving components REPLACE the TCCC
  accepted set, so all TCCC-derived outputs (scalp map, peak, region, standalone
  regressor) and the fused map use only the survivors.
  If no component reaches 0.50, a single best-|r| fallback component is kept so
  TCCC/fused still produce output (``spatial_gate_fallback_used=True``).
- EVMC is NEVER modified: it is built from the refined IED times (not the
  accepted components), so EVMC outputs are numerically identical to baseline.
- The IED reference field is derived locally here from the SAME annotated
  activity that underlies EVMC.  Consequently spatial-gated TCCC vs EVMC
  agreement is partly dependent and must NOT be treated as independent
  validation.  EVMC itself is never read or modified by this module.

The baseline temporal-selection function is never modified; this gate operates
purely on the already-produced :class:`TemplateICADetection` object.
"""

from __future__ import annotations

import numpy as np
import mne
from loguru import logger

from facet.Epilepsy.helpers.regressors import (
    _build_epileptic_map,
    compute_and_attach_ica_regressors,
)

# Fixed scalp-space spatial acceptance threshold. A temporally accepted ICA
# component is retained by TCCC only if the absolute whole-map Pearson
# correlation |r| between its reconstructed scalp map and the patient-specific
# IED reference map reaches this value. This is a pragmatic scalp-space
# adaptation, NOT the source-space 50 mm criterion of Ebrahimzadeh et al. (2021).
SPATIAL_ABS_CORR_THRESHOLD = 0.50


def _eeg_channel_names(raw) -> list:
    """EEG channel names in epileptic-map (pick) order."""
    picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    return [raw.ch_names[i] for i in picks]


def _map_peak_channel(emap, channel_names):
    """Peak electrode of a scalp map using the existing argmax(|map|) convention."""
    if emap is None or len(emap) == 0 or not channel_names:
        return None
    idx = int(np.argmax(np.abs(np.asarray(emap, dtype=float))))
    return channel_names[idx] if idx < len(channel_names) else None


def _abs_map_corr(map_a, map_b):
    """|Pearson r| between two index-aligned scalp maps (arbitrary ICA polarity).

    Both maps are built from the same EEG picks (identical channel order), so a
    plain correlation over the shared length is the whole-map spatial
    similarity.  Absolute value is used because an ICA component's sign is
    arbitrary.  Returns ``None`` if fewer than 3 channels overlap or either map
    is constant.
    """
    if map_a is None or map_b is None:
        return None
    a = np.asarray(map_a, dtype=float)
    b = np.asarray(map_b, dtype=float)
    n = min(len(a), len(b))
    if n < 3:
        return None
    a, b = a[:n], b[:n]
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(abs(np.corrcoef(a, b)[0, 1]))


def apply_spatial_validation(detection, half_win_s: float = 0.15,
                             band: tuple = (1.0, 30.0), sfreq: float = None,
                             tr: float = None,
                             threshold: float = SPATIAL_ABS_CORR_THRESHOLD):
    """Filter the TCCC accepted set by scalp-space whole-map correlation.

    A temporally accepted ICA component is retained only if the absolute
    whole-map Pearson correlation |r| between its reconstructed scalp map and the
    patient-specific IED-reference map reaches ``threshold`` (default 0.50). The
    surviving components REPLACE the TCCC accepted set in ``detection`` (in
    place), so every downstream TCCC output uses only the survivors. This is a
    pragmatic scalp-space adaptation, NOT the paper's 50 mm source-space rule.

    Fallback: if NO component reaches the threshold, the single component with
    the strongest available |r| (or, if every reconstruction failed, the
    strongest temporal candidate) is kept so TCCC/fused still produce output;
    ``spatial_gate_fallback_used`` is set True and ``spatial_accepted_components``
    is left empty (the fallback component is exposed only as the representative /
    ``final_tccc_accepted_components``, never as a spatially accepted component).

    Parameters
    ----------
    detection : TemplateICADetection
        Result of the baseline temporal TCCC selection (fitted ICA, temporally
        accepted components, refined IED times and raw). Mutated in place.
    half_win_s, band :
        Forwarded to :func:`_build_epileptic_map` for both the IED-reference
        field and each candidate's reconstructed field.
    sfreq, tr :
        Sampling frequency and fMRI TR, used to recompute the standalone
        continuous TCCC regressor for the filtered set. ``sfreq`` defaults to
        ``detection.raw.info['sfreq']`` when omitted.
    threshold :
        Absolute whole-map |r| acceptance cutoff (default 0.50).

    Returns
    -------
    gated_detection : TemplateICADetection
        The SAME object as ``detection``, with ``accepted_components`` (and the
        aligned timecourses / regressor) reduced to the spatial survivors (or the
        single fallback component).
    summary : dict
        See the keys populated below.
    """
    baseline_accepted = list(detection.accepted_components)
    raw = detection.raw
    ica = detection.ica
    refined_times = detection.refined_times
    scores = (detection.ica_selection_stats or {}).get("component_scores", {})
    window_corr = detection.per_component_window_corr or {}
    if sfreq is None and raw is not None:
        sfreq = raw.info["sfreq"]

    summary = {
        "spatial_abs_corr_threshold": float(threshold),
        "baseline_accepted_components": list(baseline_accepted),
        "temporally_accepted_candidates": list(baseline_accepted),
        "spatial_accepted_components": list(baseline_accepted),
        "final_tccc_accepted_components": list(baseline_accepted),
        "spatial_tccc_representative_component": None,
        "fused_representative_component": None,
        "spatial_representative_is_accepted": False,
        "n_temporally_accepted": len(baseline_accepted),
        "n_spatially_accepted": len(baseline_accepted),
        "n_spatially_rejected": 0,
        "spatial_rejected_indices": [],
        "spatial_error_indices": [],
        "spatial_gate_zero_pass": False,
        "spatial_gate_fallback_used": False,
        "ied_reference_peak_channel": None,
        "candidate_log": [],
    }

    if raw is None or ica is None or not baseline_accepted:
        summary["spatial_gate_error"] = "missing raw/ica/accepted components"
        logger.warning("Spatial TCCC gate skipped: missing raw/ica/accepted set.")
        return detection, summary

    eeg_names = _eeg_channel_names(raw)

    # --- (1-2) Build ONE IED-reference scalp field per subject and its peak. ---
    try:
        ied_map = _build_epileptic_map(
            raw, refined_times, half_win_s=half_win_s, band=band)
    except Exception as exc:  # noqa: BLE001
        summary["spatial_gate_error"] = f"IED reference field failed: {exc}"
        logger.warning(
            f"Spatial TCCC gate skipped: could not build IED reference field "
            f"({exc}); keeping baseline temporally-accepted set unchanged."
        )
        return detection, summary

    ied_peak = _map_peak_channel(ied_map, eeg_names)
    summary["ied_reference_peak_channel"] = ied_peak

    # --- (4) Per-candidate whole-map spatial correlation with the IED field. ---
    # Each temporally accepted component is reconstructed on its own, turned into
    # a scalp map the same way the IED reference is, and scored by the absolute
    # correlation of the two whole maps (|r|, because ICA polarity is arbitrary).
    # This is a scalp-space adaptation of the spatial-plausibility step, NOT an
    # exact reproduction of the original source-space TCCC criterion.
    candidate_log = []
    abs_r_by_idx = {}
    best_idx, best_abs_r = None, -1.0
    for idx in baseline_accepted:
        temporal_score = scores.get(idx)
        abs_r = None
        try:
            recon_ic = ica.apply(raw.copy(), include=[idx], verbose=False)
            comp_map = _build_epileptic_map(
                recon_ic, refined_times, half_win_s=half_win_s, band=band)
            abs_r = _abs_map_corr(comp_map, ied_map)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Spatial gate: component {idx} field reconstruction failed "
                f"({exc}); no spatial correlation for this component."
            )

        abs_r_by_idx[idx] = abs_r
        if abs_r is not None and abs_r > best_abs_r:
            best_abs_r, best_idx = abs_r, idx

        passed = abs_r is not None and abs_r >= threshold
        candidate_log.append({
            "component_index": idx,
            "temporal_score": temporal_score,
            "spatial_abs_corr": abs_r,
            "spatially_accepted": bool(passed),
            # selected_representative is filled in after the ranking below.
            "selected_representative": False,
        })
        logger.info(
            f"Spatial gate | comp {idx}: score="
            f"{'n/a' if temporal_score is None else f'{temporal_score:.3f}'}, "
            f"|r|={'n/a' if abs_r is None else f'{abs_r:.3f}'}, "
            f"passed={passed}"
        )

    # --- (5-6) Partition candidates by the fixed |r| >= threshold rule. ---
    survivors = [
        i for i in baseline_accepted
        if abs_r_by_idx.get(i) is not None and abs_r_by_idx[i] >= threshold
    ]
    rejected = [
        i for i in baseline_accepted
        if abs_r_by_idx.get(i) is not None and abs_r_by_idx[i] < threshold
    ]
    errored = [i for i in baseline_accepted if abs_r_by_idx.get(i) is None]

    zero_pass = len(survivors) == 0
    fallback_used = False
    if survivors:
        # One or more passed: final TCCC set = survivors; representative = the
        # surviving component with the highest |r|.
        final_components = survivors
        representative = max(survivors, key=lambda i: abs_r_by_idx[i])
    else:
        # None passed: keep a SINGLE fallback component so TCCC/fused still
        # produce output, but do NOT mark it spatially accepted.
        fallback_used = True
        if best_idx is not None:
            representative = best_idx  # strongest available |r|
        else:
            def _median_corr(i):
                wc = window_corr.get(i, [])
                return float(np.median(wc)) if len(wc) else -np.inf

            representative = max(baseline_accepted, key=_median_corr)
        final_components = [representative]
        logger.warning(
            f"Spatial TCCC gate: no component reached |r| >= {threshold:.2f}; "
            f"keeping fallback component {representative} so TCCC/fused still "
            f"produce output (spatial_gate_fallback_used=True, "
            f"spatial_accepted_components=[])."
        )

    for entry in candidate_log:
        entry["selected_representative"] = (
            entry["component_index"] == representative)

    # Apply spatial validation as a REAL FILTER: reduce the TCCC accepted set to
    # the spatially-validated survivors (or the single fallback component when
    # none pass). The representative is placed first so the standalone TCCC
    # regressor is driven by it; every downstream TCCC/fused output then uses
    # only these components.
    ordered_final = [representative] + [
        i for i in final_components if i != representative]
    tc_by_idx = dict(
        zip(detection.accepted_components, detection.component_timecourses))
    detection.accepted_components = list(ordered_final)
    detection.component_timecourses = [
        tc_by_idx[i] for i in ordered_final if i in tc_by_idx]
    if sfreq is not None:
        compute_and_attach_ica_regressors(detection, sfreq=sfreq, tr=tr)

    summary.update({
        "spatial_accepted_components": list(survivors),
        # Final TCCC set after spatial validation: the survivors, or the single
        # fallback component when none pass. ``representative`` is the highest-|r|
        # survivor, used for single-map TCCC reporting.
        "final_tccc_accepted_components": list(ordered_final),
        "spatial_tccc_representative_component": representative,
        "fused_representative_component": representative,
        "spatial_representative_is_accepted": representative in survivors,
        "n_spatially_accepted": len(survivors),
        "n_spatially_rejected": len(rejected),
        "spatial_rejected_indices": list(rejected),
        "spatial_error_indices": list(errored),
        "spatial_gate_zero_pass": zero_pass,
        "spatial_gate_fallback_used": fallback_used,
        "candidate_log": candidate_log,
    })

    logger.info(
        f"Spatial TCCC gate (|r| >= {threshold:.2f}): temporal={baseline_accepted} "
        f"-> survivors={survivors}, rejected={rejected}, errored={errored}; "
        f"final TCCC set={final_components}, representative=comp {representative} "
        f"(|r|={'n/a' if best_abs_r < 0 else f'{best_abs_r:.3f}'}, "
        f"fallback={fallback_used})."
    )
    return detection, summary
