"""Single-subject statistical evaluation for the Ebrahimzadeh / Grouiller pipeline.

One .mat file = one subject.

Produces (under ``src/facet/Epilepsy/evaluation/results/``):
  results/{subject}/results_{subject}_summary.csv
  results/{subject}/results_{subject}_component_detail.csv
  results/{subject}/arrays_{subject}.npz
  results/{subject}/fig_{subject}_acceptance.png
  results/{subject}/fig_{subject}_window_corr_distribution.png
  results/{subject}/fig_{subject}_template.png
  results/{subject}/fig_{subject}_ica_topomaps.png
  results/{subject}/fig_{subject}_grouiller_map.png
  results/{subject}/fig_{subject}_fused_map.png

Usage:
    python -m facet.Epilepsy.evaluation.evaluate_subject                    # DA00100T.mat
    python -m facet.Epilepsy.evaluation.evaluate_subject --mat-file DA00103A.mat
    python -m facet.Epilepsy.evaluation.evaluate_subject --mat-file /abs/path/to/file.mat
"""

import os
import sys

# --- Setup Python Path ---
project_root = r"D:\Medical Engineering and Analytics\Project\FACETpy"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
# -------------------------

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from facet.Epilepsy.pipeline import run_fused_pipeline
from facet.Epilepsy.helpers.preprocessing import prepare_eeg_data
from facet.Epilepsy.evaluation.plots import (
    plot_acceptance_summary, plot_window_corr_distribution,
    plot_template, 
    plot_ica_topomaps, plot_grouiller_map,
)

# ── Constants ────────────────────────────────────────────────────────────────

from facet.Epilepsy.evaluation.config import (
    SFREQ, TR, TH_RAW, HALF_WIN_S, MATCH_TOL_S, MAT_DIR, RESULTS_DIR,
)


# ── Subject record ──────────────────────────────────────────────────────────

@dataclass
class SubjectRecord:
    """All data collected from a single .mat file (= one subject)."""
    subject: str                              # stem, e.g. "DA00100T"
    mat_path: str
    n_spikes_annotated: int
    n_spikes_augmented: int
    n_accepted_components: int
    accepted_indices: list = field(default_factory=list)
    template_z: Optional[np.ndarray] = None
    per_component_window_corr: dict = field(default_factory=dict)
    ica_selection_stats: dict = field(default_factory=dict)
    regressor_ebrahimzadeh: Optional[np.ndarray] = None
    regressor_grouiller: Optional[np.ndarray] = None
    epileptic_map: Optional[np.ndarray] = None
    channel_names: list = field(default_factory=list)
    fused_epileptic_map: Optional[np.ndarray] = None
    detection: object = None
    # Spatial-validation summary; ``None`` only when no components were accepted.
    spatial_gate: Optional[dict] = None


# ── Collection ───────────────────────────────────────────────────────────────

def run_pipeline_for_subject(mat_path: str) -> SubjectRecord:
    """Run the combined pipeline on one .mat file and return a SubjectRecord.

    Spatial validation is applied after temporal TCCC acceptance, so the
    returned record's accepted components, TCCC/fused maps and regressor reflect
    the spatially-validated set.
    """
    subject = os.path.splitext(os.path.basename(mat_path))[0]
    _, _, spike_sec_raw = prepare_eeg_data(mat_path, sfreq=SFREQ)

    result = run_fused_pipeline(
        mat_path=mat_path, sfreq=SFREQ, half_win_s=HALF_WIN_S,
        th_raw=TH_RAW, match_tol_s=MATCH_TOL_S, has_fmri=True, tr=TR,
        visualize=False,
    )

    detection = result.get("detection")
    grouiller = result.get("regressor_grouiller", {})
    fused = result.get("fused") or {}
    spatial_gate = result.get("spatial_gate")

    channel_names = _eeg_channel_names(detection)

    if detection is None:
        return SubjectRecord(
            subject=subject, mat_path=mat_path,
            n_spikes_annotated=len(spike_sec_raw),
            n_spikes_augmented=0, n_accepted_components=0,
            spatial_gate=spatial_gate,
        )

    n_annotated = (
        len(detection.original_spike_sec)
        if detection.original_spike_sec else len(spike_sec_raw)
    )
    # Only the spikes ADDED by template augmentation; 0 when no augmentation
    # occurred (refined set equals the annotated set).
    n_added = max(0, len(detection.refined_times) - n_annotated)

    return SubjectRecord(
        subject=subject,
        mat_path=mat_path,
        n_spikes_annotated=n_annotated,
        n_spikes_augmented=n_added,
        n_accepted_components=len(detection.accepted_components),
        accepted_indices=detection.accepted_components,
        template_z=detection.template_z,
        per_component_window_corr=detection.per_component_window_corr or {},
        ica_selection_stats=detection.ica_selection_stats or {},
        regressor_ebrahimzadeh=detection.regressor_ica,
        regressor_grouiller=grouiller.get("regressor_hrf"),
        epileptic_map=grouiller.get("epileptic_map"),
        channel_names=channel_names,
        fused_epileptic_map=fused.get("epileptic_map"),
        detection=detection,
        spatial_gate=spatial_gate,
    )


from facet.Epilepsy.evaluation.evaluation_helpers import (
    _eeg_channel_names,
    template_channel_info,
    grouiller_peak_channel,
    grouiller_focality,
    spike_field_consistency,
    summarize_cross_method_concordance,
)


# ── DataFrames ───────────────────────────────────────────────────────────────

def build_summary_dataframe(rec: SubjectRecord) -> pd.DataFrame:
    """Build results_{subject}_summary.csv — one row with key metrics."""
    lambdas = rec.ica_selection_stats.get("component_lambdas", {})

    median_corrs, mean_lams = [], []
    for idx in rec.accepted_indices:
        wc = rec.per_component_window_corr.get(idx, [])
        median_corrs.append(f"{np.median(wc):.4f}" if wc else "N/A")
        vals = lambdas.get(idx, [])
        mean_lams.append(f"{np.mean(vals):.4f}" if vals else "N/A")

    tmpl_ch_name, tmpl_ch_type = template_channel_info(rec)

    row = {
        "subject": rec.subject,
        "mat_file": os.path.basename(rec.mat_path),
        "n_spikes_annotated": rec.n_spikes_annotated,
        "n_spikes_augmented": rec.n_spikes_augmented,
        # ── Template ──────────────────────────────────────────────────
        "template_channel": tmpl_ch_name,
        "template_channel_type": tmpl_ch_type,
        # ── Component selection (TCCC, Ebrahimzadeh 2021) ─────────────
        "n_accepted_components": rec.n_accepted_components,
        "accepted_indices": ";".join(str(i) for i in rec.accepted_indices),
        "fallback_used": bool(rec.ica_selection_stats.get("fallback_used", False)),
        "median_corr_at_IEDs": ";".join(median_corrs),
        "grouiller_peak_channel": grouiller_peak_channel(rec),
        "grouiller_focality": grouiller_focality(rec),
        "template_length_samples": (
            len(rec.template_z) if rec.template_z is not None else 0
        ),
        "max_template_spikes_used": rec.ica_selection_stats.get(
            "max_template_spikes_used"
        ),
    }
    # ── Non-circular EEG-level spatial validation (inter-spike topographic
    #    consistency: do the annotated spikes share one focal generator?) ──
    row.update(spike_field_consistency(rec))
    # ── Cross-method concordance (do the independently-derived foci agree?) ──
    row.update(summarize_cross_method_concordance(rec))
    # ── Optional experimental spatial-gate summary (only in spatial mode) ──
    if rec.spatial_gate is not None:
        sg = rec.spatial_gate
        row.update({
            "spatial_abs_corr_threshold": sg.get("spatial_abs_corr_threshold"),
            # TCCC and FUSED both use the spatially-validated set
            # (final_tccc_accepted_components); temporally_accepted_candidates is
            # the full pre-gate temporal set, retained for transparency.
            "temporally_accepted_candidates": ";".join(
                str(i) for i in sg.get("temporally_accepted_candidates",
                                       sg.get("baseline_accepted_components", []))),
            "final_tccc_accepted_components": ";".join(
                str(i) for i in sg.get("final_tccc_accepted_components", [])),
            "baseline_accepted_components": ";".join(
                str(i) for i in sg.get("baseline_accepted_components", [])),
            # Strict spatial survivors with |r| >= threshold (empty if none pass).
            "spatial_accepted_components": ";".join(
                str(i) for i in sg.get("spatial_accepted_components", [])),
            "spatial_rejected_indices": ";".join(
                str(i) for i in sg.get("spatial_rejected_indices", [])),
            "spatial_error_indices": ";".join(
                str(i) for i in sg.get("spatial_error_indices", [])),
            # Highest-|r| survivor, used for single-map TCCC reporting.
            "spatial_tccc_representative_component": sg.get(
                "spatial_tccc_representative_component"),
            "fused_representative_component": sg.get(
                "fused_representative_component",
                sg.get("spatial_tccc_representative_component")),
            "spatial_representative_is_accepted": bool(
                sg.get("spatial_representative_is_accepted", False)),
            "n_temporally_accepted": sg.get("n_temporally_accepted"),
            "n_spatially_accepted": sg.get("n_spatially_accepted"),
            "n_spatially_rejected": sg.get("n_spatially_rejected"),
            "spatial_gate_zero_pass": bool(sg.get("spatial_gate_zero_pass", False)),
            "spatial_gate_fallback_used": bool(
                sg.get("spatial_gate_fallback_used", False)),
            "ied_reference_peak_channel": sg.get("ied_reference_peak_channel"),
        })
        if sg.get("spatial_gate_error"):
            row["spatial_gate_error"] = sg.get("spatial_gate_error")
    return pd.DataFrame([row])


def build_component_detail_dataframe(rec: SubjectRecord) -> pd.DataFrame:
    """Build results_{subject}_component_detail.csv — one row per accepted component."""
    lambdas = rec.ica_selection_stats.get("component_lambdas", {})
    rows = []
    for idx in rec.accepted_indices:
        wc = rec.per_component_window_corr.get(idx, [])
        vals = lambdas.get(idx, [])
        n_above = sum(1 for v in wc if v >= TH_RAW) if wc else 0
        rows.append({
            "subject": rec.subject,
            "component_idx": idx,
            "avg_lambda": float(np.mean(vals)) if vals else np.nan,
            "median_window_corr": float(np.median(wc)) if wc else np.nan,
            "max_window_corr": float(np.max(wc)) if wc else np.nan,
            "min_window_corr": float(np.min(wc)) if wc else np.nan,
            "n_windows_above_threshold": n_above,
            "n_windows_total": len(wc),
        })
    return pd.DataFrame(rows)


# ── Validation ───────────────────────────────────────────────────────────────

def validate_outputs(rec: SubjectRecord) -> list[str]:
    """Sanity checks V1–V7.  Returns list of warning strings (empty = all OK)."""
    errors: list[str] = []
    det = rec.detection
    s = rec.subject

    # V1: No NaN/Inf in regressors
    for label, reg in [("ebrahimzadeh", rec.regressor_ebrahimzadeh),
                       ("grouiller", rec.regressor_grouiller)]:
        if reg is not None and not np.all(np.isfinite(reg)):
            errors.append(f"{s}: {label} regressor contains NaN/Inf")

    # V2: Regressor length matches expected n_TR
    if det is not None and det.raw is not None:
        total_dur = det.raw.n_times / SFREQ
        expected_n_tr = int(np.floor(total_dur / TR))
        for label, reg in [("ebrahimzadeh", rec.regressor_ebrahimzadeh),
                           ("grouiller", rec.regressor_grouiller)]:
            if reg is not None and len(reg) != expected_n_tr:
                errors.append(
                    f"{s}: {label} regressor length {len(reg)} "
                    f"!= expected {expected_n_tr}"
                )

    # V3: Accepted indices within range
    if det is not None:
        n_comp = (det.ica.n_components_
                  if hasattr(det.ica, "n_components_") else 20)
        for idx in rec.accepted_indices:
            if idx >= n_comp:
                errors.append(
                    f"{s}: accepted index {idx} >= n_components {n_comp}"
                )

    # V4: Median per-window corr >= threshold for accepted
    for idx in rec.accepted_indices:
        wc = rec.per_component_window_corr.get(idx, [])
        if wc and np.median(wc) < TH_RAW:
            errors.append(
                f"{s}: comp {idx} median corr {np.median(wc):.4f} < {TH_RAW}"
            )

    # V5: Accepted <= 3
    if rec.n_accepted_components > 3:
        errors.append(f"{s}: {rec.n_accepted_components} accepted > 3")

    # V6: Template length
    if rec.template_z is not None:
        expected_len = int(2 * HALF_WIN_S * SFREQ)
        if len(rec.template_z) != expected_len:
            errors.append(
                f"{s}: template length {len(rec.template_z)} "
                f"!= expected {expected_len}"
            )

    # V7: Epileptic map length = 19 EEG channels
    if rec.epileptic_map is not None and len(rec.epileptic_map) != 19:
        errors.append(
            f"{s}: epileptic map length {len(rec.epileptic_map)} != 19"
        )

    return errors


def _safe_to_csv(df: pd.DataFrame, path: str, **kwargs) -> str:
    """Save ``df`` to ``path``, falling back to a timestamped name if locked.

    Prevents a locked file (e.g. the CSV is open in Excel) from crashing the
    whole evaluation after the expensive pipeline has already run — losing
    that compute would otherwise force a full re-run just to get the CSV
    saved. Returns the path actually written to.
    """
    try:
        df.to_csv(path, **kwargs)
        print(f"  Saved {path}")
        return path
    except PermissionError:
        import datetime
        stem, ext = os.path.splitext(path)
        fallback = f"{stem}_{datetime.datetime.now():%Y%m%d_%H%M%S}{ext}"
        df.to_csv(fallback, **kwargs)
        print(
            f"  ⚠ {path} is locked (open in another program?) — "
            f"saved to {fallback} instead."
        )
        return fallback


# ── Main orchestrator ────────────────────────────────────────────────────────

def run_evaluation(mat_path: str):
    """Execute the full single-subject evaluation for one .mat file."""
    subject = os.path.splitext(os.path.basename(mat_path))[0]
    out_dir = os.path.join(RESULTS_DIR, subject)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Subject:          {subject}")
    print(f"Input:            {os.path.abspath(mat_path)}")
    print(f"Output directory: {os.path.abspath(out_dir)}")

    # ── Run pipeline ────────────────────────────────────────────────────
    print(f"\nRunning pipeline for {subject} ...")
    rec = run_pipeline_for_subject(mat_path)
    print(f"  → annotated={rec.n_spikes_annotated}, "
          f"augmented={rec.n_spikes_augmented}, "
          f"accepted={rec.n_accepted_components}")
    if rec.spatial_gate is not None:
        sg = rec.spatial_gate
        print(
            f"  TCCC temporal={sg.get('baseline_accepted_components')} "
            f"-> spatial survivors={sg.get('spatial_accepted_components')} "
            f"| TCCC/FUSED final={sg.get('final_tccc_accepted_components')} "
            f"(rep={sg.get('spatial_tccc_representative_component')}, "
            f"fallback={sg.get('spatial_gate_fallback_used')})"
        )

    # ── Validation (V1–V8) ──────────────────────────────────────────────
    print("\n--- Validation checks ---")
    errors = validate_outputs(rec)
    if errors:
        print(f"  ⚠ {len(errors)} validation warning(s):")
        for e in errors:
            print(f"    • {e}")
    else:
        print("  ✓ All validation checks passed.")

    # ── CSVs ────────────────────────────────────────────────────────────
    print("\n--- Generating CSVs ---")
    df_sum = build_summary_dataframe(rec)
    csv1 = os.path.join(out_dir, f"results_{subject}_summary.csv")
    _safe_to_csv(df_sum, csv1, index=False)

    df_cd = build_component_detail_dataframe(rec)
    csv2 = os.path.join(out_dir, f"results_{subject}_component_detail.csv")
    _safe_to_csv(df_cd, csv2, index=False)

    # ── Per-candidate spatial-gate log (spatial mode only) ──────────────
    if rec.spatial_gate is not None:
        clog = rec.spatial_gate.get("candidate_log", [])
        if clog:
            df_sg = pd.DataFrame(clog)
            df_sg.insert(0, "subject", subject)
            csv3 = os.path.join(
                out_dir, f"results_{subject}_spatial_gate_detail.csv")
            _safe_to_csv(df_sg, csv3, index=False)

    # ── EEG-level spatial validation (non-circular) ────────────────
    sv = spike_field_consistency(rec)
    if sv["spatial_n_spikes_used"] >= 2:
        print(
            f"  Spatial validation (n={sv['spatial_n_spikes_used']} spikes): "
            f"topo-consistency mean={sv['spatial_topo_consistency_mean']:.3f}, "
            f"median={sv['spatial_topo_consistency_median']:.3f}, "
            f"min={sv['spatial_topo_consistency_min']:.3f}, "
            f"mean-map focality={sv['spatial_mean_map_focality']:.2f}"
        )

    # ── Cross-method concordance (do independent foci agree on a location?) ──
    cc = summarize_cross_method_concordance(rec)
    print(
        f"  Concordance: "
        f"grouiller={cc['grouiller_region']}({cc['grouiller_peak_channel']}), "
        f"ica={cc['ica_region']}({cc['ica_peak_channel']}), "
        f"fused={cc['fused_region']}({cc['fused_peak_channel']}) "
        f"→ {cc['n_methods_agreeing']} agree"
        + ("  ✓ all agree" if cc["concordance_all_agree"] else "")
    )
    print(
        f"  Concordance (clinical lobe): "
        f"grouiller={cc['grouiller_lobe']}, ica={cc['ica_lobe']}, "
        f"fused={cc['fused_lobe']} → {cc['n_methods_agreeing_lobe']} agree"
        + ("  ✓ all agree" if cc["concordance_all_agree_lobe"] else "")
    )
    mc = cc.get("map_corr_mean")
    if mc is not None:
        def _f(x):
            return f"{x:.2f}" if x is not None else "n/a"
        print(
            f"  Map correlation |r|: "
            f"grouiller-fused={_f(cc['map_corr_grouiller_fused'])}, "
            f"grouiller-ica={_f(cc['map_corr_grouiller_ica'])}, "
            f"fused-ica={_f(cc['map_corr_fused_ica'])} "
            f"(mean={mc:.2f})"
            + ("  [ica corr n/a: single accepted component → fused≡ica]"
               if cc.get("map_corr_ica_degenerate") else "")
        )

    # ── Arrays (for group-level aggregation) ────────────────────────────
    npz_path = os.path.join(out_dir, f"arrays_{subject}.npz")
    np.savez(
        npz_path,
        regressor_ebrahimzadeh=(
            rec.regressor_ebrahimzadeh
            if rec.regressor_ebrahimzadeh is not None else np.array([])
        ),
        regressor_grouiller=(
            rec.regressor_grouiller
            if rec.regressor_grouiller is not None else np.array([])
        ),
        template_z=(
            rec.template_z if rec.template_z is not None else np.array([])
        ),
        epileptic_map=(
            rec.epileptic_map if rec.epileptic_map is not None else np.array([])
        ),
    )
    print(f"  Saved {npz_path}")

    # ── Figures ─────────────────────────────────────────────────────────
    print("\n--- Generating figures ---")
    plot_acceptance_summary(
        rec, os.path.join(out_dir, f"fig_{subject}_acceptance.png"))
    plot_window_corr_distribution(
        rec, os.path.join(out_dir, f"fig_{subject}_window_corr_distribution.png"))
    plot_template(
        rec, os.path.join(out_dir, f"fig_{subject}_template.png"))
    plot_ica_topomaps(
        rec, os.path.join(out_dir, f"fig_{subject}_ica_topomaps.png"))
    plot_grouiller_map(
        rec, os.path.join(out_dir, f"fig_{subject}_grouiller_map.png"))
    plot_grouiller_map(
        rec, os.path.join(out_dir, f"fig_{subject}_fused_map.png"),
        emap=rec.fused_epileptic_map, title="Fused Epileptic Map")

    print(f"\n✓ Single-subject evaluation complete for {subject}.")
    print(f"  Outputs in: {os.path.abspath(out_dir)}")
    return rec


# ── CLI ──────────────────────────────────────────────────────────────────────

def _resolve_mat_path(arg: str) -> str:
    """Turn a subject id, filename, or absolute path into a resolved .mat path."""
    if os.path.isabs(arg) or os.path.isfile(arg):
        return os.path.abspath(arg)
    # Treat as filename (with or without .mat) inside the default MAT_DIR
    candidates = [arg] if arg.lower().endswith(".mat") else [arg, f"{arg}.mat"]
    for name in candidates:
        candidate = os.path.join(MAT_DIR, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Cannot find '{arg}' — tried as absolute path and in {MAT_DIR}"
    )


def _evaluate_one_worker(path: str) -> tuple[str, bool, str]:
    """Process-pool worker: run one subject and return a picklable status.

    Only side effects (the CSV/NPZ/PNG files written under the results root)
    matter, so nothing from the heavy ``SubjectRecord`` is returned across the
    process boundary. Returns ``(subject, ok, error_message)``.
    """
    subject = os.path.splitext(os.path.basename(path))[0]
    try:
        run_evaluation(path)
        return subject, True, ""
    except Exception as e:  # noqa: BLE001 — keep the batch alive
        return subject, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Single-subject statistical evaluation "
                    "(1 .mat file = 1 subject). Accepts one or more "
                    "--mat-file values to run several subjects in a row."
    )
    parser.add_argument(
        "--mat-file", "-m", nargs="+", default=None,
        help="One or more filenames or full paths to .mat files to evaluate.",
    )
    parser.add_argument(
        "--subjects-file", default=None,
        help="Path to a text file with one subject id / filename per line.",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=1,
        help="Number of subjects to evaluate in parallel processes. Each "
             "subject is fully independent (fixed ICA seeds, its own output "
             "folder), so results are identical to sequential runs. Use -1 for "
             "all CPU cores. Default 1 (sequential).",
    )
    args = parser.parse_args()

    if args.subjects_file:
        with open(args.subjects_file) as f:
            names = [ln.strip() for ln in f if ln.strip()]
    elif args.mat_file:
        names = args.mat_file
    else:
        names = ["DA00100S.mat"]

    mat_paths = [_resolve_mat_path(n) for n in names]

    if len(mat_paths) == 1:
        run_evaluation(mat_paths[0])
        return

    failed: list[tuple[str, str]] = []

    n_jobs = args.jobs
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    n_jobs = max(1, min(n_jobs, len(mat_paths)))

    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"Running {len(mat_paths)} subjects across {n_jobs} parallel "
              f"processes ...")
        done = 0
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    _evaluate_one_worker, path): path
                for path in mat_paths
            }
            for future in as_completed(futures):
                subject, ok, err = future.result()
                done += 1
                if ok:
                    print(f"[{done}/{len(mat_paths)}] ✓ {subject}")
                else:
                    print(f"[{done}/{len(mat_paths)}] ✗ {subject} failed: {err}")
                    failed.append((subject, err))
    else:
        for i, path in enumerate(mat_paths, 1):
            subject = os.path.splitext(os.path.basename(path))[0]
            print(f"\n[{i}/{len(mat_paths)}] === {subject} ===")
            try:
                run_evaluation(path)
            except Exception as e:  # noqa: BLE001 — keep batch going
                print(f"  ✗ {subject} failed: {e}")
                failed.append((subject, str(e)))

    print(f"\nBatch complete: {len(mat_paths) - len(failed)}/{len(mat_paths)} succeeded.")
    for s, e in failed:
        print(f"  • {s}: {e}")


if __name__ == "__main__":
    main()
