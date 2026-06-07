"""Single-subject statistical evaluation for the Ebrahimzadeh / Grouiller pipeline.

One .mat file = one subject.

Produces:
  results/{subject}/results_{subject}_summary.csv
  results/{subject}/results_{subject}_component_detail.csv
  results/{subject}/fig_{subject}_acceptance.png
  results/{subject}/fig_{subject}_window_corr_distribution.png
  results/{subject}/fig_{subject}_lambda_ranking.png
  results/{subject}/fig_{subject}_template.png
  results/{subject}/fig_{subject}_regressor_comparison.png

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

import mne
import numpy as np
import pandas as pd

from facet.Epilepsy.pipeline import run_combined_pipeline
from facet.Epilepsy.helpers.preprocessing import prepare_eeg_data
from facet.Epilepsy.evaluation.plots import (
    plot_acceptance_summary, plot_window_corr_distribution,
    plot_lambda_ranking, plot_template, plot_regressor_comparison,
    plot_ica_topomaps, plot_ica_reproducibility, plot_grouiller_map,
)

# ── Constants ────────────────────────────────────────────────────────────────

SFREQ = 500.0
TR = 2.5
TH_RAW = 0.60
HALF_WIN_S = 0.15
MAT_DIR = os.path.join(project_root, "examples", "datasets", "MAT_Files")


# ── Subject record ──────────────────────────────────────────────────────────

@dataclass
class SubjectRecord:
    """All data collected from a single .mat file (= one subject)."""
    subject: str                              # stem, e.g. "DA00100T"
    mat_path: str
    n_spikes_annotated: int
    n_spikes_augmented: int
    best_channel: Optional[int]
    n_accepted_components: int
    accepted_indices: list = field(default_factory=list)
    template_z: Optional[np.ndarray] = None
    per_component_window_corr: dict = field(default_factory=dict)
    ica_selection_stats: dict = field(default_factory=dict)
    regressor_ebrahimzadeh: Optional[np.ndarray] = None
    regressor_grouiller: Optional[np.ndarray] = None
    epileptic_map: Optional[np.ndarray] = None
    channel_names: list = field(default_factory=list)
    detection: object = None


# ── Collection ───────────────────────────────────────────────────────────────

def run_pipeline_for_subject(mat_path: str) -> SubjectRecord:
    """Run the combined pipeline on one .mat file and return a SubjectRecord."""
    subject = os.path.splitext(os.path.basename(mat_path))[0]
    _, _, spike_sec_raw = prepare_eeg_data(mat_path, sfreq=SFREQ)

    result = run_combined_pipeline(
        mat_path=mat_path, sfreq=SFREQ, half_win_s=HALF_WIN_S,
        th_raw=TH_RAW, has_fmri=True, tr=TR, visualize=False,
    )

    detection = result.get("detection")
    grouiller = result.get("regressor_grouiller", {})

    channel_names = _eeg_channel_names(detection)

    if detection is None:
        return SubjectRecord(
            subject=subject, mat_path=mat_path,
            n_spikes_annotated=len(spike_sec_raw),
            n_spikes_augmented=0, best_channel=None, n_accepted_components=0,
        )

    return SubjectRecord(
        subject=subject,
        mat_path=mat_path,
        n_spikes_annotated=(
            len(detection.original_spike_sec)
            if detection.original_spike_sec else len(spike_sec_raw)
        ),
        n_spikes_augmented=len(detection.refined_times),
        best_channel=detection.best_channel,
        n_accepted_components=len(detection.accepted_components),
        accepted_indices=detection.accepted_components,
        template_z=detection.template_z,
        per_component_window_corr=detection.per_component_window_corr or {},
        ica_selection_stats=detection.ica_selection_stats or {},
        regressor_ebrahimzadeh=detection.regressor_ica,
        regressor_grouiller=grouiller.get("regressor_hrf"),
        epileptic_map=grouiller.get("epileptic_map"),
        channel_names=channel_names,
        detection=detection,
    )


def _eeg_channel_names(detection) -> list:
    """EEG channel names (in epileptic-map order) from the detection's raw."""
    raw = getattr(detection, "raw", None) if detection is not None else None
    if raw is None:
        return []
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    return [raw.ch_names[i] for i in eeg_picks]


def grouiller_peak_channel(rec: "SubjectRecord") -> Optional[str]:
    """Channel with the largest |value| in the epileptic map."""
    emap = rec.epileptic_map
    if emap is None or len(emap) == 0 or not rec.channel_names:
        return None
    peak_idx = int(np.argmax(np.abs(emap)))
    if peak_idx < len(rec.channel_names):
        return rec.channel_names[peak_idx]
    return None


def ebrahimzadeh_best_channel_name(rec: "SubjectRecord") -> Optional[str]:
    """Name of the Ebrahimzadeh best channel (index into raw.ch_names)."""
    det = rec.detection
    raw = getattr(det, "raw", None) if det is not None else None
    if raw is None or rec.best_channel is None:
        return None
    if 0 <= rec.best_channel < len(raw.ch_names):
        return raw.ch_names[rec.best_channel]
    return None


def grouiller_focality(rec: "SubjectRecord") -> float:
    """Focality = max(|map|) / median(|map|).  Higher = more focal."""
    emap = rec.epileptic_map
    if emap is None or len(emap) == 0:
        return np.nan
    a = np.abs(emap)
    med = float(np.median(a))
    return float(np.max(a) / med) if med > 0 else np.nan


def ic_run_frequencies(rec: "SubjectRecord") -> list:
    """Per accepted IC: (component_idx, run_count, n_runs)."""
    counts = rec.ica_selection_stats.get("component_run_counts", {})
    n_runs = int(rec.ica_selection_stats.get("n_runs", 0))
    return [(idx, int(counts.get(idx, 0)), n_runs) for idx in rec.accepted_indices]


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

    ic_freqs = ic_run_frequencies(rec)
    ic_freq_str = ";".join(f"IC{idx}:{count}/{n}" for idx, count, n in ic_freqs)

    row = {
        "subject": rec.subject,
        "mat_file": os.path.basename(rec.mat_path),
        "n_spikes_annotated": rec.n_spikes_annotated,
        "n_spikes_augmented": rec.n_spikes_augmented,
        "best_channel": rec.best_channel,
        "best_channel_name": ebrahimzadeh_best_channel_name(rec),
        "n_accepted_components": rec.n_accepted_components,
        "accepted_indices": ";".join(str(i) for i in rec.accepted_indices),
        "median_corr_at_IEDs": ";".join(median_corrs),
        "mean_lambda": ";".join(mean_lams),
        "ic_run_frequency": ic_freq_str,
        "n_ica_runs": int(rec.ica_selection_stats.get("n_runs", 0)),
        "grouiller_peak_channel": grouiller_peak_channel(rec),
        "grouiller_focality": grouiller_focality(rec),
        "template_length_samples": (
            len(rec.template_z) if rec.template_z is not None else 0
        ),
        "has_grouiller_regressor": rec.regressor_grouiller is not None,
        "has_ebrahimzadeh_regressor": rec.regressor_ebrahimzadeh is not None,
    }
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
    """Sanity checks V1–V8.  Returns list of warning strings (empty = all OK)."""
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

    # V5: Augmented >= annotated
    if rec.n_spikes_augmented < rec.n_spikes_annotated:
        errors.append(
            f"{s}: augmented ({rec.n_spikes_augmented}) "
            f"< annotated ({rec.n_spikes_annotated})"
        )

    # V6: Accepted <= 3
    if rec.n_accepted_components > 3:
        errors.append(f"{s}: {rec.n_accepted_components} accepted > 3")

    # V7: Template length
    if rec.template_z is not None:
        expected_len = int(2 * HALF_WIN_S * SFREQ)
        if len(rec.template_z) != expected_len:
            errors.append(
                f"{s}: template length {len(rec.template_z)} "
                f"!= expected {expected_len}"
            )

    # V8: Epileptic map length = 19 EEG channels
    if rec.epileptic_map is not None and len(rec.epileptic_map) != 19:
        errors.append(
            f"{s}: epileptic map length {len(rec.epileptic_map)} != 19"
        )

    return errors


# ── Main orchestrator ────────────────────────────────────────────────────────

def run_evaluation(mat_path: str):
    """Execute the full single-subject evaluation for one .mat file."""
    subject = os.path.splitext(os.path.basename(mat_path))[0]
    out_dir = os.path.join(os.path.dirname(__file__), "results", subject)
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
    df_sum.to_csv(csv1, index=False)
    print(f"  Saved {csv1}")

    df_cd = build_component_detail_dataframe(rec)
    csv2 = os.path.join(out_dir, f"results_{subject}_component_detail.csv")
    df_cd.to_csv(csv2, index=False)
    print(f"  Saved {csv2}")

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
    plot_lambda_ranking(
        rec, os.path.join(out_dir, f"fig_{subject}_lambda_ranking.png"))
    plot_template(
        rec, os.path.join(out_dir, f"fig_{subject}_template.png"))
    plot_regressor_comparison(
        rec, os.path.join(out_dir, f"fig_{subject}_regressor_comparison.png"))
    plot_ica_topomaps(
        rec, os.path.join(out_dir, f"fig_{subject}_ica_topomaps.png"))
    plot_ica_reproducibility(
        rec, os.path.join(out_dir, f"fig_{subject}_ica_reproducibility.png"))
    plot_grouiller_map(
        rec, os.path.join(out_dir, f"fig_{subject}_grouiller_map.png"))

    print(f"\n✓ Single-subject evaluation complete for {subject}.")
    print(f"  Outputs in: {os.path.abspath(out_dir)}")
    return rec


# ── CLI ──────────────────────────────────────────────────────────────────────

def _resolve_mat_path(arg: str) -> str:
    """Turn a filename or absolute path into a resolved absolute path."""
    if os.path.isabs(arg) or os.path.isfile(arg):
        return os.path.abspath(arg)
    # Treat as filename inside the default MAT_DIR
    candidate = os.path.join(MAT_DIR, arg)
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError(
        f"Cannot find '{arg}' — tried as absolute path and in {MAT_DIR}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Single-subject statistical evaluation "
                    "(1 .mat file = 1 subject)"
    )
    parser.add_argument(
        "--mat-file", "-m", default=None,
        help="Filename or full path to the .mat file to evaluate ",
    )
    args = parser.parse_args()

    file_name = "DA00103D.mat"
    if args.mat_file:
        mat_file_path = _resolve_mat_path(args.mat_file)
    else:
        mat_file_path = os.path.join(MAT_DIR, file_name)

    run_evaluation(mat_file_path)


if __name__ == "__main__":
    main()
