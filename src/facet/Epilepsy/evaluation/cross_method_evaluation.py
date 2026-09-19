"""Lightweight exploratory comparison of the Ebrahimzadeh and Grouiller outputs.

This module is intentionally small: it only reuses the per-subject regressors
already saved by ``evaluate_subject`` (the ``arrays_{subject}.npz`` files) and
computes a simple temporal comparison between the two methods.

These metrics are exploratory only — not definitive agreement measures. The
main evaluation focuses on validating the Ebrahimzadeh and Grouiller methods
individually.

What it computes per patient:
  * Pearson correlation between the two regressors.
  * Dice overlap of their active periods (each thresholded at its own
    90th percentile).

Output (under ``results/group/``):
  cross_method_summary.csv

Usage:
    python -m facet.Epilepsy.evaluation.cross_method_evaluation
    python -m facet.Epilepsy.evaluation.cross_method_evaluation --patient DA00100T
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
import glob
from typing import Optional

import numpy as np
import pandas as pd

# Per-subject outputs (npz) live next to evaluate_subject's out_dir, i.e.
# ``<evaluation>/results/{subject}``.
RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "results")
)
GROUP_DIR = os.path.join(RESULTS_DIR, "group")

PERCENTILE = 90.0


# ── Loading saved outputs ────────────────────────────────────────────────────

def _load_regressors(patient: str):
    """Load (regressor_ebrahimzadeh, regressor_grouiller) from the saved npz."""
    npz_path = os.path.join(RESULTS_DIR, patient, f"arrays_{patient}.npz")
    if not os.path.isfile(npz_path):
        return None, None
    with np.load(npz_path) as data:
        reg_e = np.asarray(data.get("regressor_ebrahimzadeh", []), dtype=float)
        reg_g = np.asarray(data.get("regressor_grouiller", []), dtype=float)
    reg_e = reg_e if reg_e.size else None
    reg_g = reg_g if reg_g.size else None
    return reg_e, reg_g


# ── Metrics ──────────────────────────────────────────────────────────────────

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two equal-length 1-D arrays; NaN if undefined."""
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def _active_mask(reg: np.ndarray, percentile: float) -> np.ndarray:
    """Binary active/inactive mask: True where reg >= its own percentile."""
    return reg >= np.percentile(reg, percentile)


# ── Per-patient comparison ───────────────────────────────────────────────────

def compare_patient(patient: str) -> Optional[dict]:
    """Compare the two methods for one patient. Returns a result row or None."""
    reg_e, reg_g = _load_regressors(patient)
    if reg_e is None or reg_g is None:
        print(f"  Skipped {patient} — missing one or both regressors.")
        return None

    n = min(len(reg_e), len(reg_g))
    reg_e, reg_g = reg_e[:n], reg_g[:n]

    pearson = _pearson(reg_e, reg_g)

    act_e = _active_mask(reg_e, PERCENTILE)
    act_g = _active_mask(reg_g, PERCENTILE)
    overlap = int(np.sum(act_e & act_g))
    denom = int(np.sum(act_e)) + int(np.sum(act_g))
    dice = (2.0 * overlap / denom) if denom > 0 else float("nan")

    return {
        "patient_id": patient,
        "pearson_regressor_r": pearson,
        "dice_temporal": dice,
    }


# ── Discovery + orchestration ────────────────────────────────────────────────

def _list_patients() -> list[str]:
    """Patients that have a saved arrays npz."""
    return sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(os.path.join(RESULTS_DIR, "*", "arrays_*.npz"))
    )


def run_cross_method_evaluation(patient: Optional[str] = None):
    """Compare methods for one or all patients and write the summary CSV."""
    patients = [patient] if patient else _list_patients()
    if not patients:
        print(f"No patient outputs found under {RESULTS_DIR}")
        return pd.DataFrame()

    print(f"Cross-method comparison: {len(patients)} patient(s)\n")

    rows = []
    for i, pid in enumerate(patients, 1):
        print(f"[{i}/{len(patients)}] === {pid} ===")
        row = compare_patient(pid)
        if row is not None:
            rows.append(row)
            print(f"  r={row['pearson_regressor_r']:.3f}, "
                  f"Dice={row['dice_temporal']:.3f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo patients could be compared.")
        return df

    os.makedirs(GROUP_DIR, exist_ok=True)
    csv_out = os.path.join(GROUP_DIR, "cross_method_summary.csv")
    df.to_csv(csv_out, index=False)
    print(f"\nSaved {csv_out}  ({len(df)} patient(s))")

    return df


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exploratory cross-method (Ebrahimzadeh vs Grouiller) "
                    "comparison from saved pipeline outputs."
    )
    parser.add_argument(
        "--patient", "-p", default=None,
        help="Single patient id (e.g. DA00100T). Default: all patients.",
    )
    args = parser.parse_args()

    run_cross_method_evaluation(patient=args.patient)


if __name__ == "__main__":
    main()
