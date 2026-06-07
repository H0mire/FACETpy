"""Group-level (multi-subject) evaluation.

Batches ``evaluate_subject.run_evaluation`` over every ``.mat`` file in the
dataset directory and aggregates the per-subject outputs into a single group
table + comparison figures.  All per-subject logic lives in
``evaluate_subject`` / ``plots``; this module only orchestrates.

Outputs (under ``results/group/``):
  group_summary.csv
  fig_group_acceptance.png

Usage:
    python -m facet.Epilepsy.evaluation.evaluate_group
    python -m facet.Epilepsy.evaluation.evaluate_group --skip-existing
    python -m facet.Epilepsy.evaluation.evaluate_group --aggregate-only
"""

import argparse
import glob
import os

import pandas as pd

from facet.Epilepsy.evaluation.evaluate_subject import MAT_DIR, run_evaluation
from facet.Epilepsy.evaluation.plots import plot_group_acceptance

RESULTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "results")
)
GROUP_DIR = os.path.join(RESULTS_DIR, "group")


def _list_mat_files() -> list[str]:
    return sorted(
        os.path.join(MAT_DIR, f) for f in os.listdir(MAT_DIR)
        if f.lower().endswith(".mat")
    )


# ── Batch + aggregation ─────────────────────────────────────────────────────

def run_all_subjects(skip_existing: bool = False) -> None:
    """Run ``run_evaluation`` on every .mat file.  Failures don't abort the batch."""
    mat_files = _list_mat_files()
    if not mat_files:
        print(f"No .mat files found in {MAT_DIR}")
        return

    print(f"Batch mode: {len(mat_files)} .mat file(s) in {MAT_DIR}\n")
    failed: list[tuple[str, str]] = []
    for i, path in enumerate(mat_files, 1):
        subject = os.path.splitext(os.path.basename(path))[0]
        summary_csv = os.path.join(
            RESULTS_DIR, subject, f"results_{subject}_summary.csv"
        )
        if skip_existing and os.path.isfile(summary_csv):
            print(f"\n[{i}/{len(mat_files)}] === {subject} === (skipped)")
            continue

        print(f"\n[{i}/{len(mat_files)}] === {subject} ===")
        try:
            run_evaluation(path)
        except Exception as e:  # noqa: BLE001 — keep batch going
            print(f"  ✗ {subject} failed: {e}")
            failed.append((subject, str(e)))

    print(f"\nBatch complete: {len(failed)} failed.")
    for s, e in failed:
        print(f"  • {s}: {e}")


def build_group_dataframe() -> pd.DataFrame:
    """Concatenate every per-subject summary CSV into one group table."""
    rows = []
    for csv_path in sorted(glob.glob(
        os.path.join(RESULTS_DIR, "*", "results_*_summary.csv")
    )):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        rows.append(df.iloc[0].to_dict())
    return pd.DataFrame(rows)


def run_group_evaluation(skip_existing: bool = False, aggregate_only: bool = False):
    os.makedirs(GROUP_DIR, exist_ok=True)
    if not aggregate_only:
        run_all_subjects(skip_existing=skip_existing)

    print("\n--- Aggregating per-subject results ---")
    df = build_group_dataframe()
    if df.empty:
        print("  No per-subject summaries found — nothing to aggregate.")
        return

    csv_out = os.path.join(GROUP_DIR, "group_summary.csv")
    df.to_csv(csv_out, index=False)
    print(f"  Saved {csv_out}  ({len(df)} subjects)")

    print("\n--- Generating group figures ---")
    plot_group_acceptance(
        df, os.path.join(GROUP_DIR, "fig_group_acceptance.png"))

    print(f"\n✓ Group evaluation complete.  Outputs in: {os.path.abspath(GROUP_DIR)}")


def main():
    parser = argparse.ArgumentParser(
        description="Group (multi-subject) evaluation: batch per-subject "
                    "runs + cross-subject aggregation."
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip subjects that already have a results_<subject>_summary.csv.",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip the batch run; only aggregate existing per-subject CSVs.",
    )
    args = parser.parse_args()
    run_group_evaluation(
        skip_existing=args.skip_existing,
        aggregate_only=args.aggregate_only,
    )


if __name__ == "__main__":
    main()
