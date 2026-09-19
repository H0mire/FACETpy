"""Group-level (multi-subject) evaluation.

Batches ``evaluate_subject.run_evaluation`` over every ``.mat`` file in the
dataset directory and aggregates the per-subject outputs into a single group
table + comparison figures.  All per-subject logic lives in
``evaluate_subject`` / ``plots``; this module only orchestrates.

Outputs (under ``results/group/``):
  group_summary.csv
  group_component_detail.csv
  group_stats_summary.csv                     (median/IQR/range per metric)
  group_stats_counts.csv                      (augmentation/fallback counts)
  group_accepted_components_distribution.csv
  group_template_channel_distribution.csv
  fig_group_acceptance.png
  fig_group_template_channel_distribution.png

Usage:
    python -m facet.Epilepsy.evaluation.evaluate_group
    python -m facet.Epilepsy.evaluation.evaluate_group --skip-existing
    python -m facet.Epilepsy.evaluation.evaluate_group --aggregate-only
    python -m facet.Epilepsy.evaluation.evaluate_group --aggregate-only --excel
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from facet.Epilepsy.evaluation.evaluate_subject import (
    MAT_DIR, run_evaluation,
)
from facet.Epilepsy.evaluation.plots import (
    plot_group_acceptance, plot_group_template_channel_distribution,
)

# Gate used by `augment_template` (correlation_utils.py): augmentation is only
# ever *attempted* when the annotated set is smaller than this.
AUGMENTATION_ELIGIBILITY_THRESHOLD = 10

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

def run_all_subjects(skip_existing: bool = False,
                     results_dir: str = RESULTS_DIR) -> None:
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
            results_dir, subject, f"results_{subject}_summary.csv"
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


def build_group_dataframe(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    """Concatenate every per-subject summary CSV into one group table."""
    rows = []
    for csv_path in sorted(glob.glob(
        os.path.join(results_dir, "*", "results_*_summary.csv")
    )):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        rows.append(df.iloc[0].to_dict())
    return pd.DataFrame(rows)


def build_group_component_detail_dataframe(
    results_dir: str = RESULTS_DIR,
) -> pd.DataFrame:
    """Concatenate every per-subject component-detail CSV (one row per component)."""
    frames = []
    for csv_path in sorted(glob.glob(
        os.path.join(results_dir, "*", "results_*_component_detail.csv")
    )):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_group_spatial_gate_detail_dataframe(
    results_dir: str = RESULTS_DIR,
) -> pd.DataFrame:
    """Concatenate every per-subject spatial-gate detail CSV (spatial mode only).

    These files (``results_<subject>_spatial_gate_detail.csv``) are written only
    when the spatial TCCC gate is enabled, so this returns an empty frame for
    baseline runs.
    """
    frames = []
    for csv_path in sorted(glob.glob(
        os.path.join(results_dir, "*", "results_*_spatial_gate_detail.csv")
    )):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _median_iqr_range(s: pd.Series) -> dict:
    """median, IQR (Q1/Q3), and range (min-max) for a numeric series."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {"median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan,
                "min": np.nan, "max": np.nan, "n": 0}
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    return {
        "median": float(s.median()), "q1": q1, "q3": q3, "iqr": q3 - q1,
        "min": float(s.min()), "max": float(s.max()), "n": int(s.size),
    }


def compute_group_stats(df: pd.DataFrame) -> dict:
    """Cross-subject descriptive stats derived from ``group_summary``.

    Returns a dict with:
      - ``numeric_summary``: DataFrame (rows = metric, cols = median/q1/q3/
        iqr/min/max/n) for annotated spikes, augmented spikes, accepted
        components, the per-subject delta (augmented - annotated) across ALL
        subjects, and that same delta computed ONLY over subjects where
        augmentation actually triggered (delta > 0) — the number needed for
        "added a median of X additional events" (including the untriggered
        zeros would understate it).
      - ``counts_summary``: DataFrame (single row) with the subject counts/
        percentages needed for the augmentation and TCCC-acceptance/fallback
        story: how many subjects were eligible for / triggered augmentation,
        and how many were accepted directly vs. via fallback.
      - ``accepted_components_distribution``: Series, count of subjects per
        ``n_accepted_components`` value (0, 1, 2, 3, ...).
      - ``template_channel_distribution``: Series, count of subjects per
        selected IED-template channel.
    """
    out: dict = {}
    n = len(df)

    if df.empty or "n_spikes_annotated" not in df.columns:
        out["numeric_summary"] = pd.DataFrame()
        out["counts_summary"] = pd.DataFrame()
        out["accepted_components_distribution"] = pd.Series(dtype=int)
        out["template_channel_distribution"] = pd.Series(dtype=int)
        return out

    annotated = pd.to_numeric(df["n_spikes_annotated"], errors="coerce")
    # ``n_spikes_augmented`` is the count of spikes ADDED by template
    # augmentation (0 when augmentation didn't trigger), not the total set.
    added = pd.to_numeric(df["n_spikes_augmented"], errors="coerce")
    triggered = added > 0

    summary_rows = {
        "n_spikes_annotated": _median_iqr_range(annotated),
        "n_spikes_added_by_augmentation": _median_iqr_range(added),
        "n_spikes_added_by_augmentation_among_triggered": _median_iqr_range(
            added[triggered]
        ),
    }
    if "n_accepted_components" in df.columns:
        summary_rows["n_accepted_components"] = _median_iqr_range(
            pd.to_numeric(df["n_accepted_components"], errors="coerce")
        )
    out["numeric_summary"] = pd.DataFrame(summary_rows).T

    n_eligible = int((annotated < AUGMENTATION_ELIGIBILITY_THRESHOLD).sum())
    n_triggered = int(triggered.sum())
    counts = {
        "n_subjects_total": n,
        "n_subjects_eligible_for_augmentation": n_eligible,
        "pct_subjects_eligible_for_augmentation": 100 * n_eligible / n,
        "n_subjects_augmentation_triggered": n_triggered,
        "pct_subjects_augmentation_triggered": 100 * n_triggered / n,
        "total_additional_spikes_added": int(added.clip(lower=0).sum()),
    }

    if "fallback_used" in df.columns:
        fallback = df["fallback_used"].astype(bool)
        n_fallback = int(fallback.sum())
        n_normal = n - n_fallback
        counts["n_subjects_normal_acceptance"] = n_normal
        counts["pct_subjects_normal_acceptance"] = 100 * n_normal / n
        counts["n_subjects_fallback"] = n_fallback
        counts["pct_subjects_fallback"] = 100 * n_fallback / n

    out["counts_summary"] = pd.DataFrame([counts])

    if "n_accepted_components" in df.columns:
        out["accepted_components_distribution"] = (
            pd.to_numeric(df["n_accepted_components"], errors="coerce")
            .value_counts().sort_index()
        )
    else:
        out["accepted_components_distribution"] = pd.Series(dtype=int)

    if "template_channel" in df.columns:
        out["template_channel_distribution"] = (
            df["template_channel"].value_counts(dropna=False)
        )
    else:
        out["template_channel_distribution"] = pd.Series(dtype=int)

    return out


def _write_group_excel(path: str, sheets: dict[str, pd.DataFrame]) -> None:
    """Write all non-empty group tables into one multi-sheet workbook.

    Excel sheet names are capped at 31 chars; longer keys are truncated.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, (df, index) in sheets.items():
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=name[:31], index=index)
    print(f"  Saved {path}")


def run_group_evaluation(skip_existing: bool = False, aggregate_only: bool = False,
                         excel: bool = False):
    results_dir = RESULTS_DIR
    group_dir = os.path.join(results_dir, "group")
    os.makedirs(group_dir, exist_ok=True)
    print(f"Source results dir: {os.path.abspath(results_dir)}")

    if not aggregate_only:
        run_all_subjects(skip_existing=skip_existing, results_dir=results_dir)

    print("\n--- Aggregating per-subject results ---")
    df = build_group_dataframe(results_dir)
    if df.empty:
        print("  No per-subject summaries found — nothing to aggregate.")
        return

    csv_out = os.path.join(group_dir, "group_summary.csv")
    df.to_csv(csv_out, index=False)
    print(f"  Saved {csv_out}  ({len(df)} subjects)")

    df_cd = build_group_component_detail_dataframe(results_dir)
    if df_cd.empty:
        print("  No per-subject component-detail CSVs found — skipping.")
    else:
        cd_out = os.path.join(group_dir, "group_component_detail.csv")
        df_cd.to_csv(cd_out, index=False)
        print(f"  Saved {cd_out}  ({len(df_cd)} components "
              f"across {df_cd['subject'].nunique()} subjects)")

    # Per-candidate spatial-validation detail.
    df_sg = build_group_spatial_gate_detail_dataframe(results_dir)
    if not df_sg.empty:
        sg_out = os.path.join(group_dir, "group_spatial_gate_detail.csv")
        df_sg.to_csv(sg_out, index=False)
        print(f"  Saved {sg_out}  ({len(df_sg)} candidate rows "
              f"across {df_sg['subject'].nunique()} subjects)")

    print("\n--- Generating group figures ---")
    plot_group_acceptance(
        df, os.path.join(group_dir, "fig_group_acceptance.png"))
    plot_group_template_channel_distribution(
        df, os.path.join(group_dir, "fig_group_template_channel_distribution.png"))

    print("\n--- Group descriptive statistics ---")
    stats = compute_group_stats(df)

    ns = stats["numeric_summary"]
    if not ns.empty:
        stats_out = os.path.join(group_dir, "group_stats_summary.csv")
        ns.to_csv(stats_out)
        print(f"  Saved {stats_out}")

    cs = stats["counts_summary"]
    if not cs.empty:
        counts_out = os.path.join(group_dir, "group_stats_counts.csv")
        cs.to_csv(counts_out, index=False)
        print(f"  Saved {counts_out}")

    acd = stats["accepted_components_distribution"]
    if not acd.empty:
        acd_out = os.path.join(group_dir, "group_accepted_components_distribution.csv")
        acd.rename("count").rename_axis("n_accepted_components").to_csv(acd_out)
        print(f"  Saved {acd_out}")

    tcd = stats["template_channel_distribution"]
    if not tcd.empty:
        tcd_out = os.path.join(group_dir, "group_template_channel_distribution.csv")
        tcd.rename("count").rename_axis("template_channel").to_csv(tcd_out)
        print(f"  Saved {tcd_out}")

    if excel:
        print("\n--- Writing group Excel workbook ---")
        xlsx_out = os.path.join(group_dir, "group_summary.xlsx")
        # Each entry: sheet name -> (dataframe, write_index).
        sheets = {
            "summary": (df, False),
            "component_detail": (df_cd, False),
            "spatial_gate_detail": (df_sg, False),
            "stats_summary": (ns.reset_index().rename(
                columns={"index": "metric"}), False),
            "stats_counts": (cs, False),
            "accepted_comp_dist": (
                acd.rename("count").rename_axis("n_accepted_components")
                .reset_index() if not acd.empty else pd.DataFrame(), False),
            "template_channel_dist": (
                tcd.rename("count").rename_axis("template_channel")
                .reset_index() if not tcd.empty else pd.DataFrame(), False),
        }
        _write_group_excel(xlsx_out, sheets)

    print(f"\n✓ Group evaluation complete.  Outputs in: {os.path.abspath(group_dir)}")


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
    parser.add_argument(
        "--excel", action="store_true",
        help="Also write a single multi-sheet group_summary.xlsx workbook.",
    )
    args = parser.parse_args()
    run_group_evaluation(
        skip_existing=args.skip_existing,
        aggregate_only=args.aggregate_only,
        excel=args.excel,
    )


if __name__ == "__main__":
    main()
