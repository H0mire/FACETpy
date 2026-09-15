"""Render thesis comparison figures from their canonical catalogued tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from masterthesis_guide.reproduce import load_catalog, sha256


def records(catalog, result_id):
    path = ROOT / catalog["results"][result_id]["path"]
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def ranking(catalog):
    rows = records(catalog, "table_phase1_unified_holdout_ranking")
    rows.sort(key=lambda row: float(row["median_snr_improvement_db"]))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([r["models"] for r in rows], [float(r["median_snr_improvement_db"]) for r in rows])
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("Clean-signal SNR improvement (dB; higher is better)")
    ax.set_title("Phase 1: recorded unified-holdout ranking")
    return fig, ["table_phase1_unified_holdout_ranking"]


def before_after(catalog):
    rows = [r for r in records(catalog, "table_phase3_before_after") if r["best_grid_ga_rest_uv"] != "not reported"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, key, label in [
        (-0.18, "phase2_ga_rest_uv", "Phase 2"),
        (0.18, "best_grid_ga_rest_uv", "Selected Phase 3"),
    ]:
        bars = ax.bar(x + offset, [float(r[key]) for r in rows], 0.35, label=label)
        ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_xticks(x, [r["family"] for r in rows])
    ax.set_ylabel("Residual gradient artifact (µV; lower is better)")
    ax.legend()
    return fig, ["table_phase3_before_after"]


def grid(catalog, collection="table_phase3_grid_runs"):
    rows = records(catalog, collection)
    families = ["Nested GAN", "Demucs", "ViT-Spectrogram"]
    wega = collection == "table_phase3_wega_grid_runs"
    metric = "err_uv" if wega else "ga_rest_uv"

    def valid_row(row):
        return row.get("status") == "ok" and bool(row.get(metric)) and (wega or float(row.get("naht_ratio") or 0) < 2.5)

    # Preserve every recorded cell, including invalid seam outputs.
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for i, family in enumerate(families):
        family_rows = [r for r in rows if r["family"] == family]
        rates = sorted({float(r["lr"]) for r in family_rows})
        capacities = sorted({float(r["kapazitaet"]) for r in family_rows})
        valid = [r for r in family_rows if valid_row(r)]
        best = min(valid, key=lambda r: float(r[metric])) if valid else None
        for j, weight in enumerate([0, 1, 3]):
            ax = axes[i, j]
            values = np.full((len(capacities), len(rates)), np.nan)
            selected = [r for r in family_rows if float(r["si_sdr"]) == weight]
            for row in selected:
                if row.get(metric):
                    y, x = capacities.index(float(row["kapazitaet"])), rates.index(float(row["lr"]))
                    values[y, x] = float(row[metric])
            ax.imshow(values, cmap="RdYlGn_r")
            for row in selected:
                y, x = capacities.index(float(row["kapazitaet"])), rates.index(float(row["lr"]))
                value = row.get(metric)
                invalid = not valid_row(row)
                ax.text(
                    x,
                    y,
                    (f"{float(value):.2f}" if value else "missing") + ("\ninvalid" if invalid else ""),
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                if row is best:
                    from matplotlib.patches import Rectangle

                    ax.add_patch(Rectangle((x - 0.48, y - 0.48), 0.96, 0.96, fill=False, edgecolor="black", lw=2.5))
            ax.set_xticks(range(len(rates)), [f"{r:g}" for r in rates], rotation=30)
            ax.set_yticks(range(len(capacities)), [f"{v:g}" for v in capacities])
            ax.set_title(f"{family}; SI-SDR weight {weight}")
            ax.set_xlabel("Learning rate")
            ax.set_ylabel("Capacity")
    fig.suptitle(
        ("Selection-split reconstruction error" if wega else "Residual artifact")
        + " (µV); outlines mark the best valid cell per family"
    )
    return fig, [collection]


def spikes(catalog):
    rows = records(catalog, "spike_pipeline_summary")
    families = list(dict.fromkeys(r["family"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, variant in enumerate(["phase3", "spike_aware"]):
        selected = [next(r for r in rows if r["family"] == family and r["variant"] == variant) for family in families]
        for ax, key, label in [
            (axes[0], "erhalt_pct", "Passed-through peak (% of nominal 100 µV injection)"),
            (axes[1], "ga_rest_uv", "Residual artifact (µV)"),
        ]:
            bars = ax.bar(
                np.arange(len(families)) + (i - 0.5) * 0.36,
                [float(r[key]) for r in selected],
                0.35,
                label=variant.replace("_", " "),
            )
            ax.bar_label(bars, fmt="%.1f", padding=3)
            ax.set_xticks(range(len(families)), families)
            ax.set_ylabel(label)
    axes[0].legend()
    return fig, ["spike_pipeline_summary"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=["ranking", "before_after", "grid", "wega_grid", "spikes"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    catalog = load_catalog()
    if args.figure == "wega_grid":
        fig, inputs = grid(catalog, "table_phase3_wega_grid_runs")
    else:
        fig, inputs = {"ranking": ranking, "before_after": before_after, "grid": grid, "spikes": spikes}[args.figure](
            catalog
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.figure not in ("grid", "wega_grid"):
        fig.tight_layout()
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    args.out.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "figure": args.figure,
                "inputs": {
                    rid: {
                        "path": catalog["results"][rid]["path"],
                        "sha256": sha256(ROOT / catalog["results"][rid]["path"]),
                    }
                    for rid in inputs
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
