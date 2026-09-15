"""Build the thesis-facing checkpoint comparison on the generated Niazy spike EDF."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "output" / "phase3_spike_aware_comparison"
OUT = ROOT / "output" / "thesis_results_by_phase" / "phase_3_spike_checkpoint_comparison"
REF = BASE / "reference_pipeline"
RETRIEVED = BASE / "retrieved"


def csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return next(csv.DictReader(f))


def reference_rows() -> list[dict[str, object]]:
    spikes = {r["arm"]: r for r in csv.DictReader((REF / "spike_preservation.csv").open())}
    residuals = {r["arm"]: r for r in csv.DictReader((REF / "residual_metrics.csv").open())}
    return [
        {"model": "Uncorrected chain reference", "family": "Reference", "variant": "Uncorrected",
         "key": "uncorrected", "preservation": float(spikes["uncorrected"]["erhalt_vs_unkorrigiert_pct"]),
         "residual": float(residuals["uncorrected"]["ga_rest_uv"])},
        {"model": "FARM", "family": "Reference", "variant": "FARM",
         "key": "farm", "preservation": float(spikes["farm"]["erhalt_vs_unkorrigiert_pct"]),
         "residual": float(residuals["farm"]["ga_rest_uv"])},
    ]


def checkpoint_rows() -> list[dict[str, object]]:
    variants = [("phase3", "Original Phase-3"), ("spike_aware", "Spike-aware retraining")]
    labels = {"nested_gan": ("Nested GAN", "GAN"), "demucs": ("Demucs", "Audio"), "vit_spectrogram": ("ViT-Spectrogram", "Vision")}
    rows = []
    for slug, (model, family) in labels.items():
        root = RETRIEVED / slug / "output" / "phase3_spike_aware_comparison" / slug / "pipeline"
        for variant, display in variants:
            spike = csv_row(root / variant / "spike_preservation.csv")
            residual = csv_row(root / variant / "residual_metrics.csv")
            rows.append({"model": model, "family": family, "variant": display, "key": f"{slug}_{variant}",
                         "preservation": float(spike["erhalt_pct"]), "residual": float(residual["ga_rest_uv"])})
    return rows


def write_table(rows: list[dict[str, object]]) -> None:
    fields = ["Model", "Checkpoint / reference", "Spike preservation (%)", "Residual gradient artifact (µV)", "Relative to FARM"]
    with (OUT / "table_spike_checkpoint_comparison.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({"Model": r["model"], "Checkpoint / reference": r["variant"],
                        "Spike preservation (%)": f"{r['preservation']:.1f}",
                        "Residual gradient artifact (µV)": f"{r['residual']:.3f}",
                        "Relative to FARM": f"{r['residual'] / 0.248:.1f}×"})


def summary_figure(rows: list[dict[str, object]]) -> None:
    colors = {"Reference": "#66757f", "GAN": "#8e44ad", "Audio": "#2874a6", "Vision": "#e67e22"}
    hatches = {"Uncorrected": "", "FARM": "", "Original Phase-3": "", "Spike-aware retraining": "//"}
    labels = [r["model"] if r["family"] == "Reference" else f"{r['model']}\n{r['variant'].replace(' retraining', '')}" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.3), constrained_layout=True)
    for ax, field, title, ylabel in [
        (axes[0], "preservation", "Response to four injected 100 µV spikes", "Mean passed-through peak (µV)"),
        (axes[1], "residual", "No-injection pipeline", "Residual gradient artifact (µV; lower is better)"),
    ]:
        bars = ax.bar(np.arange(len(rows)), [r[field] for r in rows],
                      color=[colors[r["family"]] for r in rows], edgecolor="#263238", linewidth=0.65,
                      hatch=[hatches[r["variant"]] for r in rows])
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(np.arange(len(rows)), labels, rotation=35, ha="right", fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for b, r in zip(bars, rows, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r[field]:.1f}",
                    ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, 118)
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.15, 60)
    axes[1].text(0.02, 0.04, "Log scale to retain visibility of FARM and model values.",
                 transform=axes[1].transAxes, fontsize=8, color="#455a64")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor="#8e44ad", edgecolor="#263238", label="Original Phase-3"),
                        plt.Rectangle((0, 0), 1, 1, facecolor="#8e44ad", edgecolor="#263238", hatch="//", label="Spike-aware retraining"),
                        plt.Rectangle((0, 0), 1, 1, facecolor="#66757f", edgecolor="#263238", label="Reference")],
               ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.savefig(OUT / "figure_spike_checkpoint_comparison_summary.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def signal_path(row: dict[str, object], injected: bool) -> Path:
    state = "with" if injected else "without"
    if row["family"] == "Reference":
        return REF / state / f"{row['key']}.npz"
    slug, variant = str(row["key"]).rsplit("_", 1)
    variant = "spike_aware" if variant == "aware" else variant
    # All model keys contain one of these exact suffixes.
    for tag in ("phase3", "spike_aware"):
        if str(row["key"]).endswith(tag):
            slug = str(row["key"])[:-(len(tag) + 1)]
            variant = tag
            break
    return RETRIEVED / slug / "output" / "phase3_spike_aware_comparison" / slug / "pipeline" / variant / state / f"{slug}_tuned.npz"


def load_fp1(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as b:
        names = [str(x) for x in b["ch_names"]]
        t = float(b["window_s"][0]) + np.arange(b["data"].shape[1]) / float(b["sfreq"])
        return t, b["data"][names.index("Fp1")].astype(float)


def waveform_figure(rows: list[dict[str, object]]) -> None:
    # Difference of injected and non-injected outputs isolates the spike response.
    groups = [r for r in rows if r["family"] in {"GAN", "Audio", "Vision"}]
    colors = {"Original Phase-3": "#455a64", "Spike-aware retraining": "#00838f", "FARM": "#2e8b57"}
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 7.1), sharex=True, constrained_layout=True)
    farm = next(r for r in rows if r["variant"] == "FARM")
    ft, fy = load_fp1(signal_path(farm, True)); _, f0 = load_fp1(signal_path(farm, False))
    mask = (ft >= 29.78) & (ft <= 30.22)
    for ax, family in zip(axes, ("GAN", "Audio", "Vision"), strict=True):
        for r in [x for x in groups if x["family"] == family]:
            t, injected = load_fp1(signal_path(r, True)); _, baseline = load_fp1(signal_path(r, False))
            ax.plot((t[mask] - 30) * 1000, (injected[mask] - baseline[mask]), lw=1.35,
                    color=colors[r["variant"]], label=r["variant"])
        ax.plot((ft[mask] - 30) * 1000, fy[mask] - f0[mask], lw=1.0, ls="--", color=colors["FARM"], label="FARM")
        ax.axvline(0, color="#607d8b", lw=0.7, alpha=0.5)
        ax.set_title(next(r["model"] for r in groups if r["family"] == family), loc="left", fontweight="bold")
        ax.grid(alpha=0.22)
        ax.set_ylabel("Injected response (µV)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time from 30 s injection peak (ms)")
    fig.suptitle("Matched response to the generated Niazy spike injection (Fp1)", fontweight="bold")
    fig.savefig(OUT / "figure_spike_checkpoint_comparison_waveforms.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def input_output_figure(rows: list[dict[str, object]]) -> None:
    """Show the actual injected input and each corrected output, not a difference."""
    chosen = [next(r for r in rows if r["variant"] == "Uncorrected"),
              next(r for r in rows if r["variant"] == "FARM")]
    for family in ("GAN", "Audio", "Vision"):
        chosen.extend(r for r in rows if r["family"] == family)
    colors = {"Uncorrected": "#6c7a89", "FARM": "#2e8b57", "Original Phase-3": "#455a64",
              "Spike-aware retraining": "#00838f"}
    fig, axes = plt.subplots(len(chosen), 1, figsize=(12.2, 10.2), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, row in zip(axes, chosen, strict=True):
        t, signal = load_fp1(signal_path(row, True))
        mask = (t >= 30.0) & (t <= 35.0)
        ax.plot(t[mask], signal[mask], lw=0.9, color=colors[row["variant"]])
        for spike_t in (30.0, 31.5, 33.0, 34.5):
            ax.axvline(spike_t, color="#607d8b", lw=0.8, alpha=0.45)
        ax.set_ylabel(row["model"] if row["family"] == "Reference" else f"{row['model']}\n{row['variant'].replace(' retraining', '')}", fontsize=8.5)
        ax.grid(alpha=0.2)
    axes[0].set_title("Generated Niazy input with a 100 µV spike and matched corrected outputs (Fp1)",
                      fontweight="bold")
    axes[-1].set_xlabel("Time (s); vertical lines mark the four injected spikes")
    fig.supylabel("Amplitude (µV); shared scale")
    fig.savefig(OUT / "figure_spike_checkpoint_comparison_input_outputs.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = reference_rows() + checkpoint_rows()
    write_table(rows)
    summary_figure(rows)
    waveform_figure(rows)
    input_output_figure(rows)


if __name__ == "__main__":
    main()
