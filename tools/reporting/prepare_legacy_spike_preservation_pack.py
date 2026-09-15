"""Create the concise legacy spike-preservation result pack from archived outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "thesis_results_by_phase" / "phase_3_legacy_spike_preservation"
SOURCE = ROOT / "output" / "spike_test"

ARMS = [
    ("Uncorrected", "uncorrected", 36.40),
    ("FARM", "farm", 0.25),
    ("Demucs", "demucs_deployment", 3.83),
    ("Nested GAN", "nested_gan_deployment", 1.41),
    ("ViT-Spectrogram", "vit_spectrogram_deployment", 1.44),
    ("DHCT-GAN", "dhct_gan_deployment", 2.45),
]


def load_preservation() -> dict[str, dict[str, str]]:
    with (SOURCE / "spike_preservation.csv").open(newline="") as f:
        return {row["arm"]: row for row in csv.DictReader(f)}


def write_table(rows: list[dict[str, object]]) -> None:
    fields = ["Arm", "Spike preservation (%)", "Residual gradient artifact (µV)", "Relative to FARM"]
    with (OUT / "table_legacy_spike_preservation.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def summary_figure(rows: list[dict[str, object]]) -> None:
    labels = [str(r["Arm"]) for r in rows]
    preservation = [float(r["Spike preservation (%)"]) for r in rows]
    residual = [float(r["Residual gradient artifact (µV)"]) for r in rows]
    colors = ["#7f8c8d", "#2e8b57", "#3274a1", "#8e44ad", "#e67e22", "#c0392b"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    for ax, values, title, ylabel in [
        (axes[0], preservation, "Spike preservation", "Preserved amplitude relative to uncorrected chain (%)"),
        (axes[1], residual, "Residual gradient artifact", "Residual artifact (µV; lower is better)"),
    ]:
        bars = ax.bar(labels, values, color=colors, edgecolor="#263238", linewidth=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for b, value in zip(bars, values, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{value:.1f}",
                    ha="center", va="bottom", fontsize=8)
    axes[0].set_ylim(0, 118)
    axes[1].set_ylim(0, 40)
    fig.suptitle("Legacy spike-preservation screen (four injected spikes, Fp1)", fontweight="bold")
    fig.savefig(OUT / "figure_legacy_spike_preservation_summary.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def signal_figure() -> None:
    time_lo, time_hi = 29.5, 35.5
    fig, axes = plt.subplots(len(ARMS), 1, figsize=(12.5, 8.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, (label, key, residual) in zip(axes, ARMS, strict=True):
        with np.load(SOURCE / "arms" / f"{key}.npz", allow_pickle=True) as data:
            names = [str(x) for x in data["ch_names"]]
            sfreq, offset = float(data["sfreq"]), float(data["window_s"][0])
            signal = data["data"][names.index("Fp1")]
        t = offset + np.arange(signal.size) / sfreq
        mask = (t >= time_lo) & (t <= time_hi)
        ax.plot(t[mask], signal[mask], lw=0.65, color="#542788" if label in {"Uncorrected", "FARM"} else "#d95f02")
        for spike_t in (30.0, 31.5, 33.0, 34.5):
            ax.axvline(spike_t, color="#607d8b", alpha=0.20, lw=0.7)
        ax.set_ylabel(label, fontsize=9)
        ax.text(0.995, 0.80, f"Residual GA: {residual:.2f} µV", ha="right", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.grid(alpha=0.18)
    axes[0].set_title("Fp1 outputs for the four injected-spike windows", fontweight="bold")
    axes[-1].set_xlabel("Time (s)")
    fig.supylabel("Amplitude (µV)")
    fig.savefig(OUT / "figure_legacy_spike_preservation_signals.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = load_preservation()
    rows = []
    for label, key, residual in ARMS:
        r = source[key]
        rel = residual / 0.25
        rows.append({
            "Arm": label,
            "Spike preservation (%)": f"{float(r['erhalt_vs_unkorrigiert_pct']):.1f}",
            "Residual gradient artifact (µV)": f"{residual:.2f}",
            "Relative to FARM": f"{rel:.1f}×",
        })
    write_table(rows)
    summary_figure(rows)
    signal_figure()


if __name__ == "__main__":
    main()
