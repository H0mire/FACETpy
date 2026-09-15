"""Create a three-second Phase-0 legacy AAS-versus-DAE signal comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/legacy_dl_delivery/aas_vs_dae_signals.npz"
OUT = ROOT / "output/thesis_results_by_phase/phase_0_legacy_feasibility/figure_phase0_signal_comparison.png"
METRIC_SOURCE = ROOT / "output/legacy_dl_delivery/aas_vs_dae_metrics.json"
METRIC_OUT = ROOT / "output/thesis_results_by_phase/phase_0_legacy_feasibility/figure_phase0_legacy_metrics.png"
NATIVE_SOURCE = ROOT / "output/legacy_dl/legacy_native.npz"
ARTIFACT_OUT = ROOT / "output/thesis_results_by_phase/phase_0_legacy_feasibility/figure_phase0_estimated_artifact_comparison.png"


def main() -> None:
    data = np.load(SOURCE, allow_pickle=True)
    sfreq = float(data["sfreq"])
    channel_names = list(data["ch_names"])
    channel = channel_names.index("Fp1")
    start_s, stop_s = 31.0, 34.0
    start, stop = round(start_s * sfreq), round(stop_s * sfreq)
    time = np.arange(start, stop) / sfreq

    signals = [
        ("AAS-corrected signal", data["aas"][channel, start:stop], "#4E79A7"),
        ("Legacy FC-DAE-corrected signal", data["dae"][channel, start:stop], "#E15759"),
    ]
    values_uv = [signal * 1e6 for _, signal, _ in signals]
    limit = max(float(np.max(np.abs(values))) for values in values_uv)
    limit = float(np.ceil(limit / 10.0) * 10.0)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("Phase 0 — Legacy correction comparison", fontsize=18, fontweight="bold")
    for axis, (label, _, color), values in zip(axes, signals, values_uv, strict=True):
        axis.plot(time, values, color=color, linewidth=1.1)
        axis.axhline(0, color="#AAB3C5", linewidth=0.8)
        axis.set_ylabel("Amplitude (µV)")
        axis.set_title(label, loc="left", fontsize=13, fontweight="bold", color=color)
        axis.set_ylim(-limit, limit)
        axis.grid(axis="x", color="#D9DEE8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(start_s, stop_s)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")

    metrics = __import__("json").loads(METRIC_SOURCE.read_text())["results"]
    panels = [
        ("Legacy SNR", "ratio · higher is better", "SNR", 1.0),
        ("RMS", "legacy scale", "RMS", 1.0),
        ("RMS2", "ratio", "RMS2", 1.0),
        ("Median peak-to-peak", "µV · lower is better", "MEDIAN", 1e6),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), layout="constrained")
    fig.suptitle("Phase 0 — Legacy evaluation metrics", fontsize=18, fontweight="bold")
    labels, colors = ["Legacy AAS", "Legacy FC-DAE"], ["#4E79A7", "#E15759"]
    for axis, (title, unit, key, scale) in zip(axes.flat, panels, strict=True):
        values = [metrics["AAS"][key] * scale, metrics["DAE"][key] * scale]
        bars = axis.bar(labels, values, color=colors, width=0.55)
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold")
        axis.set_ylabel(unit)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9DEE8", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", labelrotation=0)
        for bar, value in zip(bars, values, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}",
                      ha="center", va="bottom", fontsize=11)
    fig.savefig(METRIC_OUT, dpi=300, bbox_inches="tight", facecolor="white")

    native = np.load(NATIVE_SOURCE, allow_pickle=True)
    noisy = native["noisy"][0, start:stop]
    artifact_signals = [
        ("Artifact estimated by AAS", noisy - native["aas"][0, start:stop], "#4E79A7"),
        ("Artifact estimated by legacy FC-DAE", noisy - native["cleaned"][0, start:stop], "#E15759"),
    ]
    values_uv = [signal * 1e6 for _, signal, _ in artifact_signals]
    limit = max(float(np.max(np.abs(values))) for values in values_uv)
    limit = float(np.ceil(limit / 1000.0) * 1000.0)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, sharey=True, layout="constrained")
    fig.suptitle("Phase 0 — Estimated gradient-artifact comparison", fontsize=18, fontweight="bold")
    for axis, (label, _, color), values in zip(axes, artifact_signals, values_uv, strict=True):
        axis.plot(time, values, color=color, linewidth=0.9)
        axis.axhline(0, color="#AAB3C5", linewidth=0.8)
        axis.set_ylabel("Amplitude (µV)")
        axis.set_title(label, loc="left", fontsize=13, fontweight="bold", color=color)
        axis.set_ylim(-limit, limit)
        axis.grid(axis="x", color="#D9DEE8", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(start_s, stop_s)
    fig.savefig(ARTIFACT_OUT, dpi=300, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
