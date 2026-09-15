#!/usr/bin/env python3
"""Plot Phase-2 deployment outputs on one shared real-data segment.

This is a visual pipeline comparison, not a clean-reference accuracy plot:
the dashed FARM trace is the practical correction baseline on the same EEG
recording and the panel titles report residual gradient-artifact amplitude over
the full Phase-2 evaluation interval.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARMS = ROOT / "output/pipeline_demo/deployment_first/arms"
TABLE = ROOT / "output/thesis_results_by_phase/phase_2_pipeline_deployment/table_phase2_pipeline_results.csv"
OUT = ROOT / "output/thesis_results_by_phase/phase_2_pipeline_deployment"

MODELS = [
    ("st_gnn_deployment", "ST-GNN"),
    ("nested_gan_deployment", "Nested GAN"),
    ("vit_spectrogram_deployment", "Vision Transformer"),
    ("sepformer_deployment", "SepFormer"),
    ("dhct_gan_deployment", "DHCT-GAN"),
    ("dhct_gan_v2_deployment", "DHCT-GAN v2"),
    ("cascaded_context_dae_deployment", "Cascaded Context DAE"),
    ("demucs_deployment", "Demucs"),
    ("conv_tasnet_deployment", "Conv-TasNet"),
    ("denoise_mamba_deployment", "DenoiseMamba"),
    ("dpae_deployment", "DPAE"),
    ("cascaded_dae_deployment", "Cascaded DAE"),
    ("ic_unet_deployment", "IC-U-Net"),
]


def load_arm(name: str) -> tuple[np.ndarray, list[str], float, tuple[float, float]]:
    with np.load(ARMS / f"{name}.npz", allow_pickle=True) as packet:
        return (
            packet["data"],
            [str(x) for x in packet["ch_names"]],
            float(packet["sfreq"]),
            tuple(float(x) for x in packet["window_s"]),
        )


def residuals() -> dict[str, float]:
    with TABLE.open(newline="") as handle:
        return {row["arm"]: float(row["ga_rest_uv"]) for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-matched", action="store_true",
                        help="Plot one 125-ms trigger-aligned epoch, matching the Phase-1 time span.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw, names, sfreq, window = load_arm("uncorrected")
    farm, _, _, _ = load_arm("farm")
    channel = names.index("Fp1") if "Fp1" in names else 0
    if args.epoch_matched:
        # Phase 1 visualises 512 samples at 4096 Hz, i.e. 125 ms.  Select one
        # Phase-2 trigger and use exactly that display duration here.
        with np.load(ARMS / "uncorrected.npz", allow_pickle=True) as packet:
            triggers = packet["triggers"] / sfreq + window[0]
        trigger_time = float(triggers[np.argmin(np.abs(triggers - 31.5))])
        absolute_start = trigger_time - 0.005
        absolute_stop = absolute_start + 512 / 4096
    else:
        # Same three-second segment for every arm in the artifact-present interval.
        absolute_start, absolute_stop = 31.0, 34.0
    start = int((absolute_start - window[0]) * sfreq)
    stop = int((absolute_stop - window[0]) * sfreq)
    time = np.arange(start, stop) / sfreq + window[0]
    x = (time - absolute_start) * 1000 if args.epoch_matched else time
    # Pipeline-demo arrays are already stored in microvolts.
    raw_seg = raw[channel, start:stop]
    farm_seg = farm[channel, start:stop]
    limit = np.quantile(np.abs(raw_seg), 0.997) * 1.10
    ga_residual = residuals()

    fig, axes = plt.subplots(4, 4, figsize=(13.2, 9.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (arm, label) in zip(axes, MODELS):
        data, _, _, _ = load_arm(arm)
        corrected = data[channel, start:stop]
        ax.plot(x, raw_seg, color="#b0b0b0", lw=0.55, label="Uncorrected input")
        ax.plot(x, farm_seg, color="#249d68", lw=0.75, ls="--", label="FARM reference")
        ax.plot(x, corrected, color="#2d6fae", lw=0.85, label="Pipeline output")
        value = ga_residual.get(arm, float("nan"))
        ax.set_title(f"{label}\nGA residual {value:.2f} µV", fontsize=8.7, fontweight="bold")
        ax.set_ylim(-limit, limit)
        ax.grid(alpha=0.18, lw=0.45)

    for ax in axes[len(MODELS):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.015))
    figure_title = ("Phase 2: pipeline-corrected signals on one trigger-aligned epoch"
                    if args.epoch_matched else
                    "Phase 2: pipeline-corrected signals on a common real-data segment")
    fig.suptitle(figure_title, fontsize=14,
                 fontweight="bold", y=0.995)
    fig.text(0.5, 0.065, "Time relative to trigger (ms)" if args.epoch_matched else "Time (s)",
             ha="center", fontsize=10)
    fig.text(0.012, 0.5, "Amplitude (µV)", va="center", rotation="vertical", fontsize=10)
    fig.tight_layout(rect=(0.025, 0.08, 1, 0.96))
    suffix = "epoch_matched" if args.epoch_matched else "all_models"
    output = OUT / f"figure_phase2_corrected_signals_{suffix}.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
