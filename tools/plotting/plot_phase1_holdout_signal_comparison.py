#!/usr/bin/env python3
"""Create a like-for-like Phase-1 corrected-signal comparison.

Every panel uses the same held-out Fp1 epoch.  The gray trace is the noisy
input, the dashed green trace is the AAS-derived target, and the coloured
trace is the corrected signal reconstructed from each model's exported
artifact prediction.  The target is a proof-of-fit surrogate, not an
artifact-free ground truth signal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from masterthesis_guide.reproduce import load_catalog

DATASET = ROOT / "output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"
EVAL = ROOT / "output/model_evaluations"
OUT = ROOT / "output/thesis_results_by_phase/phase_1_unified_holdout"

MODELS = [
    ("demucs", "Demucs"),
    ("conv_tasnet", "Conv-TasNet"),
    ("cascaded_context_dae", "Cascaded Context DAE"),
    ("sepformer", "SepFormer"),
    ("cascaded_dae", "Cascaded DAE"),
    ("nested_gan", "Nested GAN"),
    ("denoise_mamba", "DenoiseMamba"),
    ("ic_unet", "IC-U-Net"),
    ("st_gnn", "ST-GNN"),
    ("vit_spectrogram", "Vision Transformer"),
    ("dpae", "DPAE"),
    ("d4pm", "D4PM"),
    ("dhct_gan_v2", "DHCT-GAN v2"),
    ("dhct_gan", "DHCT-GAN"),
]


def metric(model_id: str) -> float:
    catalog = load_catalog()
    paths = catalog["experiments"]["holdout_" + model_id]["evidence"]
    path = next(ROOT / p for p in paths if p.endswith("/metrics.json"))
    payload = json.loads(path.read_text())
    flat = payload.get("flat_metrics", payload)
    return float(flat["unified_holdout.clean_snr_improvement_db"])


def main() -> None:
    import argparse

    global DATASET, EVAL, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        help="Separate root containing original output/model_evaluations prediction arrays",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    DATASET = args.data_root / "output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"
    EVAL = (args.prediction_root or args.data_root) / "output/model_evaluations"
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    indices = np.array(json.loads((ROOT / load_catalog()["datasets"]["proof_fit"]["split"]).read_text())["indices"])
    with np.load(DATASET, allow_pickle=True) as ds:
        noisy = ds["noisy_center"][indices]
        clean = ds["clean_center"][indices]
        artifact = ds["artifact_center"][indices]
        sfreq = float(ds["sfreq"][0])
        names = [str(x) for x in ds["ch_names"]]

    channel = names.index("Fp1") if "Fp1" in names else 0
    # Strong but non-outlying artifact example: 75th percentile amplitude.
    rms = np.sqrt(np.mean(artifact[:, channel] ** 2, axis=1))
    example = int(np.argmin(np.abs(rms - np.quantile(rms, 0.75))))
    time_ms = np.arange(noisy.shape[-1]) / sfreq * 1000
    scale = np.quantile(np.abs(np.concatenate([noisy[example, channel], clean[example, channel]])), 0.995) * 1.12

    fig, axes = plt.subplots(4, 4, figsize=(13.2, 9.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (model_id, label) in zip(axes, MODELS):
        path = EVAL / model_id / "holdout_v1" / "predicted_artifact.npy"
        from masterthesis_guide.reproduce import sha256

        record = next(r for r in load_catalog()["external_predictions"] if r["experiment"] == "holdout_" + model_id)
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Original prediction bytes do not match: {path}")
        pred = np.load(path, mmap_mode="r")
        corrected = noisy[example, channel] - pred[example, channel]
        ax.plot(time_ms, noisy[example, channel] * 1e6, color="#a9a9a9", lw=0.65, label="Noisy input")
        ax.plot(time_ms, clean[example, channel] * 1e6, color="#249d68", lw=0.75, ls="--", label="AAS-derived target")
        ax.plot(time_ms, corrected * 1e6, color="#2d6fae", lw=0.9, label="Corrected signal")
        ax.set_title(f"{label}\nSNR gain {metric(model_id):+.2f} dB", fontsize=8.7, fontweight="bold")
        ax.grid(alpha=0.18, lw=0.45)
        ax.set_ylim(-scale * 1e6, scale * 1e6)

    axes[-2].axis("off")
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle(
        "Phase 1: corrected signals from a common unified-holdout epoch", fontsize=14, fontweight="bold", y=0.995
    )
    fig.text(0.5, 0.065, "Time (ms)", ha="center", fontsize=10)
    fig.text(0.012, 0.5, "Amplitude (µV)", va="center", rotation="vertical", fontsize=10)
    fig.tight_layout(rect=(0.025, 0.08, 1, 0.96))
    output = OUT / "figure_phase1_corrected_signals_all_models.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
