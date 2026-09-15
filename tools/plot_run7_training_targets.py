#!/usr/bin/env python3
"""Visualise one actual Run-7 training example and its three signal roles."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz"
HOLDOUT = ROOT / "output/niazy_proof_fit_context_512/holdout_v1_indices.json"
OUT = ROOT / "output/thesis_results_by_phase/phase_2_pipeline_deployment"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    held_out = set(json.loads(HOLDOUT.read_text())["indices"])
    with np.load(DATASET, allow_pickle=True) as ds:
        noisy = ds["noisy_center"]
        clean = ds["clean_center"]
        artifact = ds["artifact_center"]
        sfreq = float(ds["sfreq"][0])
        names = [str(v) for v in ds["ch_names"]]

    channel = names.index("Fp1") if "Fp1" in names else 0
    train_indices = np.array([i for i in range(noisy.shape[0]) if i not in held_out])
    rms = np.sqrt(np.mean(artifact[train_indices, channel] ** 2, axis=1))
    example = int(train_indices[np.argmin(np.abs(rms - np.quantile(rms, 0.75)))])
    time = np.arange(noisy.shape[-1]) / sfreq * 1000

    signals = [
        (clean[example, channel] * 1e6, "AAS-derived clean target", "#249d68"),
        (artifact[example, channel] * 1e6, "AAS-estimated artifact target", "#d96b27"),
        (noisy[example, channel] * 1e6, "Noisy training input (clean + artifact)", "#555555"),
    ]
    limit = max(np.quantile(np.abs(s[0]), 0.998) for s in signals) * 1.08
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 6.2), sharex=True)
    for ax, (signal, label, color) in zip(axes, signals):
        ax.plot(time, signal, color=color, lw=1.05)
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.22, lw=0.5)
        ax.set_ylim(-limit, limit)
    axes[-1].set_xlabel("Time within centre epoch (ms)")
    fig.suptitle("Run 7 training example: targets and constructed input", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.012, "Fp1 · training split · one 512-sample centre epoch (125 ms)", ha="center", fontsize=9)
    fig.tight_layout(rect=(0.04, 0.04, 1, 0.95))
    path = OUT / "figure_run7_training_targets_example.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
