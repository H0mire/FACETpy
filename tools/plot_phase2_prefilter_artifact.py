#!/usr/bin/env python3
"""Show the Phase-2 signal and FARM estimate before final 70-Hz low-pass."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from facet.core import Pipeline


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools/pipeline_demo"))
import reference_chain  # noqa: E402
EDF = ROOT / "examples/datasets/NiazyFMRI.edf"
OUT = ROOT / "output/thesis_results_by_phase/phase_2_pipeline_deployment"


def run_pipeline(steps: list, name: str):
    return Pipeline(steps, name=name).run().context


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # No final down-sampling or 70-Hz low-pass.  The shared preprocessing still
    # includes the documented 1-Hz high-pass and trigger alignment.
    raw = run_pipeline(reference_chain.preprocessing(EDF), "phase2_pre_filter_raw")
    farm = run_pipeline(reference_chain.preprocessing(EDF) + reference_chain.farm(), "phase2_pre_filter_farm")

    raw_mne = raw.get_raw()
    farm_mne = farm.get_raw()
    names = list(raw_mne.ch_names)
    channel = names.index("Fp1") if "Fp1" in names else 0
    sfreq = float(raw_mne.info["sfreq"])
    triggers = np.asarray(raw.metadata.triggers, dtype=float)
    trigger = float(triggers[np.argmin(np.abs(triggers / sfreq - 31.5))])
    start = max(0, int(trigger - .005 * sfreq))
    stop = min(raw_mne.n_times, start + int(round(.125 * sfreq)))
    time = (np.arange(start, stop) - trigger) / sfreq * 1000
    raw_seg = raw_mne.get_data(picks=[channel], start=start, stop=stop)[0] * 1e6
    farm_seg = farm_mne.get_data(picks=[channel], start=start, stop=stop)[0] * 1e6
    estimate = raw_seg - farm_seg
    limit = max(np.quantile(np.abs(raw_seg), .998), np.quantile(np.abs(estimate), .998)) * 1.08

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 4.9), sharex=True)
    axes[0].plot(time, raw_seg, color="#555555", lw=.85)
    axes[0].set_title("Phase-2 input before final low-pass", loc="left", fontweight="bold")
    axes[1].plot(time, estimate, color="#d96b27", lw=.9)
    axes[1].set_title("FARM artifact estimate (input − FARM output)", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_ylabel("Amplitude (µV)")
        ax.set_ylim(-limit, limit)
        ax.grid(alpha=.22, lw=.5)
    axes[-1].set_xlabel("Time relative to trigger (ms)")
    fig.suptitle("Phase 2: signal state before final 70-Hz low-pass", fontweight="bold", fontsize=14)
    fig.text(.5, .012, "Fp1 · trigger-aligned · 1-Hz high-pass retained · no final low-pass", ha="center", fontsize=9)
    fig.tight_layout(rect=(.04,.04,1,.93))
    path = OUT / "figure_phase2_prefilter_input_and_farm_artifact.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    main()
