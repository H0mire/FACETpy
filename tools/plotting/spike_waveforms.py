"""Summarize the fixed, completed checkpoint and paired-pipeline comparisons."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
base = args.data_root / "output/phase3_spike_aware_comparison"
families = {"nested_gan": "Nested GAN", "demucs": "Demucs", "vit_spectrogram": "ViT"}
wavefig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=True)
times = [30.0, 31.5, 33.0, 34.5]
for row, (family, label) in enumerate(families.items()):
    location = base / "retrieved" / family
    verified = json.loads((location / "VERIFIED.json").read_text())
    result = location / verified["comparison_dir"]
    pipeline = result / "pipeline"
    for variant, color in [("phase3", "#687888"), ("spike_aware", "#087f8c")]:
        with (
            np.load(pipeline / variant / "with" / f"{family}_tuned.npz", allow_pickle=True) as a,
            np.load(pipeline / variant / "without" / f"{family}_tuned.npz", allow_pickle=True) as b,
        ):
            assert np.array_equal(a["ch_names"], b["ch_names"]) and np.array_equal(a["window_s"], b["window_s"])
            assert float(a["sfreq"]) == float(b["sfreq"])
            ch = list(a["ch_names"]).index("Fp1")
            difference = a["data"][ch] - b["data"][ch]
            sf = float(a["sfreq"])
            offset = float(a["window_s"][0])
        for col, centre in enumerate(times):
            lo, hi = round((centre - 0.15 - offset) * sf), round((centre + 0.15 - offset) * sf)
            ax = axes[row, col]
            ax.plot(
                (np.arange(lo, hi) / sf + offset - centre) * 1000,
                difference[lo:hi],
                color=color,
                lw=1.2,
                label=variant.replace("_", " "),
            )
            ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(f"Injected spike at {centre:g} s")
            if col == 0:
                ax.set_ylabel(label + "\nPassed-through change (µV)")
            if row == 2:
                ax.set_xlabel("Time from injection peak (ms)")
for name, label, color, linestyle in [
    ("uncorrected", "Injection reference", "#444444", ":"),
    ("farm", "FARM", "#249d68", "--"),
]:
    with (
        np.load(base / "reference_pipeline/with" / f"{name}.npz", allow_pickle=True) as a,
        np.load(base / "reference_pipeline/without" / f"{name}.npz", allow_pickle=True) as b,
    ):
        if (
            not np.array_equal(a["ch_names"], b["ch_names"])
            or not np.array_equal(a["window_s"], b["window_s"])
            or float(a["sfreq"]) != float(b["sfreq"])
        ):
            raise ValueError(f"Mismatched baseline arms: {name}")
        ch = list(a["ch_names"]).index("Fp1")
        difference = a["data"][ch] - b["data"][ch]
        sf = float(a["sfreq"])
        offset = float(a["window_s"][0])
    for row in range(3):
        for col, centre in enumerate(times):
            lo, hi = round((centre - 0.15 - offset) * sf), round((centre + 0.15 - offset) * sf)
            axes[row, col].plot(
                (np.arange(lo, hi) / sf + offset - centre) * 1000,
                difference[lo:hi],
                color=color,
                ls=linestyle,
                lw=1,
                label=label,
            )
axes[0, 0].legend(fontsize=7)
wavefig.suptitle(
    "Matched pipeline response to injected spikes — Fp1\nDifference between injected and non-injected recordings; original Phase-3 versus retrained",
    fontsize=12,
)
wavefig.tight_layout(rect=[0, 0, 1, 0.94])
args.out.parent.mkdir(parents=True, exist_ok=True)
wavefig.savefig(args.out, dpi=180)
plt.close(wavefig)
