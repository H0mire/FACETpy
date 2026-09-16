"""Plot the recorded proof-fit example used to illustrate gradient-artifact amplitude."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
DATA, OUT = args.dataset, args.out
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- pick the example window -------------------------------------------------
# Window 445 / channel 25 / centre epoch has a noisy peak of ~12 mV against an
# EEG peak of ~116 µV — about a 100× amplitude ratio, which is exactly the
# "two orders of magnitude" relationship referenced in §1.1 of the thesis.
# The dataset arrays are stored in volts; we scale to µV for plotting.
WINDOW = 445
CHANNEL = 25
CENTER_IX = 3  # of 7 context epochs
SFREQ = 4096.0
V_TO_UV = 1e6

with np.load(DATA, allow_pickle=True) as d:
    noisy_ctx = d["noisy_context"][WINDOW, CENTER_IX, CHANNEL] * V_TO_UV  # (512,) µV
    clean_ctx = d["clean_context"][WINDOW, CENTER_IX, CHANNEL] * V_TO_UV  # µV
    artifact_ctx = d["artifact_context"][WINDOW, CENTER_IX, CHANNEL] * V_TO_UV  # µV

n = noisy_ctx.shape[0]
t_ms = np.arange(n) / SFREQ * 1000.0  # ms

# --- compute headline amplitude numbers --------------------------------------
peak_artifact = float(np.max(np.abs(noisy_ctx)))
peak_eeg = float(np.max(np.abs(clean_ctx)))
ratio = peak_artifact / peak_eeg if peak_eeg > 0 else float("inf")

print(f"Peak noisy (artifact-dominated): {peak_artifact:8.1f} µV")
print(f"Peak AAS-derived reference:           {peak_eeg:8.1f} µV")
print(f"Amplitude ratio:                 {ratio:8.1f}x")

# --- styling -----------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)

C_ARTIFACT = "#c1272d"  # warm red for the artifact-dominated trace
C_EEG = "#1f77b4"  # cool blue for the clean EEG

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(9.5, 5.6),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.32},
)

# Common y-limit so the amplitude difference is obvious
y_max = peak_artifact * 1.10
ylim = (-y_max, y_max)

# --- top panel: noisy (artifact + EEG) ---------------------------------------
ax = axes[0]
ax.plot(t_ms, noisy_ctx, color=C_ARTIFACT, linewidth=0.8)
ax.set_ylim(ylim)
ax.set_ylabel("Amplitude (µV)")
ax.set_title(
    "Inside the scanner — EEG + gradient artifact",
    loc="left",
    fontsize=11,
    fontweight="bold",
    pad=6,
)
ax.text(
    0.985,
    0.92,
    f"peak ≈ {peak_artifact:,.0f} µV",
    transform=ax.transAxes,
    ha="right",
    va="top",
    color=C_ARTIFACT,
    fontsize=10,
    fontweight="bold",
)
ax.axhline(0, color="black", linewidth=0.4, alpha=0.6)
ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)

# --- bottom panel: AAS-corrected (EEG only) ----------------------------------
ax = axes[1]
ax.plot(t_ms, clean_ctx, color=C_EEG, linewidth=1.0)
ax.set_ylim(ylim)  # same scale on purpose
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Amplitude (µV)")
ax.set_title(
    "AAS-derived reference (same y-axis)",
    loc="left",
    fontsize=11,
    fontweight="bold",
    pad=6,
)
ax.text(
    0.985,
    0.92,
    f"peak ≈ {peak_eeg:,.0f} µV   ({ratio:,.0f}× smaller)",
    transform=ax.transAxes,
    ha="right",
    va="top",
    color=C_EEG,
    fontsize=10,
    fontweight="bold",
)
ax.axhline(0, color="black", linewidth=0.4, alpha=0.6)
ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)

# --- inset: zoom on the clean EEG so the morphology is readable --------------
inset = ax.inset_axes([0.05, 0.05, 0.40, 0.36])
inset.plot(t_ms, clean_ctx, color=C_EEG, linewidth=1.0)
y_eeg_lim = peak_eeg * 1.15
inset.set_ylim(-y_eeg_lim, y_eeg_lim)
inset.set_xlim(t_ms[0], t_ms[-1])
inset.set_title(f"zoom ×{int(ratio)}", fontsize=8, pad=2)
inset.tick_params(labelsize=7)
inset.grid(True, linewidth=0.3, alpha=0.4)
for s in ("top", "right"):
    inset.spines[s].set_visible(False)

# --- super-title and provenance footer ---------------------------------------
fig.suptitle(
    "Gradient artifact magnitude vs. EEG amplitude (same time axis)",
    fontsize=12.5,
    fontweight="bold",
    y=0.995,
)

footer = (
    f"Source: output/niazy_proof_fit_context_512/   "
    f"window {WINDOW}, channel {CHANNEL}, centre epoch ({n} samples @ {SFREQ:.0f} Hz)"
)
fig.text(0.5, 0.01, footer, ha="center", fontsize=8, color="#555555")

fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.96))
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Wrote {OUT}")
