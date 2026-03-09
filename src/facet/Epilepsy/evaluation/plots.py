"""Evaluation figure generation (F1–F5) for single-subject analysis.

Each function takes a SubjectRecord and an output path,
produces one figure, and saves it.  No computation logic lives here.
"""

import os
import sys

# --- Setup Python Path ---
project_root = r"D:\Medical Engineering and Analytics\Project\FACETpy"
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.append(src_path)
# -------------------------

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Constants (must stay in sync with evaluate_subject.py).
TH_RAW = 0.60
TR = 2.5
HALF_WIN_S = 0.15


def plot_acceptance_summary(rec, out_path: str):
    """F1: Summary bar chart — annotated spikes, augmented spikes, accepted components."""
    labels = ["Annotated\nspikes", "Augmented\nspikes", "Accepted\ncomponents"]
    values = [rec.n_spikes_annotated, rec.n_spikes_augmented, rec.n_accepted_components]
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_title(f"Pipeline Yield — {rec.subject}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_window_corr_distribution(rec, out_path: str):
    """F2: Box plot — per-window max|r| distribution for each accepted component."""
    if not rec.accepted_indices:
        print("  Skipped window_corr_distribution — no accepted components.")
        return

    labels, data = [], []
    for idx in rec.accepted_indices:
        wc = rec.per_component_window_corr.get(idx, [])
        if wc:
            data.append(wc)
            labels.append(f"IC {idx}")

    if not data:
        print("  Skipped window_corr_distribution — no correlation data.")
        return

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.5), 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4c72b0")
        patch.set_alpha(0.6)
    ax.axhline(TH_RAW, color="red", ls="--", lw=1, label=f"Threshold ({TH_RAW})")
    ax.set_ylabel("max |r| per IED window")
    ax.set_title(f"Per-Window Correlation — {rec.subject}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_lambda_ranking(rec, out_path: str):
    """F3: Bar chart — avg λ for each accepted component."""
    lambdas = rec.ica_selection_stats.get("component_lambdas", {})

    if not rec.accepted_indices:
        print("  Skipped lambda_ranking — no accepted components.")
        return

    comp_labels, vals = [], []
    for idx in rec.accepted_indices:
        lam_list = lambdas.get(idx, [])
        vals.append(float(np.mean(lam_list)) if lam_list else 0.0)
        comp_labels.append(f"IC {idx}")

    colors = ["#4c72b0", "#55a868", "#c44e52"]

    fig, ax = plt.subplots(figsize=(max(5, len(comp_labels) * 1.5), 5))
    bars = ax.bar(comp_labels, vals,
                  color=[colors[i % len(colors)] for i in range(len(vals))],
                  width=0.45)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Average λ (mixing weight L2 norm)")
    ax.set_title(f"ICA Component λ Ranking — {rec.subject}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_template(rec, out_path: str):
    """F4: Single IED template waveform."""
    if rec.template_z is None:
        print("  Skipped template — no template available.")
        return

    t_ms = np.linspace(-HALF_WIN_S * 1e3, HALF_WIN_S * 1e3, len(rec.template_z))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_ms, rec.template_z, color="#4c72b0", lw=1.5)
    ax.axvline(0, ls="--", color="grey", lw=0.7, label="IED peak")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("z-scored amplitude")
    ax.set_title(f"IED Template — {rec.subject}  (best ch={rec.best_channel})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_regressor_comparison(rec, out_path: str):
    """F5: Overlay of Ebrahimzadeh vs Grouiller regressor."""
    if rec.regressor_ebrahimzadeh is None or rec.regressor_grouiller is None:
        print("  Skipped regressor_comparison — missing one or both regressors.")
        return

    def _norm01(x):
        x = np.asarray(x, dtype=float)
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-12)

    reg_e = rec.regressor_ebrahimzadeh
    reg_g = rec.regressor_grouiller

    fig, ax = plt.subplots(figsize=(10, 4))
    t_e = np.arange(len(reg_e)) * TR
    t_g = np.arange(len(reg_g)) * TR
    ax.plot(t_e, _norm01(reg_e), label="Ebrahimzadeh (ICA, 5 s HRF)",
            color="#4c72b0", lw=1)
    ax.plot(t_g, _norm01(reg_g), label="Grouiller (spatial corr)",
            color="#c44e52", lw=1, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title(f"Ebrahimzadeh vs Grouiller Regressors — {rec.subject}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")
