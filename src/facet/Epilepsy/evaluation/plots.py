"""Evaluation figure generation for single-subject and group analyses.

Single-subject figures take a ``SubjectRecord`` and an output path.
Group figures take the aggregated ``group_summary`` DataFrame and an output
path.  No computation logic lives here.
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
TH_RAW = 0.85
TR = 2.5
HALF_WIN_S = 0.15


def _fallback_note(rec) -> str:
    """' [FALLBACK]' if TCCC fell back to the best below-threshold component."""
    stats = rec.ica_selection_stats or {}
    fb = stats.get("fallback_used", False)
    return " [FALLBACK — below threshold]" if fb else ""



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
    ax.set_title(f"IED Template — {rec.subject}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_ica_topomaps(rec, out_path: str):
    """F6: Scalp topographies of the accepted ICA components (region review)."""
    det = rec.detection
    if det is None or not rec.accepted_indices:
        print("  Skipped ica_topomaps — no accepted components.")
        return
    ica = getattr(det, "ica", None)
    if ica is None:
        print("  Skipped ica_topomaps — no fitted ICA on detection.")
        return
    if ica.info.get_montage() is None:
        print("  Skipped ica_topomaps — no montage (channel positions) set.")
        return

    fig = ica.plot_components(picks=rec.accepted_indices, show=False)
    figs = fig if isinstance(fig, list) else [fig]
    figs[0].suptitle(
        f"Accepted ICA Topographies (TCCC) — {rec.subject}"
        f"{_fallback_note(rec)}"
    )
    figs[0].savefig(out_path, dpi=150)
    for f in figs:
        plt.close(f)
    print(f"  Saved {out_path}")


def plot_grouiller_map(rec, out_path: str):
    """F8: Topographic map of the Grouiller epileptic voltage map."""
    import mne

    emap = rec.epileptic_map
    if emap is None or len(emap) == 0:
        print("  Skipped grouiller_map — no epileptic map available.")
        return

    det = rec.detection
    raw = getattr(det, "raw", None) if det is not None else None
    if raw is None:
        print("  Skipped grouiller_map — no raw for channel positions.")
        return

    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    if len(eeg_picks) != len(emap):
        print("  Skipped grouiller_map — channel/map length mismatch.")
        return

    info = mne.pick_info(raw.info, eeg_picks)
    if info.get_montage() is None:
        print("  Skipped grouiller_map — no montage (channel positions) set.")
        return

    emap = np.asarray(emap, dtype=float)
    a = np.abs(emap)
    med = float(np.median(a))
    focality = float(np.max(a) / med) if med > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(5, 5.5))
    im, _ = mne.viz.plot_topomap(
        emap, info, axes=ax, show=False, cmap="RdBu_r", contours=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Voltage (a.u.)")
    ax.set_title(
        f"Grouiller Epileptic Map — {rec.subject}\n"
        f"focality = {focality:.2f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Group-level figures ─────────────────────────────────────────────────────

def plot_group_acceptance(df, out_path: str):
    """G1: Bar chart of accepted-component count per subject."""
    if df.empty:
        print("  Skipped group_acceptance — empty group dataframe.")
        return
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.4), 4))
    ax.bar(df["subject"], df["n_accepted_components"], color="#4c72b0")
    ax.set_ylabel("# accepted ICA components")
    ax.set_title("Accepted Components per Subject")
    ax.set_xticklabels(df["subject"], rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

