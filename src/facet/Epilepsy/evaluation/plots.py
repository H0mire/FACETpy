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

    # Marker = actual 90th percentile used by the TCCC acceptance decision.
    p90_vals = [float(np.percentile(wc, 90)) for wc in data]
    x_pos = np.arange(1, len(p90_vals) + 1)
    ax.scatter(
        x_pos,
        p90_vals,
        marker="D",
        s=40,
        c="#f28e2b",
        edgecolors="black",
        linewidths=0.6,
        zorder=3,
        label="90th percentile per component",
    )

    ax.axhline(
        TH_RAW,
        color="red",
        ls="--",
        lw=1,
        label="TCCC acceptance threshold (90th-percentile score = 0.85)",
    )
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

    # Plot ALL temporally TCCC-selected components. In spatial-gated mode
    # ``rec.accepted_indices`` has been reduced to the spatially-accepted
    # survivors (used downstream for the fused map / regions), but this figure
    # is meant to review every component TCCC accepted temporally, so pull the
    # full pre-gate temporal set from the gate summary when available.
    sg = getattr(rec, "spatial_gate", None)
    if sg is not None:
        picks = sg.get("temporally_accepted_candidates") or sg.get(
            "baseline_accepted_components") or rec.accepted_indices
    else:
        picks = rec.accepted_indices
    if not picks:
        print("  Skipped ica_topomaps — no accepted components.")
        return

    fig = ica.plot_components(picks=picks, show=False)
    figs = fig if isinstance(fig, list) else [fig]
    # The full temporal set is shown for review; the note flags the
    # spatially-validated survivors that TCCC/FUSED actually use (and the
    # highest-|r| representative).
    spatial_note = ""
    if sg is not None:
        thr = sg.get("spatial_abs_corr_threshold", 0.5)
        rep = sg.get("fused_representative_component",
                     sg.get("spatial_tccc_representative_component"))
        survivors = sg.get("final_tccc_accepted_components", [])
        if sg.get("spatial_gate_fallback_used"):
            spatial_note = (
                f" [temporal set shown; TCCC/FUSED = fallback {rep} "
                f"(none passed |r|>={thr:g})]")
        else:
            spatial_note = (
                f" [temporal set shown; TCCC/FUSED = {survivors} "
                f"(|r|>={thr:g}), rep {rep}]")
    figs[0].suptitle(
        f"Accepted ICA Topographies (TCCC) — {rec.subject}"
        f"{_fallback_note(rec)}{spatial_note}"
    )
    figs[0].savefig(out_path, dpi=150)
    for f in figs:
        plt.close(f)
    print(f"  Saved {out_path}")


def plot_grouiller_map(rec, out_path: str, emap=None, title: str = "Grouiller Epileptic Map"):
    """F8: Topographic map of an epileptic voltage map.

    Defaults to the Grouiller map (``rec.epileptic_map``); pass ``emap`` and a
    ``title`` to render another map (e.g. the fused pipeline's map) with the
    same styling.
    """
    import mne

    if emap is None:
        emap = rec.epileptic_map
    if emap is None or len(emap) == 0:
        print(f"  Skipped {title} — no epileptic map available.")
        return

    det = rec.detection
    raw = getattr(det, "raw", None) if det is not None else None
    if raw is None:
        print(f"  Skipped {title} — no raw for channel positions.")
        return

    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, exclude="bads")
    if len(eeg_picks) != len(emap):
        print(f"  Skipped {title} — channel/map length mismatch.")
        return

    info = mne.pick_info(raw.info, eeg_picks)
    if info.get_montage() is None:
        print(f"  Skipped {title} — no montage (channel positions) set.")
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
        f"{title} — {rec.subject}\n"
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


def plot_group_template_channel_distribution(df, out_path: str):
    """G2: Bar chart of how often each channel was picked as the IED template source."""
    if df.empty or "template_channel" not in df.columns:
        print("  Skipped group_template_channel_distribution — no data.")
        return
    counts = df["template_channel"].value_counts(dropna=False)
    if counts.empty:
        print("  Skipped group_template_channel_distribution — no data.")
        return
    fig, ax = plt.subplots(figsize=(max(6, len(counts) * 0.5), 4))
    ax.bar(counts.index.astype(str), counts.values, color="#55a868")
    ax.set_ylabel("# subjects")
    ax.set_title("Template Channel Selection Distribution")
    ax.set_xticklabels(counts.index.astype(str), rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")

