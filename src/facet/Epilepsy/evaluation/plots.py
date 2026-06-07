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
    figs[0].suptitle(f"Accepted ICA Topographies — {rec.subject}")
    figs[0].savefig(out_path, dpi=150)
    for f in figs:
        plt.close(f)
    print(f"  Saved {out_path}")


def plot_ica_reproducibility(rec, out_path: str):
    """F7: Bar chart — run frequency of each accepted IC across ICA runs."""
    counts = rec.ica_selection_stats.get("component_run_counts", {})
    n_runs = int(rec.ica_selection_stats.get("n_runs", 0))

    if not rec.accepted_indices or not counts:
        print("  Skipped ica_reproducibility — no reproducibility data.")
        return

    labels = [f"IC{idx}" for idx in rec.accepted_indices]
    vals = [int(counts.get(idx, 0)) for idx in rec.accepted_indices]

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 5))
    bars = ax.bar(labels, vals, color="#4c72b0", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v}/{n_runs}", ha="center", va="bottom", fontweight="bold")
    if n_runs > 0:
        ax.set_ylim(0, n_runs * 1.1)
    ax.set_ylabel("Runs the component appeared in")
    ax.set_title(f"ICA Reproducibility — {rec.subject}  (n_runs={n_runs})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
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

    ch_names = [raw.ch_names[i] for i in eeg_picks]
    emap = np.asarray(emap, dtype=float)
    peak_idx = int(np.argmax(np.abs(emap)))
    peak_channel = ch_names[peak_idx] if peak_idx < len(ch_names) else "?"
    a = np.abs(emap)
    med = float(np.median(a))
    focality = float(np.max(a) / med) if med > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(5, 5.5))
    im, _ = mne.viz.plot_topomap(
        emap, info, axes=ax, show=False, cmap="RdBu_r", contours=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Voltage (a.u.)")
    ax.set_title(
        f"Grouiller Epileptic Map — {rec.subject}\n"
        f"peak channel = {peak_channel}, focality = {focality:.2f}"
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

