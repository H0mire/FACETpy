#!/usr/bin/env python3
"""Generate publication-quality figures for the supervisor status update.

Reads:
    output/model_evaluations/<id>/holdout_v1/metrics.json   (per-model holdout metrics)
    output/model_evaluations/<id>/<run>/training.jsonl      (per-epoch loss curves)

Writes (PNG, 150 dpi):
    docs/reports/2026-05-12_status_update/figures/fig_1_ranking.png
    docs/reports/2026-05-12_status_update/figures/fig_2_run1_vs_holdout.png
    docs/reports/2026-05-12_status_update/figures/fig_3_training_curves.png
    docs/reports/2026-05-12_status_update/figures/fig_4_bug_analysis.png
    docs/reports/2026-05-12_status_update/figures/fig_5_quality_vs_cost.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "output/model_evaluations"
FIG_ROOT = REPO_ROOT / "docs/reports/2026-05-12_status_update/figures"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

# Set a consistent style — clean, paper-ready
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# Model → family → color
FAMILY: dict[str, str] = {
    "demucs": "Audio",
    "conv_tasnet": "Audio",
    "sepformer": "Audio",
    "nested_gan": "GAN",
    "dhct_gan": "GAN",
    "dhct_gan_v2": "GAN",
    "dpae": "Discriminative",
    "ic_unet": "Discriminative",
    "cascaded_dae": "Autoencoder",
    "cascaded_context_dae": "Autoencoder",
    "denoise_mamba": "SSM",
    "vit_spectrogram": "Vision",
    "st_gnn": "Graph",
    "d4pm": "Diffusion",
    "aas_naive_6nn": "Baseline (AAS)",
}

FAMILY_COLOR: dict[str, str] = {
    "Audio": "#1f77b4",          # blue
    "Discriminative": "#2ca02c", # green
    "Autoencoder": "#17becf",    # cyan
    "SSM": "#ff7f0e",            # orange
    "Vision": "#9467bd",         # purple
    "Graph": "#8c564b",          # brown
    "Diffusion": "#e377c2",      # pink
    "GAN": "#d62728",            # red
    "Baseline (AAS)": "#444444", # dark gray — reference baseline
}

DISPLAY_NAME: dict[str, str] = {
    "demucs": "Demucs",
    "conv_tasnet": "Conv-TasNet",
    "sepformer": "SepFormer",
    "nested_gan": "Nested-GAN",
    "denoise_mamba": "Denoise-Mamba",
    "ic_unet": "IC-U-Net",
    "vit_spectrogram": "ViT Spectrogram",
    "st_gnn": "ST-GNN",
    "dpae": "DPAE",
    "cascaded_dae": "Cascaded DAE",
    "cascaded_context_dae": "Cascaded Context DAE",
    "d4pm": "D4PM",
    "dhct_gan_v2": "DHCT-GAN v2",
    "dhct_gan": "DHCT-GAN v1",
    "aas_naive_6nn": "AAS (6-neighbor)",
}

RUN1_SNR: dict[str, float] = {
    "demucs": 31.28, "conv_tasnet": 22.03, "sepformer": 19.05,
    "cascaded_context_dae": 18.84, "cascaded_dae": 17.79,
    "nested_gan": 13.54, "denoise_mamba": 11.80, "ic_unet": 11.77,
    "vit_spectrogram": 11.60, "st_gnn": 11.00,
    "aas_naive_6nn": 9.16,
    "dpae": 7.48,
    "d4pm": 3.21, "dhct_gan_v2": 1.69, "dhct_gan": -7.13,
}


def load_holdout_metrics() -> dict[str, dict]:
    result: dict[str, dict] = {}
    for mid in FAMILY:
        p = EVAL_ROOT / mid / "holdout_v1" / "metrics.json"
        if not p.exists():
            continue
        payload = json.loads(p.read_text())
        flat = payload.get("flat_metrics", {})
        result[mid] = {k.replace("unified_holdout.", ""): v for k, v in flat.items()}
    return result


def load_training_curve(mid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (epochs, train_loss, val_loss) or None."""
    base = EVAL_ROOT / mid
    if not base.exists():
        return None
    # Pick the most recent run's training.jsonl
    candidates = sorted(base.glob("*/training.jsonl"))
    # Exclude the holdout_v1 dir (no training.jsonl there, but just in case)
    candidates = [c for c in candidates if c.parent.name != "holdout_v1"]
    if not candidates:
        return None
    # Prefer the largest (longest training run)
    path = max(candidates, key=lambda p: p.stat().st_size)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return None
    epochs = np.array([r["epoch"] for r in rows])
    train_loss = np.array([r.get("train_loss", np.nan) for r in rows])
    val_loss = np.array([r.get("val_loss", np.nan) for r in rows])
    return epochs, train_loss, val_loss


# ---------------------------------------------------------------------------
# Figure 1 — Unified Holdout Ranking
# ---------------------------------------------------------------------------


def fig_1_ranking(metrics: dict[str, dict]) -> None:
    rows = sorted(
        [(mid, m) for mid, m in metrics.items() if "clean_snr_improvement_db" in m],
        key=lambda kv: kv[1]["clean_snr_improvement_db"],  # ascending so top is highest
    )
    names = [DISPLAY_NAME[m] for m, _ in rows]
    snr = [r["clean_snr_improvement_db"] for _, r in rows]
    fams = [FAMILY[m] for m, _ in rows]
    colors = [FAMILY_COLOR[f] for f in fams]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(names, snr, color=colors, alpha=0.85)
    for bar, val in zip(bars, snr):
        x_text = val + 0.4 if val >= 0 else val - 0.4
        ha = "left" if val >= 0 else "right"
        ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f} dB", va="center", ha=ha, fontsize=9)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("SNR-Verbesserung [dB]   (höher = besser)")
    ax.set_title("Cross-Model Ranking auf Unified Holdout\n166 Windows × 30 Kanäle = 4980 Channel-Windows, seed=42",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(min(snr) - 4, max(snr) + 4)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()]
    ax.legend(handles=handles, loc="lower right", title="Familie",
              frameon=True, fontsize=9)
    fig.savefig(FIG_ROOT / "fig_1_ranking.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Run 1 vs Unified Holdout Consistency
# ---------------------------------------------------------------------------


def fig_2_run1_vs_holdout(metrics: dict[str, dict]) -> None:
    pairs = [
        (mid, RUN1_SNR[mid], metrics[mid]["clean_snr_improvement_db"])
        for mid in metrics
        if mid in RUN1_SNR and "clean_snr_improvement_db" in metrics[mid]
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    lo, hi = -10, 35
    ax.plot([lo, hi], [lo, hi], "--", color="#888", linewidth=0.8, label="y = x")
    # ±1 dB tolerance band
    ax.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1],
                    color="#bbb", alpha=0.25, label="±1 dB Toleranz")

    for mid, x, y in pairs:
        c = FAMILY_COLOR[FAMILY[mid]]
        ax.scatter(x, y, color=c, s=80, edgecolors="white", linewidth=1.0, zorder=3)
        # Label
        dy = 0.6
        if mid == "demucs":
            dy = -1.2
        if mid in ("dhct_gan_v2", "d4pm", "nested_gan"):
            ax.annotate(DISPLAY_NAME[mid], xy=(x, y), xytext=(x + 0.8, y + dy),
                        fontsize=8.5, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color="#666", lw=0.5))
        else:
            ax.annotate(DISPLAY_NAME[mid], xy=(x, y), xytext=(x + 0.5, y + dy),
                        fontsize=8.5)
    ax.set_xlabel("Run 1 SNR-Verbesserung [dB]\n(unterschiedliche Test-Splits pro Modell)")
    ax.set_ylabel("Unified Holdout SNR-Verbesserung [dB]\n(166 Windows, identisch für alle 12)")
    ax.set_title("Konsistenz von Run 1 zu Unified Holdout\n9 von 12 Modellen <±1 dB → ursprüngliches Ranking war qualitativ korrekt",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    fig.savefig(FIG_ROOT / "fig_2_run1_vs_holdout.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — All Training Curves (4×3 grid)
# ---------------------------------------------------------------------------


def fig_3_training_curves(metrics: dict[str, dict]) -> None:
    # Order by holdout SNR desc, so the layout reads top-to-bottom by quality
    order = sorted(
        [m for m in FAMILY if m in metrics and "clean_snr_improvement_db" in metrics[m]],
        key=lambda m: -metrics[m]["clean_snr_improvement_db"],
    )
    # Pad to 12 if needed
    while len(order) < 12:
        order.append(None)

    fig, axes = plt.subplots(4, 3, figsize=(14, 12))
    axes_flat = axes.flatten()
    for ax, mid in zip(axes_flat, order):
        if mid is None:
            ax.axis("off")
            continue
        curve = load_training_curve(mid)
        if curve is None:
            ax.set_title(f"{DISPLAY_NAME[mid]} — kein Log gefunden")
            ax.axis("off")
            continue
        epochs, tr, val = curve
        c = FAMILY_COLOR[FAMILY[mid]]
        ax.plot(epochs, tr, "-", color=c, linewidth=1.6, label="train", alpha=0.9)
        ax.plot(epochs, val, "--", color=c, linewidth=1.6, label="val", alpha=0.7)

        snr = metrics[mid]["clean_snr_improvement_db"]
        title = f"{DISPLAY_NAME[mid]}    Δ SNR = {snr:+.2f} dB"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoche", fontsize=9)
        ax.set_ylabel("Loss (log)", fontsize=9)

        # Determine if log scale is appropriate (positive values only)
        all_loss = np.concatenate([tr[~np.isnan(tr)], val[~np.isnan(val)]])
        all_loss = all_loss[all_loss > 0]
        if len(all_loss) > 1 and all_loss.max() / max(all_loss.min(), 1e-12) > 10:
            ax.set_yscale("log")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle("Trainings- und Validierungskurven aller 12 Modelle\n(sortiert nach Unified-Holdout SNR-Verbesserung)",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "fig_3_training_curves.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Bug Analysis Zoom
# ---------------------------------------------------------------------------


def fig_4_bug_analysis() -> None:
    bugs = [
        ("vit_spectrogram", "Dead-Clamp Output-Collapse",
         "expm1+clamp → Gradient null →\nval_loss bit-identisch alle 13 Epochen"),
        ("dpae", "BatchNorm Train/Eval Gap",
         "Per-Kanal Daten + BN-Running-Stats →\nval_loss-Spikes um 4 Größenordnungen"),
        ("dhct_gan", "GAN-Collapse + falscher Input",
         "Single-Epoch Input + adv. Loss-Dominanz →\nmonoton steigender Loss"),
        ("dhct_gan_v2", "GAN-Collapse (Input gefixt)",
         "Adv. Loss dominiert weiter →\nOszillation 0.13-0.77, kein Konvergenz"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (mid, bug_name, bug_text) in zip(axes.flatten(), bugs):
        curve = load_training_curve(mid)
        if curve is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        epochs, tr, val = curve
        c = FAMILY_COLOR[FAMILY[mid]]
        ax.plot(epochs, tr, "-", color=c, linewidth=2, label="train", alpha=0.9)
        ax.plot(epochs, val, "--", color=c, linewidth=2, label="val", alpha=0.7)
        ax.set_title(f"{DISPLAY_NAME[mid]} — {bug_name}",
                     fontsize=11, fontweight="bold", color="#c0392b")
        ax.set_xlabel("Epoche")
        ax.set_ylabel("Loss (log)")
        ax.set_yscale("log")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.25, linestyle="--")
        # Bug annotation box
        ax.text(0.97, 0.05, bug_text, transform=ax.transAxes,
                fontsize=9, ha="right", va="bottom",
                bbox=dict(facecolor="#fff5f5", edgecolor="#c0392b",
                          alpha=0.9, boxstyle="round,pad=0.5"))
    fig.suptitle("Identifizierte Defekte — Ursachenanalyse der 4 problematischen Modelle",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "fig_4_bug_analysis.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Inference Cost vs Quality
# ---------------------------------------------------------------------------


def fig_5_quality_vs_cost(metrics: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for mid, m in metrics.items():
        if "clean_snr_improvement_db" not in m:
            continue
        t = m.get("inference_seconds", 1.0)
        snr = m["clean_snr_improvement_db"]
        c = FAMILY_COLOR[FAMILY[mid]]
        # Marker size by art_corr (larger = better)
        corr = m.get("artifact_corr", 0.5)
        size = 50 + 350 * max(0, corr)
        ax.scatter(t, snr, s=size, color=c, alpha=0.7, edgecolors="white",
                   linewidth=1.2, zorder=3)
        ax.annotate(DISPLAY_NAME[mid], xy=(t, snr), xytext=(5, 5),
                    textcoords="offset points", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("Inferenzzeit auf 4980 Channel-Windows [s] (log-Skala)")
    ax.set_ylabel("SNR-Verbesserung [dB]")
    ax.set_title("Qualität vs. Inferenzkosten\nMarkergröße ∝ Artefakt-Korrelation",
                 fontsize=11, fontweight="bold")
    ax.axhline(0, color="#999", linewidth=0.6, linestyle=":")
    ax.grid(alpha=0.25, linestyle="--")
    # Family legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()]
    ax.legend(handles=handles, loc="lower right", title="Familie", fontsize=9)
    fig.savefig(FIG_ROOT / "fig_5_quality_vs_cost.png")
    plt.close(fig)


def fig_6_metric_matrix(metrics: dict[str, dict]) -> None:
    """Compare all metrics across all models in a 2x4 panel of sorted horizontal bars."""
    order = sorted(
        [m for m in FAMILY if m in metrics and "clean_snr_improvement_db" in metrics[m]],
        key=lambda m: metrics[m]["clean_snr_improvement_db"],  # ascending
    )
    names = [DISPLAY_NAME[m] for m in order]
    colors = [FAMILY_COLOR[FAMILY[m]] for m in order]

    # Define panels: (key, label, fmt, "higher=better" bool, optional vline at value)
    panels = [
        ("clean_snr_improvement_db",   "SNR-Verbesserung [dB]",         "{:+.2f}",  True,  0.0),
        ("artifact_corr",              "Artefakt-Korrelation (Pearson)", "{:.3f}",  True,  None),
        ("rms_recovery_ratio",         "RMS-Recovery (FACETpy-Style)\nZiel = 1.0", "{:.3f}", True, 1.0),
        ("residual_error_rms_ratio",   "Residual-Error-RMS-Ratio\n← niedriger = besser",  "{:.3f}", False, 1.0),
        ("artifact_snr_db",            "Artefakt-SNR [dB]\n(pred ≈ true)", "{:+.2f}", True, 0.0),
        ("clean_mse_reduction_pct",    "Clean-MSE-Reduktion [%]",       "{:+.1f}",  True,  0.0),
        ("rms_recovery_distance",      "|RMS-Recovery − 1|\n← niedriger = besser", "{:.3f}", False, 0.0),
        ("artifact_mae",               "Artefakt-MAE\n← niedriger = besser",        "{:.4f}", False, None),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (key, label, fmt, higher_better, vline) in zip(axes.flatten(), panels):
        vals = [metrics[m].get(key, np.nan) for m in order]
        bars = ax.barh(names, vals, color=colors, alpha=0.85)
        ax.set_title(label, fontsize=10.5, fontweight="bold")

        if vline is not None:
            ax.axvline(vline, color="#444", linewidth=0.8, linestyle=":")

        # Value labels
        max_val = np.nanmax(vals)
        min_val = np.nanmin(vals)
        pad = 0.02 * max(abs(max_val), abs(min_val), 1e-6)
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            xpos = v + pad if v >= 0 else v - pad
            ha = "left" if v >= 0 else "right"
            ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", ha=ha, fontsize=8)

        # Set wider x-limits so labels don't get clipped
        span = max_val - min_val if max_val > min_val else max(abs(max_val), 1)
        ax.set_xlim(min_val - 0.18 * span, max_val + 0.18 * span)
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8.5)

    # Family legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, title="Familie",
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    fig.suptitle("Alle Metriken nebeneinander — Unified Holdout (166 Windows × 30 Kanäle = 4980 ch-w)",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0.025, 1, 0.99])
    fig.savefig(FIG_ROOT / "fig_6_metric_matrix.png")
    plt.close(fig)


def fig_7_metric_heatmap(metrics: dict[str, dict]) -> None:
    """Normalized heatmap: rows=models (by SNR desc), cols=metrics, color=relative quality."""
    order = sorted(
        [m for m in FAMILY if m in metrics and "clean_snr_improvement_db" in metrics[m]],
        key=lambda m: -metrics[m]["clean_snr_improvement_db"],  # top = best
    )
    names = [DISPLAY_NAME[m] for m in order]

    # (key, label, "better" semantics — "higher", "lower", or "target1" for "closer to 1")
    cols = [
        ("clean_snr_improvement_db",   "SNR↑ [dB]",        "higher"),
        ("clean_snr_db_after",         "SNR after [dB]",   "higher"),
        ("artifact_corr",              "art.corr",         "higher"),
        ("rms_recovery_ratio",         "RMS-Rec (FACETpy)\n→ Ziel 1.0",  "target1"),
        ("artifact_snr_db",            "art.SNR [dB]",     "higher"),
        ("clean_mse_reduction_pct",    "MSE red. [%]",     "higher"),
        ("residual_error_rms_ratio",   "res.err RMS ratio\n→ Ziel 0",   "lower"),
        ("artifact_mse",               "art.MSE",          "lower"),
        ("artifact_mae",               "art.MAE",          "lower"),
    ]

    raw = np.array([
        [metrics[m].get(k, np.nan) for k, _, _ in cols]
        for m in order
    ])
    # Per-column min-max normalization, oriented so "1.0 = best"
    norm = np.zeros_like(raw)
    for j, (_, _, sem) in enumerate(cols):
        col = raw[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            norm[:, j] = 0.5
            continue
        if sem == "target1":
            # closer to 1.0 = better; rank by |value - 1|
            dist = np.abs(col - 1.0)
            dist_valid = dist[valid]
            lo, hi = np.min(dist_valid), np.max(dist_valid)
            if hi == lo:
                norm[:, j] = 0.5
            else:
                norm[:, j] = 1.0 - (dist - lo) / (hi - lo)
        else:
            lo, hi = np.min(col[valid]), np.max(col[valid])
            if hi == lo:
                norm[:, j] = 0.5
            else:
                scaled = (col - lo) / (hi - lo)
                norm[:, j] = scaled if sem == "higher" else 1.0 - scaled

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(norm, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], rotation=30, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)

    # Annotate cells with raw values
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            val = raw[i, j]
            if np.isnan(val):
                continue
            key = cols[j][0]
            if "db" in key:
                txt = f"{val:+.1f}"
            elif "pct" in key:
                txt = f"{val:+.0f}"
            elif "recovery_ratio" in key:
                txt = f"{val:.2f}"
            elif "corr" in key or "ratio" in key:
                txt = f"{val:.3f}"
            else:
                txt = f"{val:.4f}"
            # Text color: contrast based on normalized value
            n = norm[i, j]
            color = "#222" if 0.3 < n < 0.7 else ("#fff" if n < 0.3 else "#222")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Metrik-Heatmap — normalisiert pro Spalte (grün = best, rot = schlechtest)\nZellbeschriftung: Rohwert auf Unified Holdout",
                 fontsize=11.5, fontweight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Relative Qualität (0 = schlechtest, 1 = best, pro Spalte)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Family color strip on the left
    family_colors = [FAMILY_COLOR[FAMILY[m]] for m in order]
    for i, c in enumerate(family_colors):
        ax.add_patch(plt.Rectangle((-0.7, i - 0.5), 0.4, 1.0, color=c, clip_on=False))
    ax.text(-0.5, -0.8, "Familie", fontsize=8, ha="center", color="#555")

    fig.tight_layout()
    fig.savefig(FIG_ROOT / "fig_7_metric_heatmap.png")
    plt.close(fig)


def main() -> int:
    metrics = load_holdout_metrics()
    print(f"Loaded metrics for {len(metrics)} models")
    fig_1_ranking(metrics)
    print("  fig_1_ranking.png")
    fig_2_run1_vs_holdout(metrics)
    print("  fig_2_run1_vs_holdout.png")
    fig_3_training_curves(metrics)
    print("  fig_3_training_curves.png")
    fig_4_bug_analysis()
    print("  fig_4_bug_analysis.png")
    fig_5_quality_vs_cost(metrics)
    print("  fig_5_quality_vs_cost.png")
    fig_6_metric_matrix(metrics)
    print("  fig_6_metric_matrix.png")
    fig_7_metric_heatmap(metrics)
    print("  fig_7_metric_heatmap.png")
    print(f"\nAll figures in {FIG_ROOT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
