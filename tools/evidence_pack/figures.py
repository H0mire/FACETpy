"""Figure builders for the results evidence pack.

Every function here takes primary data and a destination path and returns
nothing; the caller registers the sources. Three rules are enforced by
construction rather than by review:

* **No selection inside a figure.** Which examples or models appear is decided by
  the caller from a rule stated in the section's provenance, never by sorting on
  the quantity being plotted.
* **Paired data is drawn paired.** Where a comparison is paired, the figure shows
  the per-example connection or the per-example difference. Two unconnected bars
  hide exactly the information the paired test uses.
* **Identical scaling within a panel group.** Example traces share one y-limit
  per figure, so a visually small residual is small in microvolts too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 200
# Colour-blind-safe (Okabe-Ito). Fixed per role so a model keeps its colour
# across figures.
C = {
    "model": "#0072B2",
    "cascade": "#0072B2",
    "farm": "#D55E00",
    "aas_ideal": "#D55E00",
    "null": "#999999",
    "null_output": "#999999",
    "clean": "#009E73",
    "noisy": "#444444",
    "alt1": "#CC79A7",
    "alt2": "#E69F00",
    "alt3": "#56B4E9",
}


def _colour(name: str, fallback: int = 0) -> str:
    if name in C:
        return C[name]
    return [C["alt1"], C["alt2"], C["alt3"], "#882255"][fallback % 4]


def _finish(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- paired

def paired_metric_panels(
    out: Path,
    per_example: dict[str, dict[int, dict[str, float]]],
    arm_a: str,
    arm_b: str,
    metrics: Sequence[tuple[str, str, str]],
    stats: dict[str, dict[str, float]],
    title: str,
) -> None:
    """One panel per metric: every example as a connected pair plus the effect.

    Left of each panel are the two arms with one line per example, so the reader
    sees how many examples actually move and in which direction. Right is the
    Hodges-Lehmann difference with its bootstrap interval — the quantity the test
    is about. A bar chart of two means would show neither.
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 4.0), squeeze=False)
    shared = sorted(set(per_example[arm_a]) & set(per_example[arm_b]))
    for ax, (key, label, unit) in zip(axes[0], metrics):
        a = np.array([per_example[arm_a][i].get(key, np.nan) for i in shared])
        b = np.array([per_example[arm_b][i].get(key, np.nan) for i in shared])
        ok = np.isfinite(a) & np.isfinite(b)
        for ai, bi in zip(a[ok], b[ok]):
            ax.plot([0, 1], [ai, bi], color="#BBBBBB", lw=0.7, zorder=1)
        ax.scatter(np.zeros(ok.sum()), a[ok], s=16, color=_colour(arm_a), zorder=3, label=arm_a)
        ax.scatter(np.ones(ok.sum()), b[ok], s=16, color=_colour(arm_b), zorder=3, label=arm_b)
        ax.plot([0, 1], [np.median(a[ok]), np.median(b[ok])], color="black", lw=2.0, zorder=4)
        ax.set_xlim(-0.4, 1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([arm_a, arm_b], rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel(unit, fontsize=8)
        ax.tick_params(labelsize=8)
        s = stats.get(key)
        if s:
            p = s.get("p_holm", float("nan"))
            hl = s.get("hodges_lehmann_difference", float("nan"))
            marker = "*" if s.get("significant") else "n.s."
            ax.text(
                0.5, 0.02,
                f"HL {hl:+.3g}\n[{s.get('ci_low', float('nan')):.3g}, {s.get('ci_high', float('nan')):.3g}]\n"
                f"p={p:.2g} {marker}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.9),
            )
    fig.suptitle(f"{title}   ·   n = {len(shared)} gepaarte Spike-Beispiele", fontsize=10)
    _finish(fig, out)


def effect_forest(
    out: Path,
    rows: Sequence[dict],
    label_key: str,
    title: str,
    xlabel: str,
    better_note: str,
) -> None:
    """Effect sizes with intervals, one row per comparison — a forest plot.

    Used where several comparisons share a metric. Non-significant rows are drawn
    grey, so they cannot be mistaken for a result.

    Significance is read from ``p_holm`` when it is available, not from whether
    the interval crosses zero. The two disagree: a bootstrap interval can exclude
    zero while the Holm correction over several metrics still keeps the row above
    alpha. Colouring by the interval alone would then paint a row as a result
    that the registered test does not support. Rows without a usable p-value fall
    back to the interval rule.
    """
    fig, ax = plt.subplots(figsize=(7.4, 0.42 * len(rows) + 1.6))
    ys = np.arange(len(rows))[::-1]
    for y, r in zip(ys, rows):
        lo, hi, hl = float(r["ci_low"]), float(r["ci_high"]), float(r["hodges_lehmann_difference"])
        p_holm = float(r.get("p_holm", float("nan")))
        significant = p_holm < 0.05 if np.isfinite(p_holm) else not (lo <= 0 <= hi)
        colour = C["model"] if significant else "#999999"
        ax.plot([lo, hi], [y, y], color=colour, lw=2.0, solid_capstyle="round")
        ax.scatter([hl], [y], s=34, color=colour, zorder=3)
        ax.text(hi, y, f"  p={float(r['p_holm']):.2g}", va="center", fontsize=7, color="#333333")
    ax.axvline(0.0, color="black", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[label_key] for r in rows], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.text(0.99, -0.14, better_note, transform=ax.transAxes, ha="right", fontsize=7, color="#555555")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)
    _finish(fig, out)


# ------------------------------------------------------------------- examples

def example_traces(
    out: Path,
    traces: dict[str, np.ndarray],
    arms: Sequence[str],
    example_index: Sequence[int],
    sfreq: float,
    title: str,
    zoom_samples: int | None = None,
    ylim_from: str = "all",
) -> None:
    """One row per example, one column per arm, shared y-limits.

    ``zoom_samples`` restricts the window to that many samples around the spike
    peak, for the morphology figure. The peak is taken from the *label*, not from
    any prediction, so all arms are cut identically.

    ``ylim_from`` decides what sets the shared y-limit of a row:

    * ``"all"`` — the largest excursion of any arm. Correct when the point is the
      *scale* of the failure, because nothing is cut off.
    * ``"clean"`` — the reference trace. Needed for a morphology figure: with
      ``"all"``, one arm's 200 µV residual oscillation flattens a 20 µV spike into
      a straight line and the figure shows nothing about spike shape. Any trace
      that leaves the axis is annotated with its true peak in the panel, so the
      cut is stated rather than hidden.
    """
    clean, spikes = traces["clean"], traces["spikes"]
    n_rows = clean.shape[0]
    cols = ["clean"] + list(arms)
    fig, axes = plt.subplots(n_rows, len(cols), figsize=(2.5 * len(cols), 1.55 * n_rows),
                             squeeze=False, sharex=False)
    scale = 1e6                                            # volts -> microvolts

    windows = []
    for r in range(n_rows):
        mask = spikes[r] > 0
        peak = int(np.argmax(np.abs(np.where(mask, clean[r], 0.0)))) if mask.any() else clean.shape[1] // 2
        if zoom_samples:
            half = zoom_samples // 2
            lo = max(0, min(peak - half, clean.shape[1] - zoom_samples))
            windows.append(slice(lo, lo + zoom_samples))
        else:
            windows.append(slice(0, clean.shape[1]))

    for r in range(n_rows):
        w = windows[r]
        row_series = [clean[r, w]] + [traces[f"corrected_{a}"][r, w] for a in arms]
        if ylim_from == "clean":
            lim = float(np.nanmax(np.abs(clean[r, w]))) * scale * 2.0
        else:
            lim = float(np.nanmax([np.nanmax(np.abs(s)) for s in row_series])) * scale * 1.15
        lim = max(lim, 1e-6)
        t = np.arange(w.start, w.stop) / sfreq * 1e3        # ms within the epoch
        for c, name in enumerate(cols):
            ax = axes[r][c]
            series = clean[r, w] if name == "clean" else traces[f"corrected_{name}"][r, w]
            if name != "clean":
                ax.plot(t, clean[r, w] * scale, color=C["clean"], lw=0.7, alpha=0.65, label="clean")
            ax.plot(t, series * scale, color=_colour(name, c), lw=0.9,
                    label=name if name != "clean" else "clean (Referenz)")
            spike_mask = spikes[r, w] > 0
            if spike_mask.any():
                ax.fill_between(t, -lim, lim, where=spike_mask, color="#FFE9A8", alpha=0.7, zorder=0)
            ax.set_ylim(-lim, lim)
            peak = float(np.nanmax(np.abs(series))) * scale
            if peak > lim:
                # The trace is cut by the shared limit. Naming the true peak keeps
                # the figure honest: a reader must not mistake the cut for a small
                # residual.
                ax.text(0.98, 0.94, f"Peak {peak:.0f} µV\n(abgeschnitten)",
                        transform=ax.transAxes, ha="right", va="top", fontsize=6,
                        color="#B03000",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#E0C0B0", alpha=0.85))
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"#{int(example_index[r])}\nµV", fontsize=7)
            if r == n_rows - 1:
                ax.set_xlabel("ms", fontsize=7)
    fig.suptitle(title, fontsize=10)
    _finish(fig, out)


def signal_deletion(out: Path, traces: dict[str, np.ndarray], arm: str, sfreq: float, title: str) -> None:
    """Raw / clean / target / prediction / recovered clean at unchanged scale.

    The degeneracy claim is that a low artifact error can coexist with a deleted
    EEG. That is only visible when the artifact-scale and EEG-scale panels sit
    above each other with their true amplitudes, which is what this draws.
    """
    n = min(3, traces["clean"].shape[0])
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 4.4), squeeze=False)
    scale = 1e6
    for c in range(n):
        t = np.arange(traces["clean"].shape[1]) / sfreq * 1e3
        art_ax, eeg_ax = axes[0][c], axes[1][c]
        art_ax.plot(t, traces["noisy"][c] * scale, color=C["noisy"], lw=0.6, label="noisy")
        art_ax.plot(t, traces["artifact"][c] * scale, color=C["farm"], lw=0.6, label="Artefakt (wahr)")
        art_ax.set_title(f"Artefaktskala — Beispiel #{int(traces['example_index'][c])}", fontsize=8)
        art_ax.tick_params(labelsize=6)
        art_ax.set_ylabel("µV", fontsize=7)

        eeg_ax.plot(t, traces["clean"][c] * scale, color=C["clean"], lw=0.9, label="clean (wahr)")
        eeg_ax.plot(t, traces[f"corrected_{arm}"][c] * scale, color=C["model"], lw=0.9, label=f"{arm}")
        eeg_ax.plot(t, traces["corrected_null_output"][c] * scale, color=C["null"], lw=0.8, ls=":", label="Nullausgabe")
        lim = float(np.nanmax(np.abs(traces["clean"][c]))) * scale * 2.2
        eeg_ax.set_ylim(-lim, lim)
        eeg_ax.set_title("EEG-Skala (gleiche Daten, y gezoomt)", fontsize=8)
        eeg_ax.tick_params(labelsize=6)
        eeg_ax.set_xlabel("ms", fontsize=7)
        eeg_ax.set_ylabel("µV", fontsize=7)
        if c == 0:
            art_ax.legend(fontsize=6, loc="upper right")
            eeg_ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(title, fontsize=10)
    _finish(fig, out)


# --------------------------------------------------------------- comparisons

def ranking_with_n(
    out: Path,
    rows: Sequence[dict],
    value_key: str,
    label_key: str,
    n_key: str,
    title: str,
    xlabel: str,
    reference: dict[str, float] | None = None,
) -> None:
    """Horizontal ranking that keeps ``n`` visible next to every bar."""
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(rows) + 1.7))
    ys = np.arange(len(rows))[::-1]
    values = [float(r[value_key]) for r in rows]
    colours = [C["farm"] if str(r.get("family", "")).startswith("Baseline") else C["model"] for r in rows]
    ax.barh(ys, values, color=colours, height=0.62)
    for y, r, v in zip(ys, rows, values):
        ax.text(v, y, f"  n={r[n_key]}", va="center", fontsize=7, color="#333333")
    if reference:
        for name, value in reference.items():
            ax.axvline(value, color="#666666", lw=0.9, ls="--")
            ax.text(value, len(rows) - 0.4, f" {name}", fontsize=7, color="#666666", rotation=90, va="top")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[label_key] for r in rows], fontsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)
    _finish(fig, out)


def context_utilisation(out: Path, rows: Sequence[dict], title: str) -> None:
    """Per-edition epoch-gradient share plus receptive-field fraction.

    Two panels because one number misleads: an edition can consume every epoch
    and still have a receptive field that reaches almost none of the samples.
    """
    rows = list(rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 0.36 * len(rows) + 2.0), sharey=True)
    ys = np.arange(len(rows))[::-1]

    centre = [float(r["centre_share"]) for r in rows]
    ax1.barh(ys, centre, color=[C["farm"] if c > 0.999 else C["model"] for c in centre], height=0.62)
    ax1.axvline(1.0, color="#666666", lw=0.9, ls="--")
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("Gradientenanteil auf der Zentrumsepoche\n(1.0 = Nachbarepochen ohne Einfluss)", fontsize=8)
    ax1.set_yticks(ys)
    ax1.set_yticklabels([f"{r['model']}  ({r['epochs']} Ep. × {r['channels']} Kan.)" for r in rows], fontsize=8)
    ax1.grid(axis="x", alpha=0.25)
    ax1.tick_params(labelsize=8)

    frac = [float(r["rf_fraction"]) for r in rows]
    ax2.barh(ys, frac, color=[C["farm"] if f < 0.05 else C["model"] for f in frac], height=0.62)
    for y, r, f in zip(ys, rows, frac):
        ax2.text(f, y, f"  {r['rf_samples']}/{r['rf_total_samples']}", va="center", fontsize=7, color="#333333")
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Rezeptives Feld je Ausgabesample\n(Anteil des geladenen Fensters)", fontsize=8)
    ax2.grid(axis="x", alpha=0.25)
    ax2.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=10)
    _finish(fig, out)


def ablation_grid(out: Path, rows: Sequence[dict], title: str) -> None:
    """Grid effects with the null baseline and FARM drawn in.

    A configuration whose error exceeds the null baseline has not learned to
    reconstruct anything, so that line is the one the reader needs.
    """
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    groups: dict[tuple[float, int], list[dict]] = {}
    for r in rows:
        groups.setdefault((float(r["learning_rate"]), int(r["initial_channels"])), []).append(r)
    markers = {0.0003: "o", 0.001: "s"}
    for (lr, ch), items in sorted(groups.items()):
        items = sorted(items, key=lambda r: (float(r["mse_weight"]), float(r["spike_weight"])))
        xs = [f"mse{int(float(r['mse_weight']))}/spk{int(float(r['spike_weight']))}" for r in items]
        ys = [float(r["err_uv"]) for r in items]
        ax.plot(xs, ys, marker=markers.get(lr, "^"), lw=1.2,
                color=C["model"] if ch == 32 else C["alt1"],
                ls="-" if lr == 0.001 else "--",
                label=f"lr {lr:g}, ch {ch}")
    ax.axhline(18.793, color=C["null"], lw=1.2, ls=":", label="Nullausgabe 18.79 µV")
    ax.axhline(110.633, color=C["farm"], lw=1.2, ls="-.", label="FARM (ideal) 110.63 µV")
    ax.set_yscale("log")
    ax.set_ylabel("Rekonstruktionsfehler RMSE (µV, log)", fontsize=9)
    ax.set_xlabel("Objective-Gewichte", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    _finish(fig, out)


def tradeoff(out: Path, points: Sequence[dict], title: str, xlabel: str, ylabel: str,
             log_x: bool = False) -> None:
    """Two-dimensional trade-off with a stated preferred direction.

    ``log_x`` is for cost axes spanning orders of magnitude: with a linear axis a
    two-hundred-fold range collapses every cheap model onto the y-axis, which
    hides exactly the comparison the figure is for.
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    if log_x:
        ax.set_xscale("log")
    for i, p in enumerate(points):
        ax.scatter(float(p["x"]), float(p["y"]), s=float(p.get("size", 70)),
                   color=_colour(p.get("role", ""), i), zorder=3,
                   marker=p.get("marker", "o"), edgecolor="white", linewidth=0.8)
        ax.annotate(p["label"], (float(p["x"]), float(p["y"])), fontsize=7,
                    textcoords="offset points", xytext=(7, 4))
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    _finish(fig, out)


def training_curves(out: Path, series: dict[str, tuple[Sequence[float], Sequence[float]]], title: str,
                    ylabel: str, incomplete: Sequence[str] = ()) -> None:
    """Loss curves on one axis definition, with incomplete runs marked."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for i, (name, (xs, ys)) in enumerate(sorted(series.items())):
        ls = ":" if name in incomplete else "-"
        ax.plot(xs, ys, lw=1.2, ls=ls, color=_colour(name, i),
                label=f"{name}{' (unvollständig)' if name in incomplete else ''}")
    ax.set_xlabel("Epoche", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7)
    _finish(fig, out)


def clean_vs_artifact(
    out: Path,
    traces: dict[str, np.ndarray],
    sfreq: float,
    title: str,
    n: int = 4,
) -> None:
    """What the data actually is: raw, artifact and clean, each at its own scale.

    Three stacked rows per example rather than one shared axis, because the
    artifact is roughly 56 times the EEG here: on a shared axis the clean trace
    is a flat line and the figure says nothing. Each panel therefore states its
    own RMS in the corner, so the reader can reconstruct the ratio the shared
    axis would have shown.
    """
    n = min(n, traces["clean"].shape[0])
    fig, axes = plt.subplots(3, n, figsize=(3.1 * n, 6.2), squeeze=False)
    scale = 1e6
    t = np.arange(traces["clean"].shape[1]) / sfreq * 1e3
    layers = [
        ("noisy", "Rohsignal (clean + Artefakt)", C["noisy"]),
        ("artifact", "Gradientenartefakt (wahr)", C["farm"]),
        ("clean", "EEG (wahr, mit IED)", C["clean"]),
    ]
    for c in range(n):
        spike_mask = traces["spikes"][c] > 0
        for r, (key, label, colour) in enumerate(layers):
            ax = axes[r][c]
            series = traces[key][c] * scale
            rms = float(np.sqrt(np.mean((traces[key][c].astype(np.float64)) ** 2))) * scale
            ax.plot(t, series, color=colour, lw=0.7)
            lim = float(np.nanmax(np.abs(series))) * 1.15 or 1.0
            if spike_mask.any():
                ax.fill_between(t, -lim, lim, where=spike_mask, color="#FFE9A8", alpha=0.75, zorder=0)
            ax.set_ylim(-lim, lim)
            ax.tick_params(labelsize=6)
            ax.text(0.98, 0.94, f"RMS {rms:.1f} µV", transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#DDDDDD", alpha=0.85))
            if c == 0:
                ax.set_ylabel(f"{label}\nµV", fontsize=7)
            if r == 0:
                ax.set_title(f"Beispiel #{int(traces['example_index'][c])}", fontsize=8)
            if r == 2:
                ax.set_xlabel("ms", fontsize=7)
    fig.suptitle(title, fontsize=10)
    _finish(fig, out)


def sensitivity_curves(
    out: Path,
    misalign_rows: Sequence[dict],
    window_rows: Sequence[dict],
    null_rmse: float,
    title: str,
) -> None:
    """Two panels: harmless window shift on the left, harmful misalignment right.

    The right panel is logarithmic and carries the null-output line, because the
    question is not "how much worse" but "does it fall below doing nothing".
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    by_model: dict[str, list[dict]] = {}
    for r in window_rows:
        by_model.setdefault(r["model"], []).append(r)
    for i, (model, items) in enumerate(sorted(by_model.items())):
        items = sorted(items, key=lambda r: r["window_shift_samples"])
        ax1.plot([r["window_shift_samples"] for r in items],
                 [r["model_rmse_uv"] for r in items],
                 marker="o", lw=1.4, color=_colour(model, i), label=model)
    ax1.axhline(null_rmse, color=C["null"], lw=1.2, ls=":", label=f"Nullausgabe {null_rmse:.1f} µV")
    ax1.set_xlabel("Fensterverschiebung (Samples) — Signal und Template bewegen sich gemeinsam", fontsize=8)
    ax1.set_ylabel("Rekonstruktionsfehler (µV)", fontsize=9)
    ax1.set_title("Absolute Position im Fenster: unkritisch", fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.tick_params(labelsize=8)
    ax1.legend(fontsize=7)

    xs = [r["trigger_misalign_samples"] for r in misalign_rows]
    ax2.plot(xs, [r["model_rmse_uv"] for r in misalign_rows], marker="o", lw=1.4,
             color=C["cascade"], label="Kaskade")
    ax2.plot(xs, [r["farm_ideal_rmse_uv"] for r in misalign_rows], marker="s", lw=1.4,
             color=C["farm"], label="FARM (ideal)")
    ax2.axhline(null_rmse, color=C["null"], lw=1.2, ls=":", label=f"Nullausgabe {null_rmse:.1f} µV")
    ax2.set_yscale("log")
    ax2.set_xlabel("Template-Fehlausrichtung (Samples; 1 Sample = 0.24 ms bei 4096 Hz)", fontsize=8)
    ax2.set_ylabel("Rekonstruktionsfehler (µV, log)", fontsize=9)
    ax2.set_title("Template um δ Samples daneben: kritisch", fontsize=9)
    ax2.grid(alpha=0.25, which="both")
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    _finish(fig, out)


# --------------------------------------------------------------- refactoring

def indicator_comparison(out: Path, rows: Sequence[dict], title: str) -> None:
    """Size-normalised engineering indicators for two code bases, side by side.

    Only indicators whose value does not scale with the amount of code are drawn.
    Plotting absolute counts would put 762 against 15 477 lines on one axis and
    invite reading "more code" as "better code", which is the failure mode this
    figure exists to avoid. The selection is by indicator id, made here and
    stated in the caption, not by looking at which bars flatter which arm.
    """
    normalised = {"comment_ratio_pct", "docstring_coverage_pct", "typed_params_pct",
                  "typed_returns_pct", "mean_function_lines", "mean_cyclomatic",
                  "test_functions_per_100_code_lines"}
    picked = [r for r in rows if r["indicator_id"] in normalised]
    if not picked:
        return
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(picked) + 1.6))
    ys = np.arange(len(picked))[::-1]
    h = 0.36
    for y, r in zip(ys, picked):
        ax.barh(y + h / 2, r["legacy_0_1_0"], height=h, color=C["farm"], label="0.1.0" if y == ys[0] else None)
        ax.barh(y - h / 2, r["current_2_0_0_classical_core"], height=h, color=C["model"],
                label="2.0.0 (klassischer Kern)" if y == ys[0] else None)
        ax.text(max(r["legacy_0_1_0"], r["current_2_0_0_classical_core"]) * 1.02, y,
                f"  {r['better_direction']} ist besser", va="center", fontsize=7, color="#555555")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in picked], fontsize=8)
    ax.set_xlabel("Wert (Anteile in %, Dichten je 100 Zeilen, Längen in Zeilen)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)
    _finish(fig, out)


def runtime_memory(out: Path, rows: Sequence[dict], title: str) -> None:
    """Wall time and peak memory per arm, with the repetition spread as error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    labels = [r["label"] for r in rows]
    xs = np.arange(len(rows))
    colours = [C["farm"] if r["arm"] == "legacy" else C["model"] for r in rows]

    axes[0].bar(xs, [r["elapsed_seconds_mean"] for r in rows],
                yerr=[r["elapsed_seconds_sd"] for r in rows], capsize=4, color=colours)
    axes[0].set_ylabel("Laufzeit (s)", fontsize=9)
    axes[0].set_title("Laufzeit, 3 Wiederholungen", fontsize=9)

    axes[1].bar(xs, [r["peak_rss_mib_mean"] for r in rows], color=colours)
    axes[1].set_ylabel("Peak-RSS (MiB)", fontsize=9)
    axes[1].set_title("Spitzenspeicher des Arbeitsprozesses", fontsize=9)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=10)
    _finish(fig, out)


def fidelity_overview(out: Path, per_model: Sequence[dict], register: Sequence[dict],
                      title: str) -> None:
    """Requirements per model package, stacked by disposition.

    Stacked rather than grouped because the interesting quantity is the *share*
    of a package's requirements that needed a documented deviation, and a grouped
    chart hides that behind differing totals.
    """
    order = ["umgesetzt", "bereits konform", "dokumentierte Abweichung",
             "bewusst ausgelassen", "unklassifiziert"]
    colours = {"umgesetzt": C["clean"], "bereits konform": C["model"],
               "dokumentierte Abweichung": C["alt2"], "bewusst ausgelassen": C["farm"],
               "unklassifiziert": "#999999"}
    packages = [r["model_package"] for r in per_model]
    counts = {p: {d: 0 for d in order} for p in packages}
    for r in register:
        pkg, disp = r["model_package"], r.get("disposition", "unklassifiziert")
        if pkg in counts and disp in counts[pkg]:
            counts[pkg][disp] += 1
    if not packages:
        return
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(packages) + 1.8))
    ys = np.arange(len(packages))[::-1]
    left = np.zeros(len(packages))
    for disp in order:
        vals = np.array([counts[p][disp] for p in packages], dtype=float)
        if vals.sum() == 0:
            continue
        ax.barh(ys, vals, left=left, color=colours[disp], label=disp, height=0.62)
        left += vals
    ax.set_yticks(ys)
    ax.set_yticklabels([p.replace("_paper_accurate_edition", " (PA)") for p in packages], fontsize=8)
    ax.set_xlabel("Erfasste Paper-Anforderungen", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)
    _finish(fig, out)


def sensitivity_grid(out: Path, rows: Sequence[dict], version: str, configs: Sequence[str],
                     metrics: Sequence[str], arms: Sequence[str], title: str) -> None:
    """One panel per metric; grouped bars per configuration, one group per arm.

    Significance is marked, but the point of the figure is the *shape*: whether
    the bars keep their order across configurations. A single-configuration bar
    chart cannot show that, which is why the sweep exists.
    """
    index = {(r["config"], r["model_id"], r["metric"]): r for r in rows
             if r["dataset_version"] == version}
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 4.2), squeeze=False)
    width = 0.8 / max(1, len(configs))
    palette = [C["model"], C["clean"], C["alt3"], C["alt2"], C["farm"]]
    for ax, metric in zip(axes[0], metrics):
        xs = np.arange(len(arms))
        for j, cfg in enumerate(configs):
            vals, marks = [], []
            for arm in arms:
                r = index.get((cfg, arm, metric))
                v = r["hodges_lehmann_vs_farm"] if r else np.nan
                vals.append(np.nan if v is None else v)
                marks.append(bool(r and r["significant"]))
            pos = xs + (j - (len(configs) - 1) / 2) * width
            ax.bar(pos, vals, width=width, color=palette[j % len(palette)],
                   label=cfg if metric == metrics[0] else None)
            for x, v, m in zip(pos, vals, marks):
                if m and np.isfinite(v):
                    ax.text(x, v, "*", ha="center", fontsize=7,
                            va="bottom" if v >= 0 else "top", color="#333333")
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=7)
        ax.set_title(metric, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("Hodges-Lehmann-Differenz zu FARM", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, ncol=len(configs), loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title + "   ·   * = nach Holm signifikant", fontsize=10)
    _finish(fig, out)


def seed_spread(out: Path, runs: Sequence[dict], seed_low: float, seed_high: float,
                title: str) -> None:
    """Best validation loss per run, with the seed range drawn as the yardstick.

    The shaded band is what three seeds of one configuration span. Without it a
    reader compares ablations against each other and reads noise as effect; with
    it the question becomes the right one — does this run leave the band?

    Runs whose loss minimises a different function are drawn hollow and are never
    ranked against the rest.
    """
    ranked = sorted((r for r in runs if r.get("best_val_loss") is not None),
                    key=lambda r: r["best_val_loss"])
    if not ranked:
        return
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(ranked) + 1.8))
    ys = np.arange(len(ranked))[::-1]
    if np.isfinite(seed_low) and np.isfinite(seed_high):
        ax.axvspan(seed_low, seed_high, color="#CCCCCC", alpha=0.55, zorder=0,
                   label="Seedstreuung der Paperkonfiguration")
    for y, r in zip(ys, ranked):
        comparable = bool(r.get("val_loss_comparable"))
        colour = C["model"] if comparable else "#BBBBBB"
        ax.scatter([r["best_val_loss"]], [y], s=46, zorder=3,
                   facecolor=colour if comparable else "none",
                   edgecolor=colour if comparable else "#888888", linewidth=1.4)
        suffix = "" if comparable else "  (andere Verlustgewichte)"
        ax.text(r["best_val_loss"], y, f"  {r['best_val_loss']:.4f}{suffix}",
                va="center", fontsize=7, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['run'][:40]}  [{r['family']}]" for r in ranked], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("bester Validierungsverlust (logarithmisch)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)
    _finish(fig, out)
