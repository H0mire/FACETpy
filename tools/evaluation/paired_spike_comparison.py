"""Paired comparison of two correction arms on the same spike examples.

Why this is separate from the evaluation. ``eval_run6_spike_preservation.py``
reports each arm's *average*, and an average cannot support the statement "A is
better than B". Two arms evaluated on the same examples differ example by example;
only those paired differences give a test, an effect size and an interval. The
results protocol requires all three for any superiority claim
(``docs/research/results_evidence_pack_execution_plan.md`` §5.5.3), so this reads
the per-example CSV the evaluation writes and produces exactly that.

What it deliberately does not do: pick a winner, or interpret. It emits the
paired statistics and the exclusions, and leaves the wording to the results note.

Unit of inference
-----------------
The Weg-A builder writes one example per target electrode, so a single injected
spike appears once per electrode — 19 times in the current dataset. Treating
those rows as independent pairs inflates ``n`` by the electrode count and makes
every p-value anti-conservative. The tool therefore reads the ``spike_event_id``
column, averages the electrode replicates of one event before testing, and
reports the event count as the sample size. Window-level numbers are still
emitted, clearly separated, because they are the right unit for a claim about
*windows* — but only the event level enters the multiplicity correction, so a
design with too few events reports "not testable" instead of borrowing
significance from its replicates.

Statistics
----------
* **Wilcoxon signed-rank** on the per-event mean differences, not a t-test: with
  few events and metrics bounded on one side (amplitude ratio, correlation)
  normality is not given and a rank test costs little power.
* **Hodges-Lehmann** median difference as the effect estimate, with a percentile
  bootstrap interval **of that same statistic** — not of the mean, which would
  leave the reported estimate outside its own interval on skewed differences.
  Reported alongside Cliff's delta, which is scale-free and readable for bounded
  metrics.
* **Holm correction** across the metrics compared in one call, since several
  metrics are tested on the same examples.

Usage::

    .venv/bin/python tools/evaluation/paired_spike_comparison.py \\
        --per-example output/model_evaluations/<run>/run6_spike_preservation_per_example.csv \\
        --arm-a model --arm-b aas_ideal \\
        --out output/model_evaluations/<run>/paired_model_vs_farm
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

#: Metrics whose paired difference is meaningful, with the direction that counts
#: as better. Amplitude ratio is special: better means *closer to 1*, so it is
#: compared on |ratio - 1| rather than on the raw value.
METRIC_DIRECTION = {
    "clean_snr_db": "higher",
    "neighborhood_snr_db": "higher",
    "contrast_db": "higher",
    "morphology_corr": "higher",
    "rmse_uv": "lower",
    "amplitude_ratio_abs_error": "lower",
    "latency_drift_abs_samples": "lower",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-example", type=Path, required=True)
    p.add_argument(
        "--per-example-b",
        type=Path,
        default=None,
        help="Take arm B from a second evaluation instead of the same file. Use this to compare "
        "two models against each other. Only legitimate when both evaluations ran on identical "
        "examples: the caller must have established that (e.g. by hashing the reference arrays), "
        "and the shared example_index set is reported so a mismatch cannot pass unnoticed.",
    )
    p.add_argument(
        "--cluster-column",
        default="spike_event_id",
        help="Column naming the unit of independence. 'spike_event_id' for the spike table "
        "(replicates = electrodes of one injected event), 'epoch_id' for the bulk table "
        "(replicates = electrodes of one centre epoch). Rows sharing the value are averaged "
        "before the test.",
    )
    p.add_argument("--arm-a", default="model")
    p.add_argument("--arm-b", default="aas_ideal")
    p.add_argument(
        "--label-a", default=None, help="Name for arm A in output filenames and the printed table."
    )
    p.add_argument("--label-b", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed; recorded in the output")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def load(path: Path, cluster_column: str = "spike_event_id") -> tuple[dict[str, dict[int, dict[str, float]]], dict[int, str]]:
    """arm -> example_index -> metric -> value, plus example_index -> cluster id.

    The cluster id is the ``spike_event_id`` column when the evaluation wrote one.
    It matters because the dataset emits one example per target electrode, so the
    same injected spike appears many times; rows sharing an event are replicates,
    not independent observations. Older CSVs have no such column and fall back to
    one cluster per row, which reproduces the previous (anti-conservative)
    behaviour — and says so in the output.
    """
    out: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    clusters: dict[int, str] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            arm = row.pop("arm")
            idx = int(row.pop("example_index"))
            event = row.pop(cluster_column, None)
            for extra in ("spike_event_id", "epoch_id", "target_channel", "has_spike"):
                row.pop(extra, None)
            clusters[idx] = str(event) if event is not None else f"row{idx}"
            values = {k: (float(v) if v not in ("", "nan") else float("nan")) for k, v in row.items()}
            # Derived comparables: "better" for these means closer to the ideal,
            # not larger, so the difference is taken on the distance to it.
            if "amplitude_ratio" in values:
                values["amplitude_ratio_abs_error"] = abs(values["amplitude_ratio"] - 1.0)
            if "latency_drift_samples" in values:
                values["latency_drift_abs_samples"] = abs(values["latency_drift_samples"])
            values.pop("", None)
            out[arm][idx] = values
    return out, clusters


def wilcoxon_signed_rank(diff: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank statistic and p-value (normal approximation).

    Zeros are dropped (Wilcoxon's own convention) and ties get average ranks with
    the standard tie correction, so a metric with repeated values is not silently
    given an over-confident p-value.
    """
    d = diff[np.isfinite(diff)]
    d = d[d != 0]
    n = d.size
    if n < 6:
        return float("nan"), float("nan")
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=np.float64)
    absd = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        ranks[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    signs = np.sign(d)[order]
    w_plus = float(ranks[signs > 0].sum())
    w_minus = float(ranks[signs < 0].sum())
    w = min(w_plus, w_minus)

    mean = n * (n + 1) / 4.0
    _, counts = np.unique(absd, return_counts=True)
    tie_term = float(np.sum(counts**3 - counts))
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return w, float("nan")
    z = (w - mean + 0.5) / np.sqrt(var)          # continuity-corrected
    p = 2.0 * 0.5 * math_erfc(abs(z) / np.sqrt(2.0))
    return w, float(min(1.0, p))


def math_erfc(x: float) -> float:
    """erfc without SciPy, so the tool has no extra dependency on a GPU host."""
    import math

    return math.erfc(x)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b) over all pairs; scale-free, in [-1, 1]."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    greater = float((a[:, None] > b[None, :]).sum())
    less = float((a[:, None] < b[None, :]).sum())
    return (greater - less) / (a.size * b.size)


def hodges_lehmann(diff: np.ndarray) -> float:
    """Median of pairwise Walsh averages — the estimator matching Wilcoxon."""
    d = diff[np.isfinite(diff)]
    if d.size == 0:
        return float("nan")
    walsh = (d[:, None] + d[None, :]) / 2.0
    return float(np.median(walsh[np.triu_indices_from(walsh)]))


#: Above this many paired differences the Hodges-Lehmann bootstrap is refused.
#: Each resample needs an n x n Walsh matrix, so the cost grows with n^2 per
#: resample; at n = 4860 that is 189 MB of work 10 000 times over. The inferential
#: level (one value per independent cluster) stays far below the cap.
HL_BOOTSTRAP_MAX_N = 400


def bootstrap_ci(
    diff: np.ndarray, n: int, seed: int, alpha: float, statistic: str = "hodges_lehmann"
) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling the statistic it is paired with.

    ``statistic="hodges_lehmann"`` matches the reported effect estimate. An
    earlier version always bootstrapped the *mean* while reporting the HL median,
    which put the point estimate on the interval boundary whenever the
    differences were skewed — an interval that excludes its own estimator is not
    usable.

    ``statistic="mean"`` is used for the descriptive window level, where n is in
    the thousands and the HL bootstrap is refused on cost grounds. The caller
    labels those columns as mean-based so the two are never mixed up.
    """
    d = diff[np.isfinite(diff)]
    if d.size < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if statistic == "mean":
        stats = rng.choice(d, size=(n, d.size), replace=True).mean(axis=1)
        lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)
    if d.size > HL_BOOTSTRAP_MAX_N:
        return float("nan"), float("nan")
    # Chunked: all resamples at once would be gigabytes of Walsh matrices.
    stats = np.empty(n, dtype=np.float64)
    chunk = max(1, min(200, int(8e6 // max(d.size ** 2, 1))))
    iu = np.triu_indices(d.size)
    done = 0
    while done < n:
        take = min(chunk, n - done)
        draws = rng.choice(d, size=(take, d.size), replace=True)
        walsh = (draws[:, :, None] + draws[:, None, :]) / 2.0
        stats[done:done + take] = np.median(walsh[:, iu[0], iu[1]], axis=1)
        done += take
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def holm(pvalues: dict[str, float], alpha: float) -> dict[str, dict[str, float | bool]]:
    """Holm-Bonferroni across the metrics tested together in one call."""
    valid = {k: v for k, v in pvalues.items() if np.isfinite(v)}
    ordered = sorted(valid.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered):
        adjusted = min(1.0, max(running, (m - rank) * p))
        running = adjusted
        out[key] = {"p_raw": p, "p_holm": adjusted, "significant": bool(adjusted < alpha)}
    for key, p in pvalues.items():
        if key not in out:
            out[key] = {"p_raw": p, "p_holm": float("nan"), "significant": False}
    return out


def main() -> None:
    args = parse_args()
    data_a, clusters_a = load(args.per_example, args.cluster_column)
    if args.per_example_b:
        data_b, clusters_b = load(args.per_example_b, args.cluster_column)
    else:
        data_b, clusters_b = data_a, clusters_a
    clusters = {**clusters_b, **clusters_a}
    if args.arm_a not in data_a:
        raise SystemExit(f"arm '{args.arm_a}' not in {args.per_example} (have: {sorted(data_a)})")
    if args.arm_b not in data_b:
        src = args.per_example_b or args.per_example
        raise SystemExit(f"arm '{args.arm_b}' not in {src} (have: {sorted(data_b)})")
    # Labels only rename; the underlying arm keys stay in the manifest so the
    # comparison can always be traced back to the two source evaluations.
    label_a = args.label_a or args.arm_a
    label_b = args.label_b or args.arm_b
    if label_a == label_b:
        raise SystemExit("--label-a and --label-b must differ; the output columns would collide")
    data = {label_a: data_a[args.arm_a], label_b: data_b[args.arm_b]}
    args.arm_a, args.arm_b = label_a, label_b

    shared = sorted(set(data[args.arm_a]) & set(data[args.arm_b]))
    only_a = sorted(set(data[args.arm_a]) - set(data[args.arm_b]))
    only_b = sorted(set(data[args.arm_b]) - set(data[args.arm_a]))
    if not shared:
        raise SystemExit("no shared example_index between the two arms — nothing is paired")

    event_ids = sorted({clusters[i] for i in shared})
    n_events = len(event_ids)

    rows: list[dict] = []
    pvalues: dict[str, float] = {}
    for metric, direction in METRIC_DIRECTION.items():
        if metric not in data[args.arm_a][shared[0]]:
            continue
        a = np.array([data[args.arm_a][i][metric] for i in shared])
        b = np.array([data[args.arm_b][i][metric] for i in shared])
        paired = np.isfinite(a) & np.isfinite(b)
        diff = a[paired] - b[paired]

        # Window level: one pair per channel-window. Correct for a claim about
        # windows, wrong for a claim about spikes, because the same spike appears
        # once per electrode.
        w_win, p_win = wilcoxon_signed_rank(diff)
        lo_win, hi_win = bootstrap_ci(diff, args.bootstrap, args.seed, args.alpha, statistic="mean")

        # Event level: collapse the electrode replicates of one spike to their
        # mean difference first, then test. This is the unit of independence, so
        # it is the level that can support a claim about spikes.
        ev_diff = []
        for event in event_ids:
            members = [i for i in shared if clusters[i] == event]
            d = np.array([data[args.arm_a][i][metric] - data[args.arm_b][i][metric] for i in members])
            d = d[np.isfinite(d)]
            if d.size:
                ev_diff.append(float(d.mean()))
        ev_diff_arr = np.asarray(ev_diff, dtype=np.float64)
        w_ev, p_ev = wilcoxon_signed_rank(ev_diff_arr)
        lo_ev, hi_ev = bootstrap_ci(ev_diff_arr, args.bootstrap, args.seed, args.alpha)
        # The event-level p-value is the one that enters the multiplicity
        # correction, so a non-testable design cannot pick up significance by
        # falling back to the window level.
        pvalues[metric] = p_ev
        rows.append({
            "metric": metric,
            "better_is": direction,
            "n_events": n_events,
            "n_paired_windows": int(paired.sum()),
            "n_dropped_nonfinite": int((~paired).sum()),
            f"median_{args.arm_a}": float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan"),
            f"median_{args.arm_b}": float(np.nanmedian(b)) if np.isfinite(b).any() else float("nan"),
            "event_mean_difference": float(ev_diff_arr.mean()) if ev_diff_arr.size else float("nan"),
            "event_hodges_lehmann_difference": hodges_lehmann(ev_diff_arr),
            "event_ci_low": lo_ev,
            "event_ci_high": hi_ev,
            "event_wilcoxon_w": w_ev,
            "event_testable": bool(np.isfinite(p_ev)),
            "window_mean_difference": float(diff.mean()) if diff.size else float("nan"),
            "window_hodges_lehmann_difference": hodges_lehmann(diff),
            "window_mean_ci_low": lo_win,
            "window_mean_ci_high": hi_win,
            "window_p_raw": p_win,
            "window_wilcoxon_w": w_win,
            "cliffs_delta_windows": (
                cliffs_delta(a, b) if paired.sum() <= HL_BOOTSTRAP_MAX_N else float("nan")
            ),
        })

    corrected = holm(pvalues, args.alpha)
    for row in rows:
        row.update({k: v for k, v in corrected[row["metric"]].items()})

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.cluster_column == "spike_event_id" else f"_{args.cluster_column}"
    csv_path = args.out / f"paired_{args.arm_a}_vs_{args.arm_b}{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "per_example_source": str(args.per_example),
        "per_example_source_b": str(args.per_example_b) if args.per_example_b else str(args.per_example),
        "arm_a": args.arm_a,
        "arm_b": args.arm_b,
        "n_shared_examples": len(shared),
        "example_indices": shared,
        "n_independent_events": n_events,
        "event_ids": event_ids,
        "windows_per_event": {e: sum(1 for i in shared if clusters[i] == e) for e in event_ids},
        "cluster_column": args.cluster_column,
        "unit_of_inference": f"{args.cluster_column} (electrode replicates averaged first)",
        "clustering_warning": (
            "The per-example CSV carries no spike_event_id; every row was treated as its own "
            "event, which reproduces the earlier anti-conservative behaviour."
            if all(e.startswith("row") for e in event_ids) else
            f"{len(shared)} channel-windows collapse to {n_events} independent spike events. "
            "Window-level statistics are reported for completeness but must not be quoted as "
            "evidence about spikes."
        ),
        "excluded_only_in_a": only_a,
        "excluded_only_in_b": only_b,
        "test": "Wilcoxon signed-rank, two-sided, normal approximation with tie and continuity "
                "correction, applied to per-event mean differences",
        "effect_estimate": "Hodges-Lehmann median of per-event mean differences; window-level "
                           "estimates and Cliff's delta reported alongside as descriptive only",
        "interval": f"event level: percentile bootstrap of the Hodges-Lehmann estimate "
                    f"({args.bootstrap} resamples, seed {args.seed}), i.e. the same statistic as "
                    f"the reported effect. Window level: bootstrap of the mean difference, "
                    f"labelled window_mean_ci_*, descriptive only.",
        "multiplicity": f"Holm-Bonferroni across {len(pvalues)} metrics, alpha {args.alpha}",
        "caveat": "amplitude_ratio and latency_drift are compared as distance to their ideal "
                  "(1.0 and 0 respectively), not as raw values",
        "rows": rows,
    }
    (args.out / f"paired_{args.arm_a}_vs_{args.arm_b}{suffix}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    width = max(len(r["metric"]) for r in rows)
    unit = "Spike-Ereignissen" if args.cluster_column == "spike_event_id" else "Clustern"
    print(f"{args.arm_a} vs {args.arm_b}   {len(shared)} Kanalfenster aus "
          f"{n_events} unabhängigen {unit} ({args.cluster_column})")
    if n_events < 6:
        print(f"  ACHTUNG: {n_events} Ereignisse reichen für keinen Vorzeichenrangtest "
              "(Mindestzahl 6). Ereignisebene ist nicht testbar; die Fensterwerte unten sind "
              "deskriptiv und dürfen nicht als Spike-Evidenz zitiert werden.")
    print(f"{'metric':{width}s} {'ev':>3s} {'median A':>10s} {'median B':>10s} "
          f"{'HL(Ereig.)':>11s} {'95% CI (Ereig.)':>22s} {'p_holm':>8s} | "
          f"{'HL(Fenster)':>12s} {'p(Fenster)':>10s}")
    print("-" * (width + 100))
    for r in rows:
        ci = f"[{r['event_ci_low']:.3g}, {r['event_ci_high']:.3g}]"
        p_ev = "n/a" if not r["event_testable"] else f"{r['p_holm']:.4g}"
        print(f"{r['metric']:{width}s} {r['n_events']:>3d} {r[f'median_{args.arm_a}']:>10.4g} "
              f"{r[f'median_{args.arm_b}']:>10.4g} {r['event_hodges_lehmann_difference']:>11.4g} "
              f"{ci:>22s} {p_ev:>8s} | {r['window_hodges_lehmann_difference']:>12.4g} "
              f"{r['window_p_raw']:>10.3g}"
              f"{'  *' if r['significant'] else ''}")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
