"""Per-window metrics, bootstrap intervals and paired tests on the unified holdout.

The unified-holdout evaluation stores one **aggregate** number per model and
metric. A single point value cannot say whether two models differ: it carries no
interval, and a ranking read off point values alone invites "model A beats
model B" claims that the data may not support.

Everything needed to fix that is already on disk. Each model's run directory
holds ``predicted_artifact.npy`` for the same 166 holdout windows, and the
holdout split is reproducible from its seed. This tool therefore:

1. **Recomputes the split from the seed** and checks the resulting hash against
   the hash every model recorded — so "one split for all models" is *verified*,
   not asserted.
2. **Recomputes the aggregate metrics** from the stored predictions and checks
   them against each model's ``metrics.json``. A mismatch means the stored
   prediction no longer corresponds to the stored number, and that must surface
   as an error rather than quietly produce a new interval around a stale value.
3. **Computes per-window values**, from which a bootstrap interval and a paired
   comparison between models follow. The unit of independence is the holdout
   window; the 30 electrodes inside a window are replicates and are averaged
   first.

Metric definitions are imported from ``tools/eval_unified_holdout.py`` rather
than restated, so the per-window values cannot drift from the aggregates they
are supposed to decompose.

Usage::

    .venv/bin/python tools/evaluation/holdout_uncertainty.py \\
        --dataset output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz \\
        --out output/model_evaluations/holdout_uncertainty
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from eval_unified_holdout import (  # noqa: E402
    compute_holdout_indices,
    compute_metrics,
    holdout_split_hash,
    load_holdout,
)

EVAL = REPO / "output" / "model_evaluations"
RUN_ID = "holdout_v1"

#: Metrics carried forward to the per-window level, with the preferred direction.
METRICS = {
    "clean_snr_improvement_db": "höher",
    "clean_snr_db_after": "höher",
    "artifact_corr": "höher",
    "residual_error_rms_ratio": "niedriger",
    "rms_recovery_ratio": "Ziel 1.0",
}

#: Tolerance for "the per-window mean reproduces the stored aggregate".
#:
#: Not zero: the stored aggregates were computed on all windows at once, while
#: the per-window values are computed window by window and then averaged. For a
#: ratio or a decibel value those two are not algebraically identical, so an
#: exact match is not the right check — agreement to within this relative
#: tolerance is.
REPRO_RTOL = 0.05

BOOTSTRAP = 10_000


def _naive_6nn(ds: dict[str, np.ndarray]) -> np.ndarray:
    """The 6-neighbour AAS baseline: mean of the context epochs around the centre.

    ``noisy_context`` is ``(N, 7, 30, 512)`` with the centre epoch at index 3, so
    the baseline is the mean over the other six. Recomputing it costs nothing and
    is checked against the stored aggregate like every other arm.
    """
    return np.delete(ds["noisy_context"], 3, axis=1).mean(axis=1)


#: Non-learned arms whose prediction can be reconstructed from the dataset alone.
RECOMPUTABLE = {"aas_naive_6nn": _naive_6nn}


def bootstrap_ci(values: np.ndarray, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap of the mean over windows."""
    v = values[np.isfinite(values)]
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(BOOTSTRAP, v.size))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def wilcoxon(diff: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank (normal approximation, tie-corrected)."""
    d = diff[np.isfinite(diff)]
    d = d[d != 0]
    n = d.size
    if n < 6:
        return float("nan"), float("nan")
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=float)
    sorted_abs = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mean_w = n * (n + 1) / 4.0
    _, counts = np.unique(sorted_abs, return_counts=True)
    tie = float(sum(c ** 3 - c for c in counts))
    var_w = (n * (n + 1) * (2 * n + 1) - tie / 2.0) / 24.0
    if var_w <= 0:
        return float("nan"), float("nan")
    z = (w_plus - mean_w) / np.sqrt(var_w)
    from math import erfc, sqrt
    return w_plus, float(erfc(abs(z) / sqrt(2.0)))


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, tuple[float, bool]]:
    items = [(k, v) for k, v in pvalues.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, tuple[float, bool]] = {k: (float("nan"), False) for k in pvalues}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        out[k] = (adj, adj < alpha)
    return out


def per_window_metrics(noisy, clean, artifact, pred, sfreq) -> dict[str, np.ndarray]:
    """One value per holdout window, electrodes averaged inside the window."""
    n = noisy.shape[0]
    out = {k: np.full(n, np.nan) for k in METRICS}
    for i in range(n):
        m = compute_metrics(noisy[i:i + 1], clean[i:i + 1], artifact[i:i + 1],
                            pred[i:i + 1], sfreq_hz=sfreq)
        for k in METRICS:
            if k == "clean_snr_improvement_db":
                out[k][i] = float(m["clean_snr_db_after"]) - float(m["clean_snr_db_before"])
            elif k in m:
                out[k][i] = float(m[k])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path,
                   default=REPO / "output/niazy_proof_fit_context_512/niazy_proof_fit_context_dataset.npz")
    p.add_argument("--out", type=Path, default=EVAL / "holdout_uncertainty")
    p.add_argument("--reference", default="demucs",
                   help="Model every other model is paired against")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=True) as d:
        n_total = int(d["clean_center"].shape[0])
    indices = compute_holdout_indices(n=n_total)
    split_hash = holdout_split_hash(indices)

    # ---- 1. the split, verified rather than asserted -----------------------
    recorded: dict[str, str] = {}
    for man in sorted(EVAL.glob(f"*/{RUN_ID}/evaluation_manifest.json")):
        cfg = json.loads(man.read_text())["config"]
        recorded[cfg.get("model_id", man.parent.parent.name)] = cfg.get("holdout_split_hash", "—")
    with_hash = {k: v for k, v in recorded.items() if v != "—"}
    agree = {k for k, v in with_hash.items() if v == split_hash}
    split_report = {
        "n_total_windows": n_total,
        "n_holdout_windows": int(indices.size),
        "seed": 42,
        "val_ratio": 0.2,
        "recomputed_hash": split_hash,
        "recorded_hashes": recorded,
        "n_models_with_recorded_hash": len(with_hash),
        "n_models_agreeing_with_recomputed": len(agree),
        "models_without_recorded_hash": sorted(k for k, v in recorded.items() if v == "—"),
        "verdict": ("Ein einziger Split, aus dem Seed reproduziert und von allen "
                    "Modellen mit Hash bestätigt")
        if with_hash and len(agree) == len(with_hash) else "Split-Hashes weichen ab",
    }
    print(f"Split: {indices.size}/{n_total} Fenster, Hash {split_hash} — "
          f"{len(agree)}/{len(with_hash)} Modelle stimmen überein")

    ds = load_holdout(args.dataset, indices)
    sfreq = ds["sfreq"]

    # ---- 2 & 3. per-window values, checked against the stored aggregates ---
    per_model: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict] = []
    repro: list[dict] = []
    for run_dir in sorted(EVAL.glob(f"*/{RUN_ID}")):
        model_id = run_dir.parent.name
        pred_path = run_dir / "predicted_artifact.npy"
        if pred_path.exists():
            pred = np.load(pred_path).astype(np.float32)
        elif model_id in RECOMPUTABLE:
            # A non-learned baseline needs no checkpoint: its prediction is a
            # closed-form function of the context the dataset already stores, so
            # it can be reconstructed exactly instead of being left without an
            # interval. The reproduction check below is what makes that safe.
            pred = RECOMPUTABLE[model_id](ds).astype(np.float32)
        else:
            repro.append({"model_id": model_id, "status": "keine gespeicherte Vorhersage",
                          "metric": "—", "stored": "", "recomputed": "", "within_tolerance": False})
            continue
        if pred.shape != ds["artifact_center"].shape:
            repro.append({"model_id": model_id, "status": f"Form {pred.shape} passt nicht",
                          "metric": "—", "stored": "", "recomputed": "", "within_tolerance": False})
            continue
        stored = json.loads((run_dir / "metrics.json").read_text())["flat_metrics"]
        agg = compute_metrics(ds["noisy_center"], ds["clean_center"], ds["artifact_center"],
                              pred, sfreq_hz=sfreq)
        ok_all = True
        for key in ("artifact_corr", "clean_snr_db_after", "residual_error_rms_ratio"):
            s = stored.get(f"unified_holdout.{key}")
            r = float(agg[key])
            ok = s is not None and np.isclose(s, r, rtol=REPRO_RTOL, atol=1e-9)
            ok_all &= bool(ok)
            repro.append({"model_id": model_id, "status": "reproduziert" if ok else "ABWEICHUNG",
                          "metric": key, "stored": s, "recomputed": r, "within_tolerance": bool(ok)})
        if not ok_all:
            print(f"  [!] {model_id}: gespeicherte Vorhersage reproduziert die gespeicherten "
                  f"Metriken nicht — von der Unsicherheitsrechnung ausgeschlossen")
            continue

        vals = per_window_metrics(ds["noisy_center"], ds["clean_center"],
                                  ds["artifact_center"], pred, sfreq)
        per_model[model_id] = vals
        for k, better in METRICS.items():
            v = vals[k]
            lo, hi = bootstrap_ci(v)
            rows.append({
                "model_id": model_id, "metric": k, "better_is": better,
                "n_windows": int(np.isfinite(v).sum()),
                "mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
                "sd": float(np.nanstd(v, ddof=1)),
                "ci_low": lo, "ci_high": hi,
                "stored_aggregate": stored.get(f"unified_holdout.{k}"),
            })
        print(f"  {model_id:24s} SNR-Gewinn {np.nanmean(vals['clean_snr_improvement_db']):+7.3f} dB "
              f"[{rows[-5]['ci_low']:+.3f}, {rows[-5]['ci_high']:+.3f}]")

    # ---- paired comparisons against one reference -------------------------
    paired: list[dict] = []
    ref = args.reference if args.reference in per_model else (sorted(per_model)[0] if per_model else None)
    if ref:
        for model_id, vals in sorted(per_model.items()):
            if model_id == ref:
                continue
            pvals, cache = {}, {}
            for k in METRICS:
                diff = vals[k] - per_model[ref][k]
                _, p = wilcoxon(diff)
                pvals[k] = p
                lo, hi = bootstrap_ci(diff)
                cache[k] = (float(np.nanmean(diff)), float(np.nanmedian(diff)), lo, hi)
            corrected = holm(pvals)
            for k in METRICS:
                mean_d, med_d, lo, hi = cache[k]
                p_holm, sig = corrected[k]
                paired.append({
                    "model_id": model_id, "reference": ref, "metric": k,
                    "n_windows": int(np.isfinite(vals[k] - per_model[ref][k]).sum()),
                    "mean_difference": mean_d, "median_difference": med_d,
                    "ci_low": lo, "ci_high": hi,
                    "p_raw": pvals[k], "p_holm": p_holm, "significant": sig,
                })

    def _write(name: str, data: list[dict]) -> None:
        if not data:
            return
        with (args.out / name).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    _write("holdout_per_model_intervals.csv", rows)
    _write("holdout_paired_vs_reference.csv", paired)
    _write("holdout_reproduction_check.csv", repro)
    for model_id, vals in per_model.items():
        with (args.out / f"per_window_{model_id}.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["window_index", *METRICS])
            for i, idx in enumerate(indices):
                w.writerow([int(idx), *(f"{vals[k][i]:.10g}" for k in METRICS)])

    (args.out / "holdout_uncertainty.json").write_text(json.dumps({
        "dataset": str(args.dataset.relative_to(REPO)),
        "split": split_report,
        "bootstrap_resamples": BOOTSTRAP,
        "unit_of_inference": "Holdout-Fenster (30 Elektroden je Fenster vorher gemittelt)",
        "reproduction_tolerance_rtol": REPRO_RTOL,
        "reference_model": ref,
        "n_models_with_intervals": len(per_model),
        "metrics": METRICS,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(per_model)} Modelle mit Intervallen, {len(paired)} gepaarte Vergleiche "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
