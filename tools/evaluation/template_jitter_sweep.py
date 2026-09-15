"""Trigger jitter at template *estimation* time — the realistic misalignment.

An earlier sweep displaced a correctly estimated template by delta samples and
reported a factor-17 collapse. That is a tautology about template subtraction:
subtract any template at the wrong offset and it adds artifact energy instead of
removing it. It says nothing about FARM, and comparing it against a direct model
that has no template is empty.

The failure that can actually happen in a pipeline is different. If trigger
detection is imprecise, the epochs entering the average are misaligned, so the
**template itself** comes out smeared and attenuated — it subtracts too little
rather than in the wrong place. This tool measures that, with ground truth:

1. Take the stored artifact epochs of each example's context.
2. Build the template as the mean of the neighbour epochs, each cropped at
   ``guard + delta_e`` with ``delta_e`` drawn per epoch from a normal of the
   given standard deviation. ``sd = 0`` is the aligned reference.
3. Correct ``clean + artifact`` with that template and compare against the known
   clean.

Two conditions are swept side by side so the two failure modes can be read
against each other:

* ``estimation`` — per-epoch jitter, template smeared, applied at the right place.
  This is the realistic one.
* ``application`` — all epochs shifted coherently, so the template is *correct*
  but subtracted delta samples off. This is the earlier, artificial condition,
  kept only as the upper bound it is.

The template's own attenuation and its correlation with the aligned template are
reported too, because "smeared" is a claim about the template and should be
measured on the template, not only inferred from the error.

Usage::

    .venv/bin/python tools/evaluation/template_jitter_sweep.py \\
        --dataset output/weg_a_farm_v9b_bcgfree_512/weg_a_spatiotemporal_dataset.npz \\
        --out output/model_evaluations/template_jitter
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

#: Jitter magnitudes in samples (standard deviation for the estimation arm,
#: absolute offset for the application arm). 4096 Hz, so 1 sample = 0.244 ms.
SWEEP = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def wilcoxon(diff: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank (normal approximation, tie-corrected)."""
    d = diff[np.isfinite(diff)]
    d = d[d != 0]
    n = d.size
    if n < 6:
        return float("nan"), float("nan")
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=float)
    sa = np.abs(d)[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    w_plus = float(ranks[d > 0].sum())
    mean_w = n * (n + 1) / 4.0
    _, counts = np.unique(sa, return_counts=True)
    tie = float(sum(c ** 3 - c for c in counts))
    var = (n * (n + 1) * (2 * n + 1) - tie / 2.0) / 24.0
    if var <= 0:
        return float("nan"), float("nan")
    from math import erfc, sqrt
    z = (w_plus - mean_w) / np.sqrt(var)
    return w_plus, float(erfc(abs(z) / sqrt(2.0)))


def hodges_lehmann(x: np.ndarray) -> float:
    v = x[np.isfinite(x)]
    if v.size == 0:
        return float("nan")
    if v.size > 400:
        rng = np.random.default_rng(0)
        v = v[rng.choice(v.size, 400, replace=False)]
    pair = (v[:, None] + v[None, :]) / 2.0
    iu = np.triu_indices(v.size, k=0)
    return float(np.median(pair[iu]))


def bootstrap_ci(x: np.ndarray, seed: int = 0) -> tuple[float, float]:
    v = x[np.isfinite(x)]
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(4000, v.size))
    means = v[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def build_template(epochs: np.ndarray, guard: int, core: int, offsets: np.ndarray) -> np.ndarray:
    """Mean of the epochs, each cropped at ``guard + offset_e``.

    ``epochs`` is ``(n_examples, n_epochs, length)``; ``offsets`` is
    ``(n_examples, n_epochs)`` in samples. Cropping rather than rolling is what a
    real trigger error does: the epoch window sits earlier or later in the
    continuous signal, it does not wrap around.
    """
    n_ex, n_ep, length = epochs.shape
    out = np.zeros((n_ex, core), dtype=np.float64)
    starts = np.clip(guard + offsets, 0, length - core).astype(np.int64)
    for e in range(n_ep):
        s = starts[:, e]
        rows = np.arange(n_ex)[:, None]
        cols = s[:, None] + np.arange(core)[None, :]
        out += epochs[rows, e, cols]
    return out / n_ep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-centre", action="store_true", default=True,
                   help="Average only the neighbour epochs, never the centre epoch itself "
                        "(the centre epoch is what is being corrected).")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=True) as b:
        guard = int(b["guard_samples"][0])
        core = int(b["core_samples"][0])
        n_ctx = int(b["context_epochs"][0])
        split = b["example_split"]
        val = np.flatnonzero(split == args.split)
        # Target electrode is channel 0 of the context (builder puts it first).
        art_ctx = b["artifact_context"][val][:, :, 0, :].astype(np.float64)
        clean = b["clean_center"][val][:, 0, guard:guard + core].astype(np.float64)
        artifact = b["artifact_center"][val][:, 0, guard:guard + core].astype(np.float64)
        epoch_id = b["center_epoch_index"][val].astype(np.int64)

    centre = n_ctx // 2
    keep = [e for e in range(n_ctx) if e != centre] if args.exclude_centre else list(range(n_ctx))
    epochs = art_ctx[:, keep, :]
    n_ex, n_ep, length = epochs.shape
    noisy = clean + artifact

    rng = np.random.default_rng(args.seed)
    zero = np.zeros((n_ex, n_ep), dtype=np.int64)
    ref_template = build_template(epochs, guard, core, zero)
    ref_err = noisy - ref_template - clean
    ref_rms = np.sqrt(np.mean(ref_err ** 2, axis=1)) * 1e6
    null_rms = np.sqrt(np.mean(clean ** 2, axis=1)) * 1e6

    # How good is the hand-built reference template compared with the bundle's own
    # AAS+PCA estimate? Stated, not hidden: a 6-epoch mean is a much weaker
    # template than FARM's, so absolute error levels here are NOT FARM's. What
    # transfers is the relative degradation between the two jitter arms.
    with np.load(args.dataset, allow_pickle=True) as b2:
        bundle_t = b2["artifact_center_template"][val][:, 0, guard:guard + core].astype(np.float64)
    bundle_rms = np.sqrt(np.mean((noisy - bundle_t - clean) ** 2, axis=1)) * 1e6
    quality = {
        "hand_built_6_epoch_mean_error_uv": float(ref_rms.mean()),
        "bundle_aas_pca_template_error_uv": float(bundle_rms.mean()),
        "null_output_error_uv": float(null_rms.mean()),
        "corr_hand_built_to_true_artifact": float(np.corrcoef(
            ref_template.ravel(), artifact.ravel())[0, 1]),
        "corr_bundle_to_true_artifact": float(np.corrcoef(
            bundle_t.ravel(), artifact.ravel())[0, 1]),
        "caveat": "Der Referenzarm ist ein Mittel ueber 6 Nachbarepochen — alles, was der "
                  "Datensatz an Epochen speichert. FARM mittelt ueber 25-30 Epochen mit "
                  "Korrelationsschwelle und erreicht ein deutlich besseres Template. "
                  "Absolute Fehlerhoehen hier sind daher nicht FARMs. Weil weniger Epochen "
                  "jeder fehlausgerichteten Epoche mehr Gewicht geben, ist die hier "
                  "gemessene Verschlechterung eine OBERE Schranke fuer ein 25-Epochen-Mittel.",
    }
    print(f"Referenz-Template (6-Epochen-Mittel): {ref_rms.mean():.1f} uV, r zum Artefakt "
          f"{quality['corr_hand_built_to_true_artifact']:.4f}")
    print(f"Buendel-Template (AAS+PCA):           {bundle_rms.mean():.1f} uV, r "
          f"{quality['corr_bundle_to_true_artifact']:.4f}")

    print(f"{n_ex} Validierungsbeispiele, {n_ep} Nachbarepochen je Template, "
          f"core {core}, guard {guard}")
    print(f"ausgerichtet: {ref_rms.mean():.2f} uV   Nullausgabe: {null_rms.mean():.2f} uV\n")
    print(f"{'Arm':12s}{'Jitter':>9}{'ms':>7}{'Fehler µV':>11}{'HL vs 0':>10}"
          f"{'95% KI':>20}{'p':>10}{'Templ.-RMS':>12}{'r zu ref':>10}")
    print("-" * 101)

    rows = []
    per_example: list[dict] = []
    for arm in ("estimation", "application"):
        for mag in SWEEP:
            if arm == "estimation":
                off = np.rint(rng.normal(0.0, mag, size=(n_ex, n_ep))).astype(np.int64) if mag > 0 else zero
            else:
                # Same jitter magnitude, but drawn ONCE per example and applied to
                # every epoch. The template then comes out correct (all epochs
                # agree) yet sits that many samples off. Drawing from the same
                # distribution as the estimation arm is what makes the two
                # comparable: identical trigger error, differing only in whether
                # it is coherent within the example or independent per epoch.
                draw = (np.rint(rng.normal(0.0, mag, size=(n_ex, 1))).astype(np.int64)
                        if mag > 0 else np.zeros((n_ex, 1), dtype=np.int64))
                off = np.repeat(draw, n_ep, axis=1)
            off = np.clip(off, -guard, guard)
            templ = build_template(epochs, guard, core, off)
            err = noisy - templ - clean
            rms = np.sqrt(np.mean(err ** 2, axis=1)) * 1e6
            diff = rms - ref_rms
            hl = hodges_lehmann(diff)
            lo, hi = bootstrap_ci(diff)
            _, pval = wilcoxon(diff)
            t_rms = np.sqrt(np.mean(templ ** 2, axis=1))
            r_ref = np.sqrt(np.mean(ref_template ** 2, axis=1))
            ratio = float(np.mean(t_rms / np.maximum(r_ref, 1e-30)))
            corr = float(np.mean([np.corrcoef(templ[i], ref_template[i])[0, 1]
                                  for i in range(min(n_ex, 400))]))
            rows.append({
                "arm": arm,
                "arm_meaning": ("Jitter je Epoche VOR der Mittelung — Template verwischt"
                                if arm == "estimation"
                                else "Template korrekt, aber versetzt angewandt"),
                "jitter_samples": mag,
                "jitter_ms": round(mag / 4.096, 4),
                "n_examples": int(n_ex),
                "mean_error_uv": float(rms.mean()),
                "median_error_uv": float(np.median(rms)),
                "hl_vs_aligned_uv": hl,
                "ci_low": lo, "ci_high": hi, "p_raw": pval,
                "worse_than_null_output": bool(rms.mean() > null_rms.mean()),
                "template_rms_ratio_to_aligned": round(ratio, 4),
                "template_corr_to_aligned": round(corr, 5),
            })
            print(f"{arm:12s}{mag:>9.1f}{mag / 4.096:>7.2f}{rms.mean():>11.2f}{hl:>10.2f}"
                  f"{f'[{lo:.2f}, {hi:.2f}]':>20}{pval:>10.2g}{ratio:>12.4f}{corr:>10.4f}")
            for i in range(n_ex):
                per_example.append({"arm": arm, "jitter_samples": mag,
                                    "epoch_id": int(epoch_id[i]),
                                    "error_uv": f"{rms[i]:.10g}",
                                    "aligned_error_uv": f"{ref_rms[i]:.10g}"})
        print()

    with (args.out / "template_jitter_sweep.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (args.out / "template_jitter_per_example.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_example[0]))
        w.writeheader()
        w.writerows(per_example)
    (args.out / "template_jitter_sweep.json").write_text(json.dumps({
        "dataset": str(args.dataset),
        "split": args.split,
        "n_examples": int(n_ex),
        "n_neighbour_epochs_per_template": int(n_ep),
        "centre_epoch_excluded": bool(args.exclude_centre),
        "core_samples": core, "guard_samples": guard,
        "sfreq_hz": 4096.0,
        "aligned_error_uv": float(ref_rms.mean()),
        "null_output_error_uv": float(null_rms.mean()),
        "reference_template_quality": quality,
        "template_definition": "Mittel der Nachbarepochen des Kontexts, je Epoche bei "
                               "guard + offset zugeschnitten",
        "arms": {
            "estimation": "Offset je Epoche aus N(0, sd) VOR der Mittelung — das Template wird "
                          "verwischt und gedämpft, aber an der richtigen Stelle subtrahiert. "
                          "Der realistische Ausfall bei ungenauer Triggererkennung.",
            "application": "Alle Epochen um denselben Betrag verschoben — das Template ist "
                           "korrekt, sitzt aber versetzt. Gleiche Jitterverteilung wie der "
                           "estimation-Arm, nur einmal je Beispiel gezogen statt je Epoche.",
        },
        "rows": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out / 'template_jitter_sweep.csv'}")


if __name__ == "__main__":
    main()
