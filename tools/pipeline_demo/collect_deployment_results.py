"""Gather every deployment-edition run from every pod into one table.

The runs are spread over four machines, several queues and three seeds, and each
one leaves a ``training.jsonl``. Reading them by hand is how a number ends up in
a thesis without its provenance, so this collects them mechanically: one row per
(family, seed), the best epoch and whether the *budget* stopped the run rather
than early stopping — because a budget-limited number is a lower bound, not a
measurement, and must not be averaged in as though it were one.

Usage::

    python tools/pipeline_demo/collect_deployment_results.py \
        --pods facetpod facetpod2 facetpod3 facetpod4 \
        --out output/deployment_editions/results.json

With ``--local`` it reads ``training_output/`` on this machine instead.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Reference points on the deployment objective, for reading the table.
PERFECT = -1.0
DELETION_FLOOR = 3.046
DO_NOTHING = 43.06

_RUN = re.compile(r"^(?P<family>[a-z0-9]+)deployment.*?(?:seed(?P<seed>\d+))?$")

#: Run-directory prefix -> family id. The directory name is the model_name
#: lowercased with underscores stripped, so it cannot be split back reliably.
FAMILIES = ["cascadedcontextdae", "cascadeddae", "convtasnet", "denoisemamba",
            "demucs", "dhctganv2", "dhctgan", "dpae", "d4pm", "icunet",
            "nestedgan", "sepformer", "stgnn", "vitspectrogram"]
PRETTY = {"cascadedcontextdae": "cascaded_context_dae", "cascadeddae": "cascaded_dae",
          "convtasnet": "conv_tasnet", "denoisemamba": "denoise_mamba",
          "demucs": "demucs", "dhctganv2": "dhct_gan_v2", "dhctgan": "dhct_gan",
          "dpae": "dpae", "d4pm": "d4pm", "icunet": "ic_unet",
          "nestedgan": "nested_gan", "sepformer": "sepformer", "stgnn": "st_gnn",
          "vitspectrogram": "vit_spectrogram"}

COLLECTOR = r'''
import glob, json, os, sys
out = []
for d in sorted(glob.glob(sys.argv[1])):
    jl = os.path.join(d, "training.jsonl")
    if not os.path.exists(jl):
        continue
    try:
        rows = [json.loads(l) for l in open(jl) if l.strip()]
    except Exception:
        continue
    if not rows:
        continue
    scored = [r for r in rows if isinstance(r.get("val_loss"), (int, float))]
    if not scored:
        continue
    best = min(scored, key=lambda r: r["val_loss"])
    out.append({
        "run": os.path.basename(d),
        "n_epochs": len(rows),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "terms": {k[4:]: v for k, v in best.items() if k.startswith("val_") and k != "val_loss"},
        "exports": sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "exports", "*.ts"))),
    })
print(json.dumps(out))
'''


def _collect(pod: str | None, pattern: str) -> list[dict]:
    if pod is None:
        proc = subprocess.run([sys.executable, "-c", COLLECTOR, pattern],
                              capture_output=True, text=True, cwd=REPO)
    else:
        # Ship the collector as a file rather than as `python -c "..."`: the
        # shell rewrites the embedded newlines and the remote interpreter then
        # sees a one-line script that fails on the first escape.
        upload = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod,
             "cat > /tmp/facet_collect.py"],
            input=COLLECTOR, capture_output=True, text=True)
        if upload.returncode != 0:
            print(f"  ! {pod}: {upload.stderr.strip()[:120]}", file=sys.stderr)
            return []
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", pod,
             f"cd /workspace/facetpy && python /tmp/facet_collect.py '{pattern}'"],
            capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"  ! {pod or 'local'}: {proc.stderr.strip()[:120]}", file=sys.stderr)
        return []
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "[]"
    try:
        rows = json.loads(line)
    except json.JSONDecodeError:
        return []
    for r in rows:
        r["pod"] = pod or "local"
    return rows


def _identify(run: str) -> tuple[str | None, int, str]:
    """Family, seed and variant from a run directory name.

    The variant matters as much as the family. A context-axis run and a
    base-contract run of the same family are *different experiments* — one has a
    comparison axis and the other does not — and averaging them would erase the
    only thing the context study measures. Same for the ST-GNN kernel widths and
    the long-budget runs.
    """
    name = run.split("_")[0]
    for prefix in FAMILIES:                      # longest names first, see FAMILIES order
        if not name.startswith(prefix + "deployment"):
            continue
        rest = name[len(prefix + "deployment"):]
        seed = 42
        m = re.search(r"seed(\d+)", rest)
        if m:
            seed = int(m.group(1))
            rest = rest[:m.start()] + rest[m.end():]
        rest = rest.replace("niazyprooffit", "")
        # Strip the budget marker once, then read what kind of run it was. Doing
        # it in the other order produced "k129h16long_long".
        long_budget = rest.endswith("long")
        if long_budget:
            rest = rest[: -len("long")]

        if rest.startswith("ctx"):
            variant = f"ctx_{rest[3:]}" if rest[3:] else "ctx"
        elif rest.startswith(("k", "bn")):       # kernel study, bottleneck arms
            variant = rest
        else:
            variant = "base"
        if long_budget:
            variant = "long" if variant == "base" else f"{variant}_long"
        return PRETTY[prefix], seed, variant
    return None, 42, "base"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pods", nargs="*", default=["facetpod", "facetpod2", "facetpod3", "facetpod4"])
    ap.add_argument("--local", action="store_true", help="read training_output/ here instead")
    ap.add_argument("--pattern", default="training_output/*deployment*")
    ap.add_argument("--out", type=Path, default=REPO / "output/deployment_editions/results.json")
    args = ap.parse_args()

    raw: list[dict] = []
    for pod in ([None] if args.local else args.pods):
        found = _collect(pod, args.pattern)
        print(f"{pod or 'local':12s} {len(found)} Läufe")
        raw.extend(found)

    rows = []
    for r in raw:
        family, seed, variant = _identify(r["run"])
        if family is None:
            continue
        running = not r["exports"]
        rows.append({**r, "family": family, "seed": seed, "variant": variant,
                     # No export means the run has not reached its export step,
                     # so it is still going. Without this distinction a job that
                     # is merely *in progress* looks identical to one the epoch
                     # ceiling cut off, and both would be averaged in.
                     "running": running,
                     "budget_limited": (not running) and r["best_epoch"] >= r["n_epochs"],
                     "has_export": bool(r["exports"])})

    # vit_spectrogram seed 42 exists twice and the two are *different
    # architectures*: the first run is the base magnitude-only inpainter, whose
    # ceiling turned out to lie below the deletion floor, and the second is the
    # complex-mask rewrite. Averaging them would mix the evidence for a finding
    # with the result that replaced it, so the old one is named and dropped here
    # rather than silently outvoted by epoch count.
    rows = [r for r in rows if not (
        r["family"] == "vit_spectrogram" and r["best_val_loss"] > 10.0)]

    # One run per (family, seed, variant): later queues re-ran a few, and the
    # duplicate with fewer epochs is the one that was interrupted. A finished
    # run always beats an unfinished one for the same key.
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["family"], r["seed"], r["variant"])
        prior = best.get(key)
        if prior is None or (prior["running"], prior["n_epochs"]) < (r["running"], r["n_epochs"]):
            if prior is None or (not r["running"], r["n_epochs"]) > (not prior["running"], prior["n_epochs"]):
                best[key] = r
    rows = sorted(best.values(), key=lambda r: (r["family"], r["variant"], r["seed"]))

    print(f"\n{'Familie':22s} {'Var':5s} {'Seed':>4s} {'best':>9s} {'Epoche':>10s} "
          f"{'e_ratio':>8s} {'Export':>7s}")
    print("-" * 78)
    for r in rows:
        er = r["terms"].get("energy_ratio", float("nan"))
        print(f"{r['family']:22s} {r['variant']:5s} {r['seed']:4d} {r['best_val_loss']:9.4f} "
              f"{r['best_epoch']:5d}/{r['n_epochs']:<4d} {er:8.3f} "
              f"{'ja' if r['has_export'] else 'NEIN':>7s}"
              + ("  LÄUFT" if r["running"] else "  BUDGET" if r["budget_limited"] else ""))

    summary = {}
    for family in sorted({r["family"] for r in rows}):
        vals = [r["best_val_loss"] for r in rows
                if r["family"] == family and r["variant"] == "base"
                and not r["budget_limited"] and not r["running"]]
        if not vals:
            continue
        summary[family] = {
            "n_seeds": len(vals),
            "mean": statistics.fmean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "values": sorted(vals),
        }

    print(f"\n{'Familie':22s} {'n':>2s} {'Mittel':>9s} {'SD':>8s}   Werte")
    print("-" * 78)
    for family, s in sorted(summary.items(), key=lambda kv: kv[1]["mean"]):
        sd = f"{s['sd']:8.4f}" if s["sd"] is not None else "       —"
        vals = ", ".join(f"{v:.4f}" for v in s["values"])
        print(f"{family:22s} {s['n_seeds']:2d} {s['mean']:9.4f} {sd}   {vals}")
    print(f"\nAnker: perfekt {PERFECT}, Löschen {DELETION_FLOOR}, nichts tun {DO_NOTHING}")
    print("LÄUFT = noch nicht fertig, BUDGET = von max_epochs beendet statt von early stopping.")
    print("Beide bleiben aus dem Mittelwert heraus: das eine ist unvollständig, das andere eine")
    print("untere Schranke.")

    #: energy_ratio is RMS(recovered clean) / RMS(true clean). Far below 1 means
    #: the model scaled the signal down rather than corrected it -- which the
    #: objective does not fully punish, because SI-SDR is scale-invariant.
    collapsed = [r for r in rows if not r["running"]
                 and r["terms"].get("energy_ratio", 1.0) < 0.5]
    if collapsed:
        print("\nSkalen-Kollaps (energy_ratio < 0.5) trotz brauchbarem Verlust:")
        for r in collapsed:
            print(f"  {r['family']:22s} seed {r['seed']}  best {r['best_val_loss']:8.4f}  "
                  f"energy_ratio {r['terms']['energy_ratio']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"runs": rows, "summary": summary,
                                    "anchors": {"perfect": PERFECT,
                                                "deletion_floor": DELETION_FLOOR,
                                                "do_nothing": DO_NOTHING}},
                                   indent=2), encoding="utf-8")
    print(f"\ngeschrieben: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
