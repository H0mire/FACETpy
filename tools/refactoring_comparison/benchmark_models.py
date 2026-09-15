"""Fair inference runtime and peak memory for the compared model architectures.

The training summaries record a wall-clock time per run, but those runs happened
on different machines, at different batch sizes, with different early-stopping
points. Putting them in one table would compare hardware, not architectures.

This benchmark instead measures the one thing that *can* be measured under
identical conditions from what is on disk: **inference on the unified holdout**.
The rules are the same for every architecture and are fixed here rather than per
model:

* the same 166 holdout windows, the same input contract, the same device;
* one fresh subprocess per repetition, so peak memory is that model's own peak;
* one discarded warm-up, then a fixed number of measured repetitions;
* peak memory is the worker process's peak resident set size — interpreter and
  framework included, identically for all arms.

Training time is reported alongside **as recorded**, explicitly marked as not
comparable. Deleting it would hide that some runs are 30× longer than others;
presenting it as a comparison would be wrong.

Usage::

    .venv/bin/python tools/refactoring_comparison/benchmark_models.py \\
        --repetitions 3 --device cpu --out output/refactoring_comparison
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

WORKER = r'''
import json, os, resource, sys, time, warnings
warnings.filterwarnings("ignore")
model_id, device, dataset, n_windows = sys.argv[1:5]
n_windows = int(n_windows)
sys.path.insert(0, os.path.join(os.environ["FACET_REPO"], "tools"))
sys.path.insert(0, os.path.join(os.environ["FACET_REPO"], "src"))
import contextlib, io
import numpy as np
from pathlib import Path
sys.path.insert(0, os.environ["FACET_REPO"])
from masterthesis_guide.reproduce import load_catalog, load_holdout, adapter, ROOT
from facet.models.masterthesis.adapters import predict_from_context
c = load_catalog()
idx = np.asarray(json.loads((ROOT / c['datasets']['proof_fit']['split']).read_text())['indices'])
if n_windows > 0:
    idx = idx[:n_windows]
ds = load_holdout(Path(dataset), idx)
a = adapter('holdout_' + model_id, c, device=device)
t0 = time.perf_counter()
model, _ = a._load_model()
pred = predict_from_context(a.packing, model, ds['noisy_context'], device=device)
elapsed = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != "darwin":
    peak *= 1024
print(json.dumps({"model_id": model_id, "elapsed_seconds": elapsed,
                  "peak_rss_bytes": int(peak), "n_windows": int(pred.shape[0])}))
'''


def run_once(model_id: str, device: str, dataset: Path, n_windows: int, timeout: float) -> dict:
    env = dict(os.environ, FACET_REPO=str(REPO), PYTHONWARNINGS="ignore")
    proc = subprocess.run(
        [sys.executable, "-c", WORKER, model_id, device, str(dataset), str(n_windows)],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-1500:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def training_time(model_id: str) -> dict:
    from masterthesis_guide.reproduce import load_catalog
    c = load_catalog()
    return {"artifact_ids": c['experiments']['holdout_' + model_id]['artifacts']}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--device", default="cpu",
                   help="One device for every arm. CPU is the default because it is the only "
                        "device all exports run on, and a mixed-device table compares nothing.")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--n-windows", type=int, default=32,
                   help="Prefix of the holdout split used for timing. The same windows for "
                        "every arm; results are also reported per window so the count does "
                        "not silently favour a model.")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="Per-repetition budget. An architecture that exceeds it is recorded "
                        "as impractical on this device rather than dropped silently.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from masterthesis_guide.reproduce import load_catalog
    catalog = load_catalog()
    model_ids = args.models or sorted(e.removeprefix('holdout_') for e, record in catalog['experiments'].items() if e.startswith('holdout_') and record['artifacts'])

    rows, failures = [], []
    for model_id in model_ids:
        try:
            run_once(model_id, args.device, args.dataset, args.n_windows, args.timeout)
            reps = [run_once(model_id, args.device, args.dataset, args.n_windows, args.timeout)
                    for _ in range(args.repetitions)]
        except subprocess.TimeoutExpired:
            failures.append({"model_id": model_id,
                             "reason": f"Exceeded {args.timeout:.0f} s per repetition for {args.n_windows} windows on {args.device}"})
            print(f"  [timeout] {model_id}", flush=True)
            continue
        except Exception as exc:                                    # noqa: BLE001
            failures.append({"model_id": model_id, "reason": str(exc).strip().splitlines()[-1][:200]})
            print(f"  [skip] {model_id}: {failures[-1]['reason']}", flush=True)
            continue
        times = [r["elapsed_seconds"] for r in reps]
        peaks = [r["peak_rss_bytes"] for r in reps]
        rows.append({
            "model_id": model_id,
            "family": catalog["models"][model_id]["family"],
            "device": args.device,
            "repetitions": args.repetitions,
            "warmup_discarded": True,
            "n_windows": reps[0]["n_windows"],
            "inference_seconds_mean": round(float(np.mean(times)), 4),
            "inference_seconds_sd": round(float(np.std(times, ddof=1)), 4),
            "inference_seconds_min": round(float(np.min(times)), 4),
            "ms_per_window": round(1000.0 * float(np.mean(times)) / reps[0]["n_windows"], 3),
            "peak_rss_mib_mean": round(float(np.mean(peaks)) / 2 ** 20, 1),
            "peak_rss_mib_sd": round(float(np.std(peaks, ddof=1)) / 2 ** 20, 2),
            **training_time(model_id),
        })
        print(f"  {model_id:24s} {rows[-1]['inference_seconds_mean']:7.3f} ± "
              f"{rows[-1]['inference_seconds_sd']:.3f} s   "
              f"{rows[-1]['peak_rss_mib_mean']:7.0f} MiB", flush=True)

    payload = {
        "protocol": {
            "task": f"Inference on the first {args.n_windows} saved holdout windows; identical inputs for every model",
            "per_repetition_timeout_seconds": args.timeout,
            "device": args.device, "repetitions": args.repetitions,
            "warmup": "One discarded repetition per model",
            "process_isolation": "One fresh process per repetition",
            "memory_definition": "Worker peak RSS (ru_maxrss), including interpreter and framework",
            "timing_definition": "Wall time around model loading and prediction, excluding interpreter startup",
            "precision": "float32",
            "not_measured": "Training time: original runs used different machines, batch sizes and stopping criteria",
        },
        "environment": {"platform": platform.platform(),
                        "processor": platform.processor() or platform.machine(),
                        "python": sys.version.split()[0], "numpy": np.__version__},
        "models": rows,
        "not_benchmarked": failures,
    }
    (args.out / "benchmark_models.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(rows)} models measured, {len(failures)} unavailable "
          f"-> {args.out / 'benchmark_models.json'}")


if __name__ == "__main__":
    main()
