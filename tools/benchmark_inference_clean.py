#!/usr/bin/env python3
"""Apples-to-apples CPU inference benchmark for unified-holdout evaluator.

Issue: the original `inference_seconds` metric in each model's
output/model_evaluations/<id>/holdout_v1/metrics.json was measured by
running every model in the same Python process sequentially. The first
model in the sequence eats all the cold-start cost (TorchScript load,
MKL kernel JIT, allocator warmup, mlock cache fill). Subsequent models
inherit a warm process and report artificially fast numbers.

This script fixes that by spawning a FRESH subprocess for every model,
running one warmup pass (discarded) and three timed passes (median
reported). It updates each model's metrics.json in place so downstream
aggregators and figures use the corrected numbers.

Usage:
    uv run python tools/benchmark_inference_clean.py
    uv run python tools/benchmark_inference_clean.py --models cascaded_dae demucs
    uv run python tools/benchmark_inference_clean.py --warmup 1 --runs 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO_ROOT / "output/model_evaluations"

ALL_MODELS = [
    "cascaded_dae",
    "cascaded_context_dae",
    "dpae",
    "ic_unet",
    "denoise_mamba",
    "vit_spectrogram",
    "st_gnn",
    "conv_tasnet",
    "demucs",
    "sepformer",
    "nested_gan",
    "d4pm",
    "dhct_gan",
    "dhct_gan_v2",
]

CHILD_SCRIPT = """
import json
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path("{repo_root}")
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from eval_unified_holdout import (  # noqa: E402
    MODELS,
    INFERENCE_FUNCS,
    ModelSpec,
    _resolve_ts_path,
    compute_holdout_indices,
    load_holdout,
    DATASET_PATH,
)

warnings.filterwarnings("ignore")

mid = "{model_id}"
n_warmup = {warmup}
n_runs = {runs}

spec = MODELS[mid]
resolved_ts = _resolve_ts_path(spec.ts_path)
resolved_ts_cpu = _resolve_ts_path(spec.ts_path_cpu)
if resolved_ts != spec.ts_path or resolved_ts_cpu != spec.ts_path_cpu:
    spec = ModelSpec(
        **{{**spec.__dict__, "ts_path": resolved_ts, "ts_path_cpu": resolved_ts_cpu}}
    )

holdout = compute_holdout_indices()
ds = load_holdout(DATASET_PATH, holdout)
infer = INFERENCE_FUNCS[mid]

# Warmup passes (excluded from timing)
for _ in range(n_warmup):
    _ = infer(spec, ds, device="cpu")

# Timed passes
times = []
for _ in range(n_runs):
    t0 = time.perf_counter()
    _ = infer(spec, ds, device="cpu")
    times.append(time.perf_counter() - t0)

print(json.dumps({{
    "model_id": mid,
    "times_seconds": times,
    "n_warmup": n_warmup,
    "n_runs": n_runs,
}}))
"""


def benchmark_one(model_id: str, *, warmup: int, runs: int, timeout: int) -> dict | None:
    """Run model in a fresh subprocess and return timing dict."""
    script = CHILD_SCRIPT.format(
        repo_root=str(REPO_ROOT),
        model_id=model_id,
        warmup=warmup,
        runs=runs,
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except subprocess.TimeoutExpired:
        print(f"  [timeout] {model_id} exceeded {timeout}s")
        return None
    except subprocess.CalledProcessError as e:
        print(f"  [error] {model_id} child failed: {e.stderr[-400:]}")
        return None

    # Last non-empty line should be the JSON payload
    line = next(
        (ln for ln in reversed(proc.stdout.splitlines()) if ln.strip().startswith("{")),
        None,
    )
    if line is None:
        print(f"  [parse error] no JSON line in stdout for {model_id}")
        return None
    return json.loads(line)


def update_metrics_json(model_id: str, median_seconds: float) -> bool:
    """Patch inference_seconds in the model's holdout_v1/metrics.json."""
    path = EVAL_ROOT / model_id / "holdout_v1" / "metrics.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    if "metrics" in payload:
        for bucket_name, bucket in payload["metrics"].items():
            if isinstance(bucket, dict) and "inference_seconds" in bucket:
                bucket["inference_seconds_legacy"] = bucket["inference_seconds"]
                bucket["inference_seconds"] = median_seconds
                bucket["inference_timing_note"] = (
                    "remeasured 2026-05-16 with cold-start isolation (fresh subprocess per model, "
                    "1 warmup pass excluded, median of 3 timed passes)"
                )
    if "flat_metrics" in payload:
        for k in list(payload["flat_metrics"].keys()):
            if k.endswith("inference_seconds"):
                payload["flat_metrics"][k] = median_seconds
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Subprocess timeout per model (seconds). Default 900 = 15 min "
        "to accommodate the slowest (d4pm DDPM reverse loop).",
    )
    parser.add_argument(
        "--no-update",
        action="store_true",
        help="Print results but don't modify metrics.json files.",
    )
    args = parser.parse_args()

    targets = args.models if args.models else ALL_MODELS
    n_windows = 166 * 30  # 4980 channel-windows
    results: list[tuple[str, float, float, float]] = []  # (mid, median, min, max)

    print(f"Benchmark plan: {len(targets)} model(s), {args.warmup} warmup + {args.runs} timed each,")
    print(f"  fresh subprocess per model, n=4980 channel-windows per pass.\n")

    for mid in targets:
        print(f"=== {mid} ===")
        out = benchmark_one(mid, warmup=args.warmup, runs=args.runs, timeout=args.timeout)
        if out is None:
            continue
        times = out["times_seconds"]
        median = statistics.median(times)
        results.append((mid, median, min(times), max(times)))
        ms_per_win = median * 1000.0 / n_windows
        per_run = ", ".join(f"{t:.3f}s" for t in times)
        print(f"  runs:    {per_run}")
        print(f"  median:  {median:.3f}s  ({ms_per_win:.3f} ms / channel-window)")
        if not args.no_update:
            if update_metrics_json(mid, median):
                print(f"  patched: output/model_evaluations/{mid}/holdout_v1/metrics.json")
        print()

    # Summary
    if results:
        results.sort(key=lambda r: r[1])
        print("\nSummary (sorted by median wall-clock):\n")
        print(f"  {'Model':25s}  {'Median':>8s}  {'Min':>8s}  {'Max':>8s}  {'ms/win':>8s}")
        for mid, median, lo, hi in results:
            ms = median * 1000.0 / n_windows
            print(f"  {mid:25s}  {median:>7.3f}s  {lo:>7.3f}s  {hi:>7.3f}s  {ms:>7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
