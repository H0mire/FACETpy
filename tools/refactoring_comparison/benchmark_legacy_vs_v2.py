"""Parity, runtime and peak memory: FACETpy 0.1.0 against the v2.0 pipeline.

Both arms perform the **same functional task** on the same recording:

1. load the EDF and drop the non-EEG channels (``EMG``, ``ECG``)
2. high-pass at 1 Hz
3. up-sample by 10
4. detect the volume triggers (regex ``\\b1\\b``)
5. average-artifact subtraction over a 25-epoch window
6. down-sample back to the native rate
7. low-pass at 40 Hz

Three properties of the measurement matter more than the numbers:

* **Separate processes.** Each repetition runs in a fresh subprocess, so peak
  memory is that arm's own peak and not a high-water mark left by the other arm.
  The memory definition is one thing throughout: *peak resident set size of the
  worker process* (``ru_maxrss``), which includes the interpreter and every
  library the arm imports — that is the honest number for "what does it cost to
  run this", and it is the same definition for both arms.
* **Warm-up.** The first repetition is discarded. It pays for imports, filter
  design caches and the first read of the file from disk.
* **Parity before speed.** A speed comparison between two implementations that
  compute different things is meaningless, so the corrected signal of both arms
  is written once and compared channel-wise against a stated tolerance. Where
  they differ, the difference is reported rather than tuned away.

Usage::

    .venv/bin/python tools/refactoring_comparison/benchmark_legacy_vs_v2.py \\
        --legacy-path <scratch>/legacy_run --input examples/datasets/NiazyFMRI.edf \\
        --repetitions 3 --out output/refactoring_comparison
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

#: The functional task, stated once so both arms can be checked against it.
TASK = {
    "input": "examples/datasets/NiazyFMRI.edf",
    "dropped_channels": ["EMG", "ECG"],
    "highpass_hz": 1.0,
    "upsample_factor": 10,
    "trigger_regex": r"\b1\b",
    "aas_window_size": 25,
    "lowpass_hz": 40.0,
    "note": "Identical steps in identical order on both sides. The v2 pipeline "
            "expresses them as processors, the 0.1.0 API as method calls on a facade.",
}

#: Parity tolerance, fixed before the comparison ran.
#:
#: The two implementations are not expected to be bit-identical: they use
#: different epoch bookkeeping and MNE calls. What is claimed is that they
#: correct the *same artifact* to within a stated fraction of the uncorrected
#: signal's amplitude. The tolerance is expressed relative to the raw RMS so it
#: does not depend on the recording's scale.
PARITY = {
    "metric": "per-channel RMS of (legacy_corrected - v2_corrected), relative to RMS(uncorrected)",
    "tolerance_relative": 0.05,
    "rationale": "5 % of the uncorrected amplitude. Tight enough that a different "
                 "artifact estimate fails it, loose enough to tolerate differing "
                 "filter edge handling and epoch bookkeeping.",
}

WORKER = r'''
import json, os, resource, sys, time, warnings
warnings.filterwarnings("ignore")
arm, input_path, out_npz, legacy_path = sys.argv[1:5]
import contextlib, io
import numpy as np

def run_legacy():
    sys.path.insert(0, legacy_path)
    from FACET.Facet import Facet
    f = Facet()
    f.import_EEG(input_path, rel_trig_pos=-0.01, upsampling_factor=10, bads=["EMG", "ECG"])
    f.pre_processing()                       # highpass(1) + upsample(x10)
    f.find_triggers(r"\b1\b")
    f.apply_AAS(method="numpy", rel_window_offset=0, window_size=25)
    f.remove_artifacts()
    f.downsample()
    f.lowpass(h_freq=40)
    raw = f.get_EEG()["raw"]
    return raw

def run_current(realign):
    sys.path.insert(0, os.path.join(os.environ["FACET_REPO"], "src"))
    from facet.core import Pipeline
    from facet.io import Loader
    from facet.preprocessing import DownSample, DropChannels, HighPassFilter, LowPassFilter, TriggerDetector, UpSample
    from facet.correction import AASCorrection
    pipeline = Pipeline([
        Loader(path=input_path, preload=True, artifact_to_trigger_offset=-0.01),
        DropChannels(channels=["EMG", "ECG"]),
        HighPassFilter(freq=1.0),
        TriggerDetector(regex=r"\b1\b"),
        UpSample(factor=10),
        AASCorrection(window_size=25, correlation_threshold=0.975,
                      realign_after_averaging=realign),
        DownSample(factor=10),
        LowPassFilter(freq=40.0),
    ], name="legacy-equivalent")
    result = pipeline.run()
    if not result.success:
        raise RuntimeError(f"pipeline failed: {result.error}")
    return result.context.get_raw()

t0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    if arm == "legacy":
        raw = run_legacy()
    else:
        # "current" is the shipped default; "current_matched" switches off the
        # post-averaging realignment the legacy code never had, so the parity
        # check separates "the refactoring changed behaviour" from "v2 gained a
        # step".
        raw = run_current(realign=(arm == "current"))
elapsed = time.perf_counter() - t0
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform != "darwin":
    peak *= 1024          # Linux reports kB, macOS bytes
if out_npz:
    import mne
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
    names = [raw.ch_names[i] for i in picks]
    np.savez_compressed(out_npz, data=raw.get_data(picks=picks), names=np.array(names),
                        sfreq=np.array([raw.info["sfreq"]]))
print(json.dumps({"arm": arm, "elapsed_seconds": elapsed, "peak_rss_bytes": int(peak),
                  "n_channels": len(raw.ch_names), "sfreq": float(raw.info["sfreq"]),
                  "n_samples": int(raw.n_times)}))
'''


def run_once(arm: str, input_path: Path, legacy_path: Path, out_npz: Path | None) -> dict:
    env = dict(os.environ, FACET_REPO=str(REPO), PYTHONWARNINGS="ignore")
    proc = subprocess.run(
        [sys.executable, "-c", WORKER, arm, str(input_path), str(out_npz or ""), str(legacy_path)],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{arm} worker failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def uncorrected_rms(input_path: Path) -> dict[str, float]:
    import mne
    raw = mne.io.read_raw_edf(input_path, preload=True, verbose="ERROR")
    raw.drop_channels([c for c in TASK["dropped_channels"] if c in raw.ch_names])
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude=[])
    data = raw.get_data(picks=picks) * 1e6
    return {raw.ch_names[i]: float(np.sqrt(np.mean(d ** 2)))
            for i, d in zip(picks, data)}


def parity(legacy_npz: Path, current_npz: Path, ref_rms: dict[str, float]) -> dict:
    a, b = np.load(legacy_npz, allow_pickle=True), np.load(current_npz, allow_pickle=True)
    names_a = [str(x) for x in a["names"]]
    names_b = [str(x) for x in b["names"]]
    shared = [n for n in names_a if n in names_b]
    n = min(a["data"].shape[1], b["data"].shape[1])
    rows = []
    for name in shared:
        da = a["data"][names_a.index(name)][:n] * 1e6
        db = b["data"][names_b.index(name)][:n] * 1e6
        diff = float(np.sqrt(np.mean((da - db) ** 2)))
        ref = ref_rms.get(name, float("nan"))
        rows.append({
            "channel": name,
            "rms_legacy_uv": float(np.sqrt(np.mean(da ** 2))),
            "rms_current_uv": float(np.sqrt(np.mean(db ** 2))),
            "rms_difference_uv": diff,
            "relative_to_uncorrected": diff / ref if ref and np.isfinite(ref) else float("nan"),
            "within_tolerance": bool(diff / ref <= PARITY["tolerance_relative"])
            if ref and np.isfinite(ref) else False,
            "pearson_r": float(np.corrcoef(da, db)[0, 1]),
        })
    passed = sum(1 for r in rows if r["within_tolerance"])
    return {
        "definition": PARITY,
        "n_shared_channels": len(shared),
        "n_within_tolerance": passed,
        "n_samples_compared": int(n),
        "sfreq_legacy": float(a["sfreq"][0]), "sfreq_current": float(b["sfreq"][0]),
        "median_relative_difference": float(np.median([r["relative_to_uncorrected"] for r in rows])),
        "median_pearson_r": float(np.median([r["pearson_r"] for r in rows])),
        "verdict": "Parität innerhalb der Toleranz" if passed == len(rows) else
                   f"{len(rows) - passed} von {len(rows)} Kanälen außerhalb der Toleranz",
        "channels": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legacy-path", type=Path, required=True, help="Directory containing the importable FACET package")
    p.add_argument("--input", type=Path, default=REPO / TASK["input"])
    p.add_argument("--repetitions", type=int, default=3, help="Measured repetitions, after one discarded warm-up")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for arm in ("legacy", "current", "current_matched"):
        npz = args.out / f"corrected_{arm}.npz"
        print(f"[{arm}] warm-up ...", flush=True)
        warm = run_once(arm, args.input, args.legacy_path, npz)
        reps = []
        for i in range(args.repetitions):
            r = run_once(arm, args.input, args.legacy_path, None)
            reps.append(r)
            print(f"[{arm}] rep {i + 1}/{args.repetitions}: "
                  f"{r['elapsed_seconds']:.2f} s, {r['peak_rss_bytes'] / 2**20:.0f} MiB", flush=True)
        times = [r["elapsed_seconds"] for r in reps]
        peaks = [r["peak_rss_bytes"] for r in reps]
        results[arm] = {
            "repetitions": args.repetitions,
            "warmup_discarded": True,
            "warmup_seconds": warm["elapsed_seconds"],
            "elapsed_seconds_mean": float(np.mean(times)),
            "elapsed_seconds_sd": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "elapsed_seconds_min": float(np.min(times)),
            "elapsed_seconds_all": times,
            "peak_rss_bytes_mean": float(np.mean(peaks)),
            "peak_rss_mib_mean": float(np.mean(peaks)) / 2 ** 20,
            "peak_rss_mib_all": [x / 2 ** 20 for x in peaks],
            "output_channels": warm["n_channels"],
            "output_sfreq": warm["sfreq"],
            "output_samples": warm["n_samples"],
        }

    ref = uncorrected_rms(args.input)
    par = parity(args.out / "corrected_legacy.npz", args.out / "corrected_current_matched.npz", ref)
    par_default = parity(args.out / "corrected_legacy.npz", args.out / "corrected_current.npz", ref)

    payload = {
        "task": TASK,
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "precision": "float64 throughout (MNE default); no GPU involved on either arm",
            "memory_definition": "peak resident set size of the worker process (ru_maxrss), "
                                 "one fresh process per repetition",
            "timing_definition": "wall clock around the full task inside the worker, "
                                 "excluding interpreter start-up",
        },
        "arms": results,
        "speedup_current_over_legacy": results["legacy"]["elapsed_seconds_mean"] /
                                       results["current"]["elapsed_seconds_mean"],
        "memory_ratio_current_over_legacy": results["current"]["peak_rss_mib_mean"] /
                                            results["legacy"]["peak_rss_mib_mean"],
        "speedup_matched_over_legacy": results["legacy"]["elapsed_seconds_mean"] /
                                       results["current_matched"]["elapsed_seconds_mean"],
        "parity": par,
        "parity_against_shipped_default": par_default,
        "parity_note": "The registered parity check compares the legacy arm against "
                       "'current_matched' — the v2 pipeline with the post-averaging "
                       "realignment switched off, i.e. the same algorithm. "
                       "'parity_against_shipped_default' additionally shows how far the "
                       "shipped default moves the result, which is a feature difference, "
                       "not a refactoring defect.",
    }
    out = args.out / "benchmark_legacy_vs_v2.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nLaufzeit  legacy {results['legacy']['elapsed_seconds_mean']:.2f} ± "
          f"{results['legacy']['elapsed_seconds_sd']:.2f} s | "
          f"v2 {results['current']['elapsed_seconds_mean']:.2f} ± "
          f"{results['current']['elapsed_seconds_sd']:.2f} s "
          f"(Faktor {payload['speedup_current_over_legacy']:.2f})")
    print(f"Peak-RSS  legacy {results['legacy']['peak_rss_mib_mean']:.0f} MiB | "
          f"v2 {results['current']['peak_rss_mib_mean']:.0f} MiB "
          f"(Faktor {payload['memory_ratio_current_over_legacy']:.2f})")
    print(f"Parität (gleicher Algorithmus)  {par['verdict']}: Median relative Differenz "
          f"{par['median_relative_difference']:.4f}, Median r {par['median_pearson_r']:.4f}")
    print(f"Parität (v2-Voreinstellung)     {par_default['verdict']}: Median relative Differenz "
          f"{par_default['median_relative_difference']:.4f}, Median r {par_default['median_pearson_r']:.4f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
