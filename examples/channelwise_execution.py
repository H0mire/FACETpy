"""
Channel-wise execution — parallelise correction and filtering across EEG channels.

FACETpy's ``ParallelExecutor`` automatically splits a recording into channel
groups, processes each group in a separate worker process, and merges the
results back into a single context.  The only thing a processor needs to opt
in is two class-level flags::

    parallel_safe = True   # No cross-channel dependencies; safe for workers
    channel_wise  = True   # Split/merge by channel index

This file walks through four patterns:

  A. Inspecting which processors in a pipeline support channel-wise execution.
  B. Enabling channel-wise execution with pipeline.run(parallel=True).
  C. Choosing backend and worker count; measuring the speed-up.
  D. Writing a custom processor that participates in channel-wise execution.
"""

import multiprocessing
import time
from pathlib import Path

import numpy as np

from facet import (
    AASCorrection,
    DownSample,
    EDFExporter,
    HighPassFilter,
    Loader,
    LowPassFilter,
    MetricsReport,
    Pipeline,
    RMSCalculator,
    TriggerAligner,
    TriggerDetector,
    UpSample,
)
from facet.core import ParallelExecutor, ProcessingContext
from facet.core.processor import Processor

INPUT_FILE    = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_DIR    = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE   = str(OUTPUT_DIR / "corrected_channelwise.edf")
TRIGGER_REGEX = r"\b1\b"
UPSAMPLE      = 10


# ---------------------------------------------------------------------------
# A. Inspect which processors support channel-wise execution
# ---------------------------------------------------------------------------

def inspect_pipeline_flags():
    """Print parallel flags for every processor in a typical pipeline."""
    pipeline = Pipeline([
        Loader(path=INPUT_FILE, preload=True),
        TriggerDetector(regex=TRIGGER_REGEX),
        HighPassFilter(freq=1.0),
        UpSample(factor=UPSAMPLE),
        TriggerAligner(ref_trigger_index=0, upsample_for_alignment=False),
        AASCorrection(window_size=30),
        DownSample(factor=UPSAMPLE),
        LowPassFilter(freq=70),
        RMSCalculator(),
        MetricsReport(),
        EDFExporter(path=OUTPUT_FILE, overwrite=True),
    ], name="Flag inspection")

    print(f"\n{'Processor':<28} {'parallel_safe':<16} {'channel_wise':<14} {'run_once'}")
    print("-" * 72)
    for proc in pipeline.processors:
        ps  = getattr(proc, "parallel_safe", False)
        cw  = getattr(proc, "channel_wise",  False)
        ro  = getattr(proc, "run_once",      False)
        print(f"{proc.name:<28} {str(ps):<16} {str(cw):<14} {ro}")

    eligible = [p.name for p in pipeline.processors if getattr(p, "channel_wise", False)]
    print(f"\nProcessors eligible for channel-wise execution: {eligible}")


# ---------------------------------------------------------------------------
# B. Run a correction pipeline with channel-wise execution enabled
# ---------------------------------------------------------------------------

def run_channelwise(n_jobs: int = -1):
    """
    Standard correction pipeline — channel-wise execution for eligible steps.

    AASCorrection, HighPassFilter, and LowPassFilter each have
    ``channel_wise = True``, so they will be split across *n_jobs* worker
    processes when ``parallel=True`` is passed to ``pipeline.run()``.

    Steps that are not channel_wise (TriggerDetector, Loader, EDFExporter)
    run serially as usual.
    """
    pipeline = Pipeline([
        Loader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=-0.005),
        TriggerDetector(regex=TRIGGER_REGEX),
        HighPassFilter(freq=1.0),                 # channel_wise=True
        UpSample(factor=UPSAMPLE),
        TriggerAligner(ref_trigger_index=0),      # channel_wise=True, run_once=True
        AASCorrection(window_size=30,             # channel_wise=True
                      correlation_threshold=0.975),
        DownSample(factor=UPSAMPLE),
        LowPassFilter(freq=70),                   # channel_wise=True
        RMSCalculator(),
        MetricsReport(),
        EDFExporter(path=OUTPUT_FILE, overwrite=True),
    ], name="Channel-wise AAS")

    result = pipeline.run(
        parallel=True,   # enable channel-wise execution
        n_jobs=n_jobs,   # -1 → all CPUs, -2 → all but one, N → exactly N
    )
    result.print_summary()
    return result


# ---------------------------------------------------------------------------
# C. Backend comparison and speed measurement
# ---------------------------------------------------------------------------

def benchmark_backends():
    """
    Compare wall-clock time across serial, threading, and multiprocessing
    backends by using ParallelExecutor directly on a single processor.

    Use this to decide which backend is best for your hardware:
    - serial       → useful as a baseline / for debugging
    - threading    → low overhead, but limited by the GIL (good for I/O-bound)
    - multiprocessing → true CPU parallelism; best for compute-heavy processors
    """
    # Build a minimal context once so we don't measure I/O in every trial.
    # Loader accepts None as the initial context (it creates a fresh one from the file).
    loader = Loader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=-0.005)
    ctx = loader.execute(None)
    ctx = TriggerDetector(regex=TRIGGER_REGEX).execute(ctx)
    ctx = UpSample(factor=UPSAMPLE).execute(ctx)

    processor = AASCorrection(window_size=30)
    n_cpus    = multiprocessing.cpu_count()

    print(f"\nBenchmarking AASCorrection on {len(ctx.get_raw().ch_names)} channels "
          f"({n_cpus} CPU cores available)\n")

    timings = {}
    for backend in ("serial", "threading", "multiprocessing"):
        executor = ParallelExecutor(n_jobs=n_cpus, backend=backend)

        t0 = time.perf_counter()
        executor.execute(processor, ctx)
        elapsed = time.perf_counter() - t0

        timings[backend] = elapsed
        print(f"  {backend:<20} {elapsed:.2f} s")

    serial_t = timings["serial"]
    for backend, t in timings.items():
        if backend != "serial":
            speedup = serial_t / tl
            print(f"  {backend} speed-up vs serial: {speedup:.2f}×")

    return timings


def benchmark_n_jobs():
    """
    Measure how processing time changes with different worker counts.

    Run this to find the sweet spot between parallelism overhead and gain
    for your specific machine and dataset.
    """
    loader = Loader(path=INPUT_FILE, preload=True, artifact_to_trigger_offset=-0.005)
    ctx = loader.execute(None)
    ctx = TriggerDetector(regex=TRIGGER_REGEX).execute(ctx)
    ctx = UpSample(factor=UPSAMPLE).execute(ctx)

    processor = AASCorrection(window_size=30)
    n_cpus    = multiprocessing.cpu_count()
    candidates = sorted({1, 2, min(4, n_cpus), n_cpus})

    print(f"\nScaling AASCorrection over {len(ctx.get_raw().ch_names)} channels\n")
    print(f"  {'n_jobs':<10} {'time (s)':<12} {'speed-up'}")
    print("  " + "-" * 34)

    serial_t = None
    for n in candidates:
        executor = ParallelExecutor(n_jobs=n, backend="multiprocessing")
        t0 = time.perf_counter()
        executor.execute(processor, ctx)
        elapsed = time.perf_counter() - t0

        if n == 1:
            serial_t = elapsed
        speedup = f"{serial_t / elapsed:.2f}×" if serial_t else "—"
        print(f"  {n:<10} {elapsed:<12.2f} {speedup}")


# ---------------------------------------------------------------------------
# D. Custom processor with channel-wise execution
# ---------------------------------------------------------------------------

class ChannelZScoreNormalizer(Processor):
    """
    Normalise each EEG channel to zero mean and unit variance independently.

    Because the z-score for channel A doesn't depend on channel B, this
    processor is safe to run channel-wise in parallel.
    """

    name        = "channel_zscore_normalizer"
    description = "Per-channel z-score normalisation (mean=0, std=1)"

    requires_raw  = True
    modifies_raw  = True

    parallel_safe = True
    channel_wise  = True

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw  = context.get_raw().copy()
        data = raw._data  # shape: [n_channels, n_samples]

        mean = data.mean(axis=1, keepdims=True)
        std  = data.std(axis=1, keepdims=True)
        std  = np.where(std == 0, 1.0, std)   # avoid div-by-zero on flat channels

        raw._data = (data - mean) / std
        return context.with_raw(raw)


def run_custom_channelwise_processor():
    """
    Plug a custom channel-wise processor into a standard pipeline.

    pipeline.run(parallel=True) treats ChannelZScoreNormalizer exactly like
    the built-in AASCorrection — the executor splits channels, processes in
    parallel, and merges the result back seamlessly.
    """
    pipeline = Pipeline([
        Loader(path=INPUT_FILE, preload=True),
        TriggerDetector(regex=TRIGGER_REGEX),
        ChannelZScoreNormalizer(),    # runs channel-wise when parallel=True
        HighPassFilter(freq=1.0),
        UpSample(factor=UPSAMPLE),
        AASCorrection(window_size=30),
        DownSample(factor=UPSAMPLE),
        LowPassFilter(freq=70),
        MetricsReport(),
    ], name="Custom channel-wise normalizer")

    result = pipeline.run(parallel=True, n_jobs=-1)
    result.print_summary()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== A. Pipeline flag inspection ===")
    inspect_pipeline_flags()

    print("\n=== B. Channel-wise correction pipeline ===")
    run_channelwise(n_jobs=-1)

    print("\n=== C. Backend comparison ===")
    benchmark_backends()

    print("\n=== C. n_jobs scaling ===")
    benchmark_n_jobs()

    print("\n=== D. Custom channel-wise processor ===")
    run_custom_channelwise_processor()
