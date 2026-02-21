"""
Inline steps — lambdas, def functions, and the pipe operator.

Shows the two ergonomic shortcuts added in v2.1:

  1. Plain callables (lambda or def) as pipeline steps — no need to subclass
     Processor or wrap in LambdaProcessor explicitly.

  2. The | operator on ProcessingContext — apply processors one by one outside
     a Pipeline, useful for interactive / notebook prototyping.
"""

from facet import (
    Pipeline,
    ProcessingContext,
    EDFLoader,
    EDFExporter,
    TriggerDetector,
    HighPassFilter,
    LowPassFilter,
    UpSample,
    DownSample,
    AASCorrection,
    SNRCalculator,
)

INPUT_FILE = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_FILE = "./output/corrected_inline.edf"


# ---------------------------------------------------------------------------
# A. Inline steps in a Pipeline
# ---------------------------------------------------------------------------
#
# Any callable that accepts a ProcessingContext and returns one is valid.
# Use a lambda for one-liners, a def for anything longer.

def drop_non_eeg(ctx: ProcessingContext) -> ProcessingContext:
    """Keep only EEG and stimulus channels."""
    raw = ctx.get_raw().copy().pick(picks=["eeg", "stim"], verbose=False)
    return ctx.with_raw(raw)


def log_sfreq(ctx: ProcessingContext) -> ProcessingContext:
    """Print current sampling frequency — useful for debugging."""
    print(f"  [debug] sfreq = {ctx.get_sfreq()} Hz, "
          f"n_triggers = {len(ctx.get_triggers() or [])}")
    return ctx


pipeline = Pipeline([
    EDFLoader(path=INPUT_FILE, preload=True),

    # lambda: one-liner channel selection
    lambda ctx: ctx.with_raw(
        ctx.get_raw().copy().drop_channels(
            [ch for ch in ["EKG", "EMG", "EOG", "ECG"]
             if ch in ctx.get_raw().ch_names],
            on_missing="ignore",
        )
    ),

    TriggerDetector(regex=r"\b1\b"),

    # def: multi-line custom step
    log_sfreq,

    HighPassFilter(freq=1.0),
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10),
    LowPassFilter(freq=70),

    SNRCalculator(),

    # lambda: inline debug tap between steps
    lambda ctx: (
        print(f"  [debug] SNR = {ctx.metadata.custom.get('metrics', {}).get('snr', 'N/A'):.3f}")
        or ctx
    ),

    EDFExporter(path=OUTPUT_FILE, overwrite=True),
], name="Inline Steps")

result = pipeline.run()
print(f"\nPipeline finished — success={result.success}, SNR={result.metrics.get('snr', 'N/A'):.3f}")


# ---------------------------------------------------------------------------
# B. Pipe operator — processors outside a Pipeline
# ---------------------------------------------------------------------------
#
# Ideal for notebooks or exploratory scripts where you want to apply a handful
# of steps to an already-loaded Raw object.

import mne  # noqa: E402

raw = mne.io.read_raw_edf(INPUT_FILE, preload=True, verbose=False)
ctx = ProcessingContext(raw)

# Chain processors with |
ctx = (
    ctx
    | HighPassFilter(1.0)
    | TriggerDetector(regex=r"\b1\b")
    | UpSample(factor=10)
    | AASCorrection(window_size=30)
    | DownSample(factor=10)
    | SNRCalculator()
)

print(f"\nPipe-operator result: SNR = {ctx.metadata.custom.get('metrics', {}).get('snr', 'N/A'):.3f}")

# Plain callables also work with |
ctx = ctx | log_sfreq
