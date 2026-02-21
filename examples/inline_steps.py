"""
Inline steps — custom functions and the pipe operator.

Shows two ergonomic shortcuts:

  1. Plain ``def`` functions as pipeline steps — useful when you need a quick
     custom step without writing a full Processor subclass.

  2. The ``|`` operator on ProcessingContext — apply processors one by one,
     ideal for interactive / notebook prototyping.

Also demonstrates the convenience processors DropChannels, PickChannels, and
PrintMetric that eliminate common boilerplate lambdas.
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
    DropChannels,
    PickChannels,
    PrintMetric,
    load_edf,
)

INPUT_FILE  = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_FILE = "./output/corrected_inline.edf"


# ---------------------------------------------------------------------------
# A. Inline def steps in a Pipeline
# ---------------------------------------------------------------------------
#
# Any function that takes a ProcessingContext and returns one is a valid step.
# Use a def for anything that needs more than one line.

def log_sfreq(ctx: ProcessingContext) -> ProcessingContext:
    """Print current sampling frequency — handy for sanity-checking."""
    print(f"  sfreq = {ctx.get_sfreq()} Hz, "
          f"n_triggers = {len(ctx.get_triggers() or [])}")
    return ctx


pipeline = Pipeline([
    EDFLoader(path=INPUT_FILE, preload=True),

    # Drop non-EEG channels by name — no lambda needed
    DropChannels(channels=["EKG", "EMG", "EOG", "ECG"]),

    TriggerDetector(regex=r"\b1\b"),

    # Custom def step: log sampling frequency for verification
    log_sfreq,

    HighPassFilter(freq=1.0),
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10),
    LowPassFilter(freq=70),

    SNRCalculator(),

    # Print a metric inline — no (print(...) or ctx) tricks required
    PrintMetric("snr"),

    EDFExporter(path=OUTPUT_FILE, overwrite=True),
], name="Inline Steps")

result = pipeline.run()
result.print_summary()


# ---------------------------------------------------------------------------
# B. Pipe operator — chain processors outside a Pipeline
# ---------------------------------------------------------------------------
#
# load_edf() gives you a ProcessingContext without importing MNE directly.
# Then chain processors with | — ideal for notebooks or exploratory scripts.

ctx = load_edf(INPUT_FILE, preload=True)

ctx = (
    ctx
    | HighPassFilter(1.0)
    | TriggerDetector(regex=r"\b1\b")
    | UpSample(factor=10)
    | AASCorrection(window_size=30)
    | DownSample(factor=10)
    | SNRCalculator()
)

print(f"\nPipe-operator result: SNR = {ctx.get_metric('snr', 'N/A'):.3f}")

# Custom def steps also work with |
ctx = ctx | log_sfreq
