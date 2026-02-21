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
    MetricsReport,
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
from facet.helpers.interactive import WaitForConfirmation

INPUT_FILE  = "./examples/datasets/NiazyFMRI.edf"
OUTPUT_FILE = "./output/corrected_inline.edf"


# ---------------------------------------------------------------------------
# A. Inline def steps in a Pipeline
# ---------------------------------------------------------------------------
#
# Any function that takes a ProcessingContext and returns one is a valid step.
# But returning nothing is equivalent to returning the context unchanged.
# Use a def for anything that needs more than one line.

def log_sfreq(ctx: ProcessingContext) -> ProcessingContext:
    triggers = ctx.get_triggers()
    n_triggers = len(triggers) if triggers is not None else 0
    print(f"  sfreq = {ctx.get_sfreq()} Hz, " f"n_triggers = {n_triggers}")
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

# result = pipeline.run()
# result.print_summary()


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
    | WaitForConfirmation(
    message="✅ Execution Finished! Press Enter to print the result..."
    )
    | MetricsReport(name="Pipe-operator result")
)

from rich import print
from rich.panel import Panel

print(
    Panel.fit(
        "[bold green]✨ Execution finished! ✨\n\n[dim]Press [bold]q[/bold] to quit.",
        border_style="green",
        title="FACETpy",
    )
)
print(f"\nPipe-operator result: SNR = {ctx.get_metric('snr', 'N/A'):.3f}")

# Custom def steps also work with |
ctx = ctx | log_sfreq
