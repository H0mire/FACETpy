# FACETpy Quick Start Guide

Get started with FACETpy in minutes!

## Installation

```bash
# From source
git clone https://github.com/your-org/facetpy.git
cd facetpy
pip install -e .

# Or from PyPI (when released)
pip install facetpy
```

## Basic Usage

### 1. Simple Correction Pipeline

```python
from facet import create_standard_pipeline

# Create a standard pipeline
pipeline = create_standard_pipeline(
    input_path="my_data.edf",
    output_path="corrected.edf",
    trigger_regex=r"\b1\b"  # Your trigger pattern
)

# Run it!
result = pipeline.run()

if result.success:
    print(f"✓ Correction completed in {result.execution_time:.2f}s")
    print(f"SNR: {result.context.metadata.custom['metrics']['snr']:.2f}")
else:
    print(f"✗ Failed: {result.error}")
```

### 2. Custom Pipeline

```python
from facet.core import Pipeline
from facet.io import Loader, EDFExporter
from facet.preprocessing import TriggerDetector, UpSample, DownSample
from facet.correction import AASCorrection, ANCCorrection
from facet.evaluation import SNRCalculator, MetricsReport

# Build your custom pipeline
pipeline = Pipeline([
    # Load data
    Loader(path="data.edf", preload=True),

    # Detect triggers
    TriggerDetector(regex=r"\b1\b"),

    # Upsample for precision
    UpSample(factor=10),

    # Main correction
    AASCorrection(
        window_size=30,
        correlation_threshold=0.975
    ),

    # Optional: Adaptive noise cancellation
    ANCCorrection(
        filter_order=5,
        hp_freq=1.0
    ),

    # Downsample back
    DownSample(factor=10),

    # Evaluate
    SNRCalculator(),
    MetricsReport(),

    # Export
    EDFExporter(path="corrected.edf", overwrite=True)
], name="My Custom Pipeline")

# Run with parallel processing
result = pipeline.run(parallel=True, n_jobs=-1)
```

### 3. Step-by-Step Processing

```python
from facet.core import ProcessingContext
from facet.io import Loader
from facet.preprocessing import TriggerDetector, UpSample
from facet.correction import AASCorrection

# Load data
loader = Loader(path="data.edf", preload=True)
context = loader.execute(ProcessingContext())

# Detect triggers
detector = TriggerDetector(regex=r"\b1\b")
context = detector.execute(context)

print(f"Found {len(context.get_triggers())} triggers")

# Upsample
upsampler = UpSample(factor=10)
context = upsampler.execute(context)

# Apply correction
aas = AASCorrection(window_size=30)
context = aas.execute(context)

# Access results
corrected_raw = context.get_raw()
corrected_raw.save("corrected.fif")
```

## Common Patterns

### Batch Processing Multiple Files

```python
from facet.core import Pipeline, ProcessingContext
from facet.io import Loader, EDFExporter
from facet.preprocessing import TriggerDetector, UpSample, DownSample
from facet.correction import AASCorrection

# Define correction pipeline (reusable)
correction_steps = Pipeline([
    TriggerDetector(regex=r"\b1\b"),
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10)
])

# Process multiple files
files = ["subject1.edf", "subject2.edf", "subject3.edf"]

for input_file in files:
    print(f"\nProcessing {input_file}...")

    # Load
    loader = Loader(path=input_file, preload=True)
    context = loader.execute(ProcessingContext())

    # Correct
    result = correction_steps.run(initial_context=context)

    if result.success:
        # Export
        output_file = input_file.replace('.edf', '_corrected.edf')
        exporter = EDFExporter(path=output_file, overwrite=True)
        exporter.execute(result.context)
        print(f"  ✓ Saved to {output_file}")
    else:
        print(f"  ✗ Failed: {result.error}")
```

### Conditional Processing

```python
from facet.core import Pipeline, ConditionalProcessor
from facet.correction import AASCorrection, PCACorrection
from facet.evaluation import SNRCalculator

def needs_extra_correction(context):
    """Apply PCA if SNR is poor after AAS."""
    metrics = context.metadata.custom.get('metrics', {})
    snr = metrics.get('snr', float('inf'))
    return snr < 10  # Threshold

pipeline = Pipeline([
    # ... loading and preprocessing ...

    # Main correction
    AASCorrection(window_size=30),

    # Evaluate
    SNRCalculator(),

    # Conditionally apply PCA
    ConditionalProcessor(
        condition=needs_extra_correction,
        processor=PCACorrection(n_components=0.95)
    ),

    # ... export ...
])
```

### Accessing Intermediate Results

```python
from facet.core import Pipeline

pipeline = Pipeline([
    # ... your processors ...
])

result = pipeline.run()

# Access final context
final_raw = result.context.get_raw()
triggers = result.context.get_triggers()
metrics = result.context.metadata.custom.get('metrics', {})

# Access processing history
for entry in result.context.get_history():
    print(f"{entry['processor']}: {entry['timestamp']}")
```

## Configuration Tips

### Trigger Detection

```python
# For fMRI volume triggers
TriggerDetector(regex=r"\b1\b")

# For slice triggers (TR marker)
TriggerDetector(regex=r"TR")

# For QRS complexes (BCG)
from facet.preprocessing import QRSTriggerDetector
QRSTriggerDetector()
```

### AAS Parameters

```python
AASCorrection(
    window_size=30,              # Number of epochs in sliding window
    correlation_threshold=0.975, # Correlation threshold (0-1)
    realign_after_averaging=True # Realign to averaged artifacts
)

# For highly non-stationary artifacts
AASCorrection(window_size=20, correlation_threshold=0.95)

# For very stable artifacts
AASCorrection(window_size=50, correlation_threshold=0.98)
```

### Parallel Processing

```python
# Use all CPU cores
result = pipeline.run(parallel=True, n_jobs=-1)

# Use specific number of cores
result = pipeline.run(parallel=True, n_jobs=4)

# Sequential (no parallelization)
result = pipeline.run(parallel=False)
```

## Troubleshooting

### No Triggers Found

```python
# Check your regex pattern
detector = TriggerDetector(regex=r"\b1\b", save_to_annotations=True)
context = detector.execute(context)

# Manually inspect annotations
raw = context.get_raw()
print(raw.annotations)
```

### Memory Issues

```python
# For large files, ensure preload is managed
loader = Loader(path="large_file.edf", preload=True)

# Or process in chunks (future feature)
```

### ANC Fails

```python
# Ensure AAS was run first
pipeline = Pipeline([
    # ...
    AASCorrection(window_size=30),  # Creates estimated_noise
    ANCCorrection(filter_order=5)   # Uses estimated_noise
])

# Check if C extension is available
from facet.correction import ANCCorrection
anc = ANCCorrection(use_c_extension=False)  # Force Python fallback
```

## Next Steps

- Read the [complete pipeline example](examples/complete_pipeline_example.py)
- Check [REFACTORING_COMPLETE.md](REFACTORING_COMPLETE.md) for detailed features
- See [CLAUDE.md](CLAUDE.md) for architecture details
- Write your own custom processors!

## Getting Help

- **Issues:** https://github.com/your-org/facetpy/issues
- **Discussions:** https://github.com/your-org/facetpy/discussions
- **Documentation:** https://facetpy.readthedocs.io (when available)

## Example: Full Correction Script

```python
#!/usr/bin/env python3
"""
Complete fMRI artifact correction script.
Usage: python correct.py input.edf output.edf
"""

import sys
from facet import create_standard_pipeline

def main():
    if len(sys.argv) != 3:
        print("Usage: python correct.py <input.edf> <output.edf>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print(f"Correcting {input_file}...")

    # Create and run pipeline
    pipeline = create_standard_pipeline(
        input_path=input_file,
        output_path=output_file,
        trigger_regex=r"\b1\b",
        use_anc=True,
        use_pca=False
    )

    result = pipeline.run(parallel=True)

    if result.success:
        print(f"\n✓ Success! Saved to {output_file}")
        print(f"  Execution time: {result.execution_time:.2f}s")

        # Print metrics
        metrics = result.context.metadata.custom.get('metrics', {})
        if 'snr' in metrics:
            print(f"  SNR: {metrics['snr']:.2f}")
        if 'rms_ratio' in metrics:
            print(f"  RMS improvement: {metrics['rms_ratio']:.2f}x")
    else:
        print(f"\n✗ Failed: {result.error}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Save as `correct.py` and run:
```bash
python correct.py data.edf corrected.edf
```

Happy correcting! 🎉
