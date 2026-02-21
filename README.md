<p align="center">
  <img src="docs/source/_static/logo.png" width="300" />
</p>

# FACETpy — fMRI Artifact Correction and Evaluation Toolbox

[![Documentation Status](https://readthedocs.org/projects/facetpy/badge/?version=latest)](https://facetpy.readthedocs.io/en/latest/?badge=latest)

A Python toolbox for correcting fMRI-induced EEG artifacts using Averaged
Artifact Subtraction (AAS) and other advanced methods.  Built on
[MNE-Python](https://mne.tools), it provides a modular pipeline
architecture that lets researchers process, evaluate, and compare correction
results with minimal code.

**Key features**

- Load EEG from EDF, GDF, and BIDS formats
- Artifact correction: AAS, PCA, Adaptive Noise Cancellation (ANC)
- Full evaluation suite: SNR, RMS, Median Artifact, FFT-based metrics
- One-call results: `result.print_metrics()`, `result.print_summary()`
- Batch processing across subjects/sessions with `Pipeline.map()`
- Generate synthetic EEG for algorithm testing
- Rich progress display in the terminal


## Quick start

```python
from facet import (
    Pipeline, EDFLoader, EDFExporter,
    TriggerDetector, UpSample, DownSample, AASCorrection,
)

pipeline = Pipeline([
    EDFLoader(path="data.edf", preload=True),
    TriggerDetector(regex=r"\b1\b"),
    UpSample(factor=10),
    AASCorrection(window_size=30),
    DownSample(factor=10),
    EDFExporter(path="corrected.edf", overwrite=True),
], name="Quickstart")

result = pipeline.run()
result.print_summary()   # Done in 4.2s  snr=18.3  rms_ratio=0.14
```


## Installation

Requires **Python ≥ 3.11** and [Poetry](https://python-poetry.org) ≥ 1.4.

```bash
# 1 — install Poetry (skip if already installed)
conda install -c conda-forge "poetry>=1.4"

# 2 — install FACETpy and all dependencies
poetry install

# 3 — activate the virtual environment
poetry shell
```

Optional extras:

```bash
poetry install -E deeplearning   # TensorFlow-based models
poetry install -E notebooks      # Jupyter notebook support
poetry install -E gui            # PyQt6 GUI components
poetry install -E all            # everything above
```


### Build the C extension (optional)

The fast Adaptive Noise Cancellation (ANC) step uses a compiled C extension.
Build it once after installing:

```bash
poetry run build-fastranc
```

If the extension is absent, ANC is skipped automatically and the rest of the
toolbox works normally.


## Running the examples

All examples are in the `examples/` folder and use the bundled
`NiazyFMRI.edf` dataset.  Run them from the project root:

```bash
# Recommended order for new users:
poetry run python examples/quickstart.py          # minimal pipeline
poetry run python examples/evaluation.py          # metrics & comparison
poetry run python examples/advanced_workflows.py  # conditional, parallel, factory
poetry run python examples/batch_processing.py    # multiple files at once
poetry run python examples/inline_steps.py        # custom steps & pipe operator
poetry run python examples/complete_pipeline_example.py  # full clinical pipeline
poetry run python examples/eeg_generation_visualization_example.py  # synthetic EEG
```


## Testing

```bash
# Run the full test suite
poetry run pytest

# Only fast unit tests (skip slow integration tests)
poetry run pytest -m "not slow"

# A single test file
poetry run pytest tests/test_core_pipeline.py -v

# With coverage report
poetry run pytest --cov=facet --cov-report=html
open htmlcov/index.html
```


## Documentation

```bash
# Build HTML docs
poetry run sphinx-build -b html docs/source docs/build

# Live-rebuild on every file save (requires sphinx-autobuild)
poetry run sphinx-autobuild docs/source docs/build

# Open locally
open docs/build/index.html
```

Full online documentation: https://facetpy.readthedocs.io/


## Project structure

```
src/facet/
├── core/           Pipeline, Processor, ProcessingContext, BatchResult
├── io/             EDFLoader, GDFLoader, BIDSLoader, EDFExporter, BIDSExporter
├── preprocessing/  Filters, Resample, TriggerDetector, Alignment, Transforms
├── correction/     AASCorrection, PCACorrection, ANCCorrection
├── evaluation/     SNRCalculator, RMSCalculator, MetricsReport, RawPlotter
├── misc/           EEGGenerator (synthetic data)
└── pipelines.py    create_standard_pipeline() factory

examples/
├── quickstart.py                         minimal pipeline
├── evaluation.py                         metrics & comparison
├── advanced_workflows.py                 conditional, parallel, factory
├── batch_processing.py                   multiple files
├── inline_steps.py                       custom steps & pipe operator
├── complete_pipeline_example.py          full clinical pipeline
└── eeg_generation_visualization_example.py  synthetic EEG
```


## License

GPLv3 — see `LICENSE` for details.

Author: Janik Michael Mueller
