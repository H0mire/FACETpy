<p align="center">
  <picture>
    <source srcset="docs/source/_static/logo_dark_theme.png" media="(prefers-color-scheme: dark)">
    <img src="docs/source/_static/logo_light_theme.png" width="300" alt="FACETpy logo" />
  </picture>
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
    Pipeline, Loader, EDFExporter,
    TriggerDetector, UpSample, DownSample, AASCorrection,
)

pipeline = Pipeline([
    Loader(path="data.edf", preload=True),
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

Requires **Python 3.11 or 3.12**.  No additional package manager is required — standard `pip` works.

### Option A (recommended): pip + virtualenv
```bash
# 1 — create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2 — install FACETpy and all dependencies
pip install .
```

### Option B (optional): Conda workflow
```bash
conda create -n facetpy python=3.12 -y
conda activate facetpy
pip install .
```

Optional extras (deep learning, notebooks, GUI, docs):
```bash
pip install ".[deeplearning]"   # TensorFlow-based models
pip install ".[notebooks]"      # Jupyter notebook support
pip install ".[gui]"            # PyQt6 GUI components
pip install ".[docs]"           # Sphinx documentation toolchain
pip install ".[all]"            # everything above
```

For an editable (development) install that picks up source changes immediately:
```bash
pip install -e ".[dev]"
```

> **Poetry users** — the `pyproject.toml` is still fully Poetry-compatible.
> Running `poetry install` continues to work as before.


### Build the C extension (optional)

The fast Adaptive Noise Cancellation (ANC) step uses a compiled C extension.
Build it once after installing:

```bash
python -m facet.build
# or, if installed via Poetry: poetry run build-fastranc
```

If the extension is absent, ANC is skipped automatically and the rest of the
toolbox works normally.


## Running the examples

All examples are in the `examples/` folder and use the bundled
`NiazyFMRI.edf` dataset.  Run them from the project root (with your venv active):

```bash
# Recommended order for new users:
python examples/quickstart.py          # minimal pipeline
python examples/evaluation.py          # metrics & comparison
python examples/advanced_workflows.py  # conditional, parallel, factory
python examples/batch_processing.py    # multiple files at once
python examples/inline_steps.py        # custom steps & pipe operator
python examples/complete_pipeline_example.py  # full clinical pipeline
python examples/eeg_generation_visualization_example.py  # synthetic EEG
```


## Testing

First install the dev dependencies:
```bash
pip install -e ".[dev]"
# or: pip install -r requirements-dev.txt && pip install -e .
```

```bash
# Run the full test suite
pytest

# Only fast unit tests (skip slow integration tests)
pytest -m "not slow"

# A single test file
pytest tests/test_core_pipeline.py -v

# With coverage report
pytest --cov=facet --cov-report=html
open htmlcov/index.html
```


## Documentation

```bash
# Install docs dependencies
pip install ".[docs]"
# or: pip install -r requirements-docs.txt && pip install -e .

# Build HTML docs
sphinx-build -b html docs/source docs/build

# Open locally
open docs/build/index.html
```

Full online documentation: https://facetpy.readthedocs.io/

For comprehensive build instructions, theme configuration, and contribution guidelines see [`docs/README.md`](docs/README.md).


## Project structure

```
src/facet/
├── core/           Pipeline, Processor, ProcessingContext, BatchResult
├── io/             Loader, BIDSLoader, EDFExporter, BIDSExporter
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


## VS Code Tasks

Tasks are defined in `.vscode/tasks.json` and can be run via **Ctrl+Shift+P** → **Tasks: Run Task**.

| Task | Shortcut | Description |
|---|---|---|
| **Test: Run All** | default test task | Full test suite with coverage report |
| **Test: Run Current File** | | Run pytest on the file open in the editor |
| **Test: Unit Only** | | Only tests marked `@pytest.mark.unit` |
| **Test: Integration Only** | | Only tests marked `@pytest.mark.integration` |
| **Test: Skip Slow** | | All tests except those marked `@pytest.mark.slow` |
| **Test: Show Coverage Report** | | Open `htmlcov/index.html` in the browser |
| **Lint: Check (Ruff)** | | Check `src/` and `tests/` for lint errors |
| **Lint: Fix (Ruff)** | | Auto-fix lint errors in place |
| **Format: Check (Ruff)** | | Verify formatting without changing files |
| **Format: Apply (Ruff)** | | Apply ruff formatting to `src/` and `tests/` |
| **Build: FastRANC C Extension** | | Compile via `python -m facet.build` |
| **Build: Install Dependencies** | | `pip install -e .` |
| **Build: Install All Extras** | | `pip install -e ".[all]"` |
| **Build: Install Dev Dependencies** | | `pip install -e ".[dev]"` (includes pytest, ruff) |
| **Docs: Build HTML** | | Build Sphinx documentation |
| **Docs: Open in Browser** | | Open the built docs in the browser |
| **Docs: Build & Open** | | Build docs and open immediately |
| **Run: Current Python File** | | Execute the file open in the editor |
| **Review: Uncommitted Changes (Codex)** | | Codex AI review of all local changes |
| **Review: Against Branch (Codex)** | | Codex AI review against a selected base branch (prompts for branch) |
| **QA: Full Check (Lint + Format + Test)** | | Lint + format check + full test suite in sequence |

## License

GPLv3 — see `LICENSE` for details.

Author: Janik Michael Mueller
