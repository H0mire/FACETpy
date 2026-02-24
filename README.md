# FACETpy — fMRI Artifact Correction and Evaluation Toolbox



[Documentation Status](https://facetpy.readthedocs.io/en/latest/?badge=latest)

> A Python toolbox for correcting fMRI-induced EEG artifacts using Averaged
> Artifact Subtraction (AAS) and other advanced methods.  Built on
> [MNE-Python](https://mne.tools), it provides a modular pipeline
> architecture that lets researchers process, evaluate, and compare correction
> results with minimal code.

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

Requires **Python 3.11 or 3.12** and [Poetry](https://python-poetry.org) >= 1.4.
Conda is optional, not required.

### Option A (recommended): system Python + Poetry

```bash
# 1 — verify Python
python3 --version

# 2 — install Poetry (pick one)
pipx install poetry
# or: python3 -m pip install --user poetry

# 3 — install FACETpy and dependencies
poetry install --no-interaction
```

### Option B (optional): Conda workflow

```bash
conda create -n facetpy python=3.12 -y
conda activate facetpy
conda install -c conda-forge poetry -y
poetry install --no-interaction
```

Optional extras:

```bash
poetry install -E deeplearning   # TensorFlow-based models
poetry install -E notebooks      # Jupyter notebook support
poetry install -E gui            # PyQt6 GUI components
poetry install -E docs           # Sphinx documentation toolchain
poetry install -E all            # everything above
```

Run commands with `poetry run ...` (for example, `poetry run python examples/quickstart.py`).

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
# Install docs dependencies
poetry install -E docs

# Build HTML docs
poetry run sphinx-build -b html docs/source docs/build

# Open locally
open docs/build/index.html
```

Full online documentation: [https://facetpy.readthedocs.io/](https://facetpy.readthedocs.io/)

For comprehensive build instructions, theme configuration, and contribution guidelines see `[docs/README.md](docs/README.md)`.

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


| Task                                      | Shortcut          | Description                                                         |
| ----------------------------------------- | ----------------- | ------------------------------------------------------------------- |
| **Test: Run All**                         | default test task | Full test suite with coverage report                                |
| **Test: Run Current File**                |                   | Run pytest on the file open in the editor                           |
| **Test: Unit Only**                       |                   | Only tests marked `@pytest.mark.unit`                               |
| **Test: Integration Only**                |                   | Only tests marked `@pytest.mark.integration`                        |
| **Test: Skip Slow**                       |                   | All tests except those marked `@pytest.mark.slow`                   |
| **Test: Show Coverage Report**            |                   | Open `htmlcov/index.html` in the browser                            |
| **Lint: Check (Ruff)**                    |                   | Check `src/` and `tests/` for lint errors                           |
| **Lint: Fix (Ruff)**                      |                   | Auto-fix lint errors in place                                       |
| **Format: Check (Ruff)**                  |                   | Verify formatting without changing files                            |
| **Format: Apply (Ruff)**                  |                   | Apply ruff formatting to `src/` and `tests/`                        |
| **Build: FastRANC C Extension**           |                   | Compile the FastRANC C extension                                    |
| **Build: Install Dependencies**           |                   | `poetry install`                                                    |
| **Build: Install All Extras**             |                   | `poetry install -E all`                                             |
| **Build: Update Dependencies**            |                   | `poetry update`                                                     |
| **Docs: Build HTML**                      |                   | Build Sphinx documentation                                          |
| **Docs: Open in Browser**                 |                   | Open the built docs in the browser                                  |
| **Docs: Build & Open**                    |                   | Build docs and open immediately                                     |
| **Run: Current Python File**              |                   | Execute the file open in the editor                                 |
| **Review: Uncommitted Changes (Codex)**   |                   | Codex AI review of all local changes                                |
| **Review: Against Branch (Codex)**        |                   | Codex AI review against a selected base branch (prompts for branch) |
| **QA: Full Check (Lint + Format + Test)** |                   | Lint + format check + full test suite in sequence                   |


## License

GPLv3 — see `LICENSE` for details.

Author: Janik Michael Mueller