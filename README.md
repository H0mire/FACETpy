<p align="center">
  <a href="https://facetpy.readthedocs.io/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/H0mire/facetpy/main/docs/source/_static/logo_dark_theme.png">
      <img src="https://raw.githubusercontent.com/H0mire/facetpy/main/docs/source/_static/logo_light_theme.png" alt="FACETpy logo" width="300">
    </picture>
  </a>
</p>

<h3 align="center">FACETpy - EEG-Data Correction Framework</h3>

<p align="center">
  A Python toolbox for correcting EEG artifacts using Averaged Artifact Subtraction (AAS) and other advanced methods. Built on MNE-Python.
  <br>
  <a href="https://facetpy.readthedocs.io/"><strong>Explore FACETpy docs »</strong></a>
  <br>
  <br>
  <a href="https://github.com/H0mire/facetpy/issues/new?assignees=&labels=bug">Report bug</a>
  ·
  <a href="https://github.com/H0mire/facetpy/issues/new?assignees=&labels=feature">Request feature</a>
  ·
  <a href="https://facetpy.readthedocs.io/">Documentation</a>
</p>

[![Documentation Status](https://readthedocs.org/projects/facetpy/badge/?version=latest)](https://facetpy.readthedocs.io/en/latest/?badge=latest)

Built on [MNE-Python](https://mne.tools), FACETpy provides a modular pipeline architecture that lets researchers process, evaluate, and compare correction results with minimal code.

**Key features**

- Load EEG from EDF, GDF, and BIDS formats
- Artifact correction: AAS, PCA, Adaptive Noise Cancellation (ANC)
- Full evaluation suite: SNR, RMS, Median Artifact, FFT-based metrics
- One-call results: `result.print_metrics()`, `result.print_summary()`
- Batch processing across subjects/sessions with `Pipeline.map()`
- Generate synthetic EEG for algorithm testing
- Rich progress display in the terminal


## Quick start

Quick installation from PyPI (requires Python 3.11/3.12/3.13):

```bash
pip install facetpy
```

Strongly recommended for fast ANC performance:

```bash
python -m facet.build
```

For the Full Setup + Early Access to features  (Repository, Examples, Contributing):  
Unix (WSL/macOS/Linux) - bootstrap shortcut:

```bash
curl -fsSL https://raw.githubusercontent.com/H0mire/facetpy/main/scripts/bootstrap.sh | sh
cd facetpy
```

## Preview

<p align="center">
  <img src="https://raw.githubusercontent.com/H0mire/facetpy/main/docs/source/_static/run_example.png" alt="FACETpy example run" width="700" />
</p>


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

Requires **Python 3.11, 3.12, or 3.13**.
For normal usage, Poetry is not required.

### Normal usage (recommended): install from PyPI

```bash
pip install facetpy
```

The package name on PyPI is `facetpy`; import it in Python as `facet`.

### Contributing setup (source + Poetry)

Poetry is required for contribution workflows (tests, linting, docs).

Unix (macOS/Linux) - bootstrap shortcut (installs poetry):

```bash
curl -fsSL https://raw.githubusercontent.com/H0mire/facetpy/main/scripts/bootstrap.sh | sh
cd facetpy
```

Unix (macOS/Linux) - existing clone (installs peotry):

```bash
./scripts/install.sh
```

Other platforms (including Windows) with Poetry installed:

```bash
git clone https://github.com/H0mire/facetpy.git
cd facetpy
poetry install --no-interaction
```

The Unix `./scripts/install.sh` script:
- checks for Python 3.11/3.12/3.13
- checks whether Poetry is installed
- asks whether Poetry should be installed if missing
- runs `poetry install --no-interaction`

The bootstrap script:
- clones FACETpy into `./facetpy`
- runs `./scripts/install.sh` inside that clone

Manual Poetry installation (contributors):

```bash
# 1 — verify Python
python --version

# 2 — install Poetry (pick one)
pipx install poetry
# or: pip install --user poetry

# 3 — install repository dependencies
poetry install --no-interaction
```

Optional contributor extras:
```text
poetry install -E deeplearning   # TensorFlow-based models
poetry install -E notebooks      # Jupyter notebook support
poetry install -E gui            # PyQt6 GUI components
poetry install -E docs           # Sphinx documentation toolchain
poetry install -E all            # everything above
```

Run contributor commands with `poetry run ...` (for example, `poetry run pytest`).


### Build the C extension (strongly recommended for ANC)

The fast Adaptive Noise Cancellation (ANC) path is significantly faster with
the compiled FastRANC C extension. Build it once after installing.

Without Poetry:

```bash
python -m facet.build
```

With Poetry:

```bash
poetry run build-fastranc
```

If the extension is not compiled, ANC uses a slower Python fallback and the
rest of the toolbox still works.


## Running the examples

All examples are in the `examples/` folder and use the bundled
`NiazyFMRI.edf` dataset.

To run repository examples, clone the repository and install FACETpy in your
active Python environment (no Poetry required):

```bash
git clone https://github.com/H0mire/facetpy.git
cd facetpy
pip install facetpy
```

Then run from the project root:

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

```bash
# Run the full test suite
poetry run pytest

# Only fast unit tests (skip slow integration tests)
poetry run pytest -m "not slow"

# A single test file
poetry run pytest tests/test_core_pipeline.py -v

# With coverage report
poetry run pytest --cov=facet --cov-report=html
```

Open the coverage report:

```bash
python -m webbrowser htmlcov/index.html
```


## Documentation

```bash
# Install docs dependencies
poetry install -E docs

# Build HTML docs
poetry run sphinx-build -b html docs/source docs/build

```

Open docs locally:

```bash
python -m webbrowser docs/build/index.html
```

Full online documentation: https://facetpy.readthedocs.io/

For comprehensive build instructions, theme configuration, and contribution guidelines see [`docs/README.md`](docs/README.md).
For PyPI release steps, see [`RELEASING.md`](RELEASING.md).


## Contributing

Contributing uses the source/Poetry workflow. See installation guide above.  
Use `./scripts/install.sh` on Unix, otherwise run `poetry install --no-interaction`.
Follow [`docs/source/development/contributing.rst`](docs/source/development/contributing.rst) for the full setup and checks.


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
| **Build: FastRANC C Extension** | | Compile the FastRANC C extension |
| **Build: Install Dependencies** | | `poetry install` |
| **Build: Install All Extras** | | `poetry install -E all` |
| **Build: Update Dependencies** | | `poetry update` |
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
