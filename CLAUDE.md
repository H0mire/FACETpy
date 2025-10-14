# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FACETpy is a Python toolbox for correcting EEG artifacts in simultaneous EEG-fMRI recordings using Averaged Artifact Subtraction (AAS) and other advanced correction methods. The project is built on MNE-Python and provides a modular, pipeline-based architecture for flexible EEG processing.

**Key Features:**
- Import EEG data from various formats (EDF, GDF, BIDS)
- Advanced artifact detection and correction using AAS, ANC, and PCA
- Comprehensive evaluation framework with SNR, RMS, and other metrics
- Flexible processing pipeline with support for parallel execution
- Built-in trigger detection and alignment algorithms

## Development Commands

### Environment Setup
```bash
# Install Poetry 1.4+ if needed
conda install -c conda-forge poetry=1.4

# Install project dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Building C Extensions
The project includes a C extension for fast adaptive noise cancellation (ANC):
```bash
# Compile the FastRANC C extension
poetry run build-fastranc

# Or manually via the script defined in pyproject.toml
python -m facet.build
```

Note: The C extension (libfastranc.dylib/so/dll) is optional. If not built, ANC features will be unavailable but the rest of the toolbox works normally.

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_core_pipeline.py

# Run with markers
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run a specific test function
pytest tests/test_core_pipeline.py::test_pipeline_execution -v
```

Test fixtures are defined in `tests/conftest.py` and provide sample EEG data, contexts, and processors.

### Documentation
```bash
# Navigate to docs directory
cd docs

# Build HTML documentation
make html

# Auto-rebuild on changes (if available)
make livehtml

# View docs
open build/html/index.html
```

Documentation is built with Sphinx and uses the RTD theme. Source files are in `docs/source/`.

## Architecture

### Core Architecture (v2.0 - New Modular System)

The codebase has undergone a major refactoring to a modular, processor-based architecture:

**Core Components (`src/facet/core/`):**

1. **Processor (`processor.py`)** - Base class for all processing operations
   - All processors inherit from `Processor` ABC
   - Implement `process(context) -> context` method
   - Built-in validation via `validate()` method
   - Automatic history tracking in context
   - Flags: `requires_triggers`, `requires_raw`, `modifies_raw`, `parallel_safe`
   - Composite processors: `SequenceProcessor`, `ConditionalProcessor`, `SwitchProcessor`

2. **ProcessingContext (`context.py`)** - Immutable context passed between processors
   - Wraps MNE Raw object and metadata
   - Stores processing history (provenance tracking)
   - Methods: `get_raw()`, `with_raw()`, `has_triggers()`, `get_metadata()`
   - Includes `ProcessingMetadata` for triggers, artifact info, custom data

3. **Pipeline (`pipeline.py`)** - Orchestrates processor execution
   - Sequential execution of processors with validation
   - Error handling and progress tracking
   - Parallel execution support via `ParallelExecutor`
   - Returns `PipelineResult` with context, success status, timing
   - Methods: `run()`, `add()`, `insert()`, `remove()`, `validate_all()`, `describe()`

4. **Registry (`registry.py`)** - Global processor registry
   - Functions: `register_processor()`, `get_processor()`, `list_processors()`
   - Enables processor lookup by name for serialization/config

5. **ParallelExecutor (`parallel.py`)** - Parallel processing support
   - Channel-wise parallelization for compatible processors
   - Uses joblib for multiprocessing

### Module Organization

```
src/facet/
├── core/              # Core pipeline infrastructure
├── io/                # Data loaders and exporters (EDF, BIDS, GDF)
├── preprocessing/     # Preprocessing processors (filters, resampling, triggers, alignment)
├── correction/        # Correction algorithms (AAS, ANC, PCA)
├── evaluation/        # Evaluation metrics (SNR, RMS, median artifact)
├── helpers/           # Utility functions and C extensions
├── resources/         # Translations and resources
├── eeg_obj.py        # Legacy EEG object wrapper
└── facet.py          # Legacy API (maintained for backwards compatibility)
```

### Legacy API (`facet.py`)

The old API based on the `facet` class is still available for backwards compatibility but is not the recommended approach for new code. It uses a framework-based architecture with `AnalysisFramework`, `CorrectionFramework`, and `EvaluationFramework`.

### Pipeline Pattern (Recommended)

The v2.0 architecture uses a pipeline pattern where:
1. Each processing step is a `Processor` subclass
2. Processors receive a `ProcessingContext` and return a new context
3. Contexts are immutable and track history
4. Pipelines compose processors into workflows
5. Results are wrapped in `PipelineResult` objects

Example:
```python
from facet.core import Pipeline
from facet.io import EDFLoader, EDFExporter
from facet.preprocessing import TriggerDetector, UpSample
from facet.correction import AASCorrection

pipeline = Pipeline([
    EDFLoader(path="data.edf", preload=True),
    TriggerDetector(regex=r"\b1\b"),
    UpSample(factor=10),
    AASCorrection(window_size=30),
    EDFExporter(path="output.edf")
], name="Basic Pipeline")

result = pipeline.run()
```

See `examples/complete_pipeline_example.py` for comprehensive examples including parallel execution, conditional processing, and batch workflows.

## Important Implementation Notes

### Processor Development
When creating new processors:
- Inherit from `Processor` in `facet.core`
- Implement `process(context) -> context` method
- Override `validate(context)` for prerequisite checks
- Set class attributes: `name`, `description`, `requires_*`, `parallel_safe`
- Initialize parameters in `__init__` and call `super().__init__()`
- Use `context.with_raw(new_raw)` to return modified context
- Never modify context in-place; always return new context

### Context Management
- Contexts are passed through pipelines and track all operations
- Use `context.get_raw()` to access MNE Raw object
- Use `context.metadata.triggers` for trigger positions
- Store custom data in `context.metadata.custom` dict
- Processing history is automatically tracked via `add_history_entry()`

### MNE Integration
- All EEG data is stored as MNE Raw objects (`mne.io.Raw`)
- Always use `.copy()` when modifying Raw objects to avoid in-place changes
- Sampling frequency: `raw.info['sfreq']`
- Access data: `raw._data` (shape: [n_channels, n_samples])
- Annotations are used for triggers: `raw.annotations`

### Trigger Handling
- Triggers are stored as sample positions (integers) in `context.metadata.triggers`
- Trigger detection uses regex patterns on annotation descriptions
- Common pattern: `r"\b1\b"` for trigger value "1"
- After upsampling, triggers must be scaled: `triggers * upsample_factor`

### Parallel Processing
- Set `parallel_safe = True` on processors that can be parallelized
- Channel-wise operations are good candidates for parallelization
- Use `pipeline.run(parallel=True, n_jobs=-1)` to enable
- Not all processors support parallelization (e.g., I/O operations)

### Testing Patterns
- Use fixtures from `tests/conftest.py` for sample data
- `sample_raw`: Basic MNE Raw object
- `sample_raw_with_artifacts`: Raw with simulated fMRI artifacts
- `sample_context`: ProcessingContext with triggers and metadata
- `sample_edf_file`: Temporary EDF file for I/O tests
- Mark tests with `@pytest.mark.unit`, `@pytest.mark.integration`, etc.

## Common Patterns

### Creating a Standard Correction Pipeline
```python
import facet

# Using convenience function
pipeline = facet.create_standard_pipeline(
    "input.edf",
    "output.edf",
    trigger_regex=r"\b1\b",
    upsample_factor=10
)
result = pipeline.run()
```

### Custom Processor Example
```python
from facet.core import Processor, ProcessingContext

class CustomFilter(Processor):
    name = "custom_filter"
    requires_raw = True
    modifies_raw = True

    def __init__(self, cutoff: float):
        self.cutoff = cutoff
        super().__init__()

    def validate(self, context: ProcessingContext):
        super().validate(context)
        # Add custom validation

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        # Apply custom filtering
        raw.filter(l_freq=None, h_freq=self.cutoff)
        return context.with_raw(raw)
```

### Conditional Processing
```python
from facet.core import ConditionalProcessor, Pipeline

def needs_extra_correction(ctx):
    snr = ctx.metadata.custom.get('snr', float('inf'))
    return snr < 10

pipeline = Pipeline([
    # ... initial steps ...
    SNRCalculator(),
    ConditionalProcessor(
        condition=needs_extra_correction,
        processor=PCACorrection(n_components=0.95)
    ),
    # ... final steps ...
])
```

## File Paths and Structure

- Main source: `src/facet/`
- Tests: `tests/` with fixtures in `conftest.py`
- Examples: `examples/complete_pipeline_example.py`
- Documentation: `docs/source/` (Sphinx RST format)
- C extension source: `src/facet/helpers/fastranc.c`
- Build script: `src/facet/build.py`

## Dependencies

Key dependencies (from `pyproject.toml`):
- Python ^3.11
- MNE 1.6.0 (EEG/MEG processing)
- NumPy 2.1.3 (numerical operations)
- SciPy ^1.15.3 (signal processing)
- pandas ^2.2.3 (data handling)
- matplotlib ^3.10.3 (plotting)
- scikit-learn ^1.4.2 (PCA)
- neurokit2 ^0.2.7 (BCG detection)
- TensorFlow ^2.19.0 (deep learning features)

Dev dependencies:
- pytest ^8.1.1 (testing)
- Sphinx ^7.2.6 (documentation)
- sphinx-rtd-theme ^2.0.0

## Version and Licensing

- Current version: 2.0.0
- License: GPLv3
- Author: Janik Michael Mueller
- Documentation: https://facetpy.readthedocs.io/
