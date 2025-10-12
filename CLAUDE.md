# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FACETpy is a Python toolbox for correcting EEG artifacts in simultaneous EEG-fMRI recordings using Averaged Artifact Subtraction (AAS). The package provides advanced artifact detection, correction, and evaluation capabilities built on top of MNE-Python.

## Development Commands

### Installation and Setup

```bash
# Install dependencies using Poetry (recommended)
poetry install

# Activate virtual environment
poetry shell

# Build C extension for Fast RAN correction
poetry run build-fastranc
# Or directly: python -m facet.build
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_facet.py

# Run with verbose output
pytest -v
```

### Documentation

```bash
# Navigate to docs directory
cd docs

# Build HTML documentation
make html

# Auto-rebuild on changes (live reload)
make livehtml
```

The compiled documentation will be in `docs/build/html/`.

### Building the C Extension

The package includes a C extension (`fastranc.c`) for fast Adaptive Noise Cancellation:

```bash
# Use the Poetry script
poetry run build-fastranc

# Or use the module directly
python -m facet.build
```

This compiles `src/facet/helpers/fastranc.c` into a platform-specific shared library (`.so` on Linux, `.dylib` on macOS, `.dll` on Windows).

## Architecture Overview

### Core Components

**Main API (`facet.py`)**: The `facet` class is the primary interface. It orchestrates three framework components and manages the EEG data lifecycle. Users interact primarily with this class.

**EEG Data Object (`eeg_obj.py`)**: The `EEG` class encapsulates all EEG data and metadata, including:
- `mne_raw`: Current processed MNE Raw object
- `mne_raw_orig`: Original unprocessed MNE Raw object for reference
- `estimated_noise`: Accumulated noise estimates from corrections
- `loaded_triggers`: Trigger positions marking artifact occurrences
- Artifact timing parameters (`artifact_length`, `artifact_to_trigger_offset`, etc.)
- Filter parameters for ANC and PCA preprocessing

### Three Framework System

The package uses a separation-of-concerns architecture with three specialized frameworks:

#### 1. AnalysisFramework (`frameworks/analysis.py`)
Handles data I/O and trigger detection:
- Imports EEG from multiple formats (EDF, GDF, BIDS, EEGLAB)
- Exports processed data to various formats
- Finds triggers using regex patterns or QRS detection
- Detects and adds missing triggers automatically
- Derives critical processing parameters (artifact length, filter settings)
- Manages annotations and events

Key method: `derive_parameters()` calculates artifact length, timing boundaries, and filter parameters based on detected triggers.

#### 2. CorrectionFramework (`frameworks/correction.py`)
Implements artifact correction algorithms:
- **Averaged Artifact Subtraction (AAS)**: Calculates weighted averaging matrices using correlation-based epoch selection
- **Motion-based correction**: Uses fMRI realignment parameters via Moosmann method
- **Adaptive Noise Cancellation (ANC)**: Applies fast RAN-C algorithm (C implementation)
- **PCA-based artifact removal**: Removes periodic artifacts using principal component analysis
- **Trigger alignment**: Aligns triggers and performs subsample alignment
- Filtering and resampling operations

Key workflow:
1. `prepare()` - Prepares data for correction
2. `calc_matrix_aas()` or `calc_matrix_motion()` - Computes averaging weights
3. `remove_artifacts()` - Subtracts averaged artifacts
4. `apply_ANC()` - Applies adaptive filtering
5. `apply_PCA()` - Removes residual periodic artifacts

#### 3. EvaluationFramework (`frameworks/evaluation.py`)
Provides metrics for assessing correction quality:
- **SNR**: Signal-to-Noise Ratio (comparing artifact vs. clean segments)
- **RMS**: Root Mean Square ratio (before/after correction)
- **RMS2**: RMS ratio relative to clean reference
- **MEDIAN**: Median peak-to-peak artifact amplitude

Evaluation compares corrected data against clean reference periods (time windows without artifacts).

### Deep Learning Module (`frameworks/deeplearning.py`)

The package includes experimental deep learning-based artifact removal:

- **ManifoldAutoencoder**: A convolutional autoencoder that learns artifact manifolds
- **ArtifactEstimator**: High-level interface for training and applying deep learning models

The autoencoder learns to predict artifacts by training on paired clean/noisy epochs, enabling artifact removal without relying on averaging assumptions. Supports both TensorFlow and PyTorch implementations.

### Helpers and Utilities

- `helpers/fastranc.c` and `helpers/fastranc.py`: Fast C-based adaptive noise cancellation
- `helpers/bcg_detector.py`: QRS detection for BCG artifact triggers
- `helpers/moosmann.py`: Motion-based artifact correction using fMRI parameters
- `helpers/crosscorr.py`: Cross-correlation for trigger alignment
- `helpers/alignsubsample.py`: Subsample-level trigger alignment
- `utils/facet_result.py`: Result container for storing correction outputs
- `utils/i18n.py`: Internationalization support

## Typical Processing Pipeline

```python
# 1. Import and initialize
f = facet()
f.import_eeg(path, fmt='edf', upsampling_factor=10,
             artifact_to_trigger_offset=-0.005)

# 2. Find triggers (artifact occurrences)
f.find_triggers(event_regex)
f.find_missing_triggers()  # Optional: detect missing triggers

# 3. Preprocessing
f.highpass(1)
f.upsample()

# 4. Align triggers for precise artifact localization
f.align_triggers(ref_trigger=0)
f.align_subsample(ref_trigger=0)  # Optional: subsample alignment

# 5. Calculate and remove artifacts
f.calc_matrix_aas()  # Or f.calc_matrix_motion(file_path)
f.remove_artifacts()

# 6. Post-processing
f.get_correction().apply_PCA()  # Optional: remove residual artifacts
f.downsample()
f.lowpass(70)
f.apply_ANC()

# 7. Evaluate results
results = f.evaluate(measures=["SNR", "RMS", "MEDIAN"])
f.plot([results])

# 8. Export
f.export_eeg('output.edf', fmt='edf')
```

## Important Implementation Details

### Upsampling Strategy
The package uses aggressive upsampling (typically 10x) during processing to achieve subsample-level precision in trigger alignment. This is critical for accurate artifact subtraction. Data is downsampled after correction.

### Trigger Alignment
Trigger positions are refined using cross-correlation to account for:
- Scanner clock drift
- Asynchronous sampling between EEG and MRI systems
- Subsample timing variations

The `align_triggers()` method adjusts trigger positions, and `align_subsample()` performs phase-shift-based fine alignment.

### Adaptive Averaging Windows
The `calc_matrix_aas()` method uses a sliding window approach where artifact templates are averaged only from highly correlated epochs (correlation > 0.975 by default). This adapts to non-stationary artifacts and subject motion.

### Memory-Efficient Processing
For large datasets, use `apply_per_channel()` which processes one channel at a time to reduce memory usage. This is especially important when dealing with high sampling rates or long recordings.

### Volume vs. Slice Triggers
The code handles both volume triggers (one per fMRI volume) and slice triggers (one per slice acquisition). The `_check_volume_gaps()` method automatically detects which mode is present based on trigger spacing.

## Key Concepts

**Artifact Window**: The time segment containing an artifact, defined by `artifact_to_trigger_offset` and `artifact_length`. The offset is typically negative (e.g., -0.005s) to account for trigger delay.

**Acquisition Window**: The time range containing all artifacts plus padding (`time_acq_start` to `time_acq_end`). Padding ensures filter edge effects don't contaminate artifact regions.

**Estimated Noise**: Accumulated artifact estimates stored in `EEG.estimated_noise`. This is used as a reference signal for ANC and can be analyzed separately.

**Correlation Threshold**: Used in `calc_chosen_matrix()` to determine which epochs are similar enough to include in artifact averaging (default: 0.975).

## Data Format Support

- **BIDS**: Full support for Brain Imaging Data Structure format
- **EDF/GDF**: European Data Format and General Data Format
- **EEGLAB**: EEGLAB .set files

All formats are imported via MNE-Python, so any format supported by MNE can potentially work.

## Testing

The main test file is `tests/test_facet.py`. When adding new features:
- Add unit tests for new correction algorithms
- Test with both simulated and real EEG-fMRI data
- Verify that evaluation metrics show improvement after correction

## Common Pitfalls

1. **Incorrect artifact_to_trigger_offset**: If triggers don't align with artifact peaks, correction will fail. Use `plot_eeg()` to visually inspect trigger placement.

2. **Missing triggers**: Run `find_missing_triggers()` if periodic artifacts are present but not all triggers were detected.

3. **Memory issues**: Use `preload=False` when importing large files, then process with `apply_per_channel()`.

4. **Filter order warnings**: The automatic filter parameter derivation can produce very high filter orders for slow sampling rates. Check logs for warnings.

5. **Shape mismatches in deep learning**: Ensure clean and noisy data have identical shapes and trigger counts before training the autoencoder.

## Code Style

- Use `loguru` for logging (already imported as `logger` in most modules)
- MNE objects use 0-based channel indexing
- Trigger positions are stored as integer sample indices
- Time values are in seconds, sample indices are integers
- Private methods start with `_` (e.g., `_derive_art_length()`)
