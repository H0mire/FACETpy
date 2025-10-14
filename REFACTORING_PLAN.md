# FACETpy Refactoring Plan

## Executive Summary

This document outlines a comprehensive refactoring of FACETpy to transform it from an experimental codebase into a professional, modular, and extensible EEG artifact correction toolbox.

## Current Pain Points

### Architecture Issues
1. **Tight Coupling**: The three frameworks (Analysis, Correction, Evaluation) are tightly coupled through the main `facet` class
2. **God Object Anti-pattern**: The `facet` class acts as a facade that delegates to frameworks, creating unnecessary indirection
3. **Implicit Dependencies**: Processing steps have hidden dependencies and ordering requirements
4. **No Pipeline Abstraction**: The workflow is hardcoded in user scripts, not declarative
5. **Poor Separation of Concerns**: Data, processing logic, and orchestration are mixed

### Usability Issues
1. **Rigid Workflow**: Users must follow a specific sequence of method calls
2. **Non-discoverable API**: No clear indication of which methods can be called when
3. **Inconsistent State Management**: EEG object mutates throughout processing
4. **Limited Flexibility**: Hard to skip steps, substitute algorithms, or customize behavior
5. **Poor Error Messages**: Failures don't clearly indicate what went wrong or how to fix it

### Developer Experience Issues
1. **Monolithic Files**: `correction.py` has 1000+ lines, `analysis.py` has 900+ lines
2. **No Plugin System**: Adding new correction methods requires modifying core files
3. **Mixed Concerns**: I/O, computation, and business logic intertwined
4. **No Testing Infrastructure**: Hard to unit test individual processing steps
5. **No Type Hints**: Makes IDE support and refactoring difficult

### Performance Issues
1. **No Multiprocessing**: All processing is single-threaded
2. **Memory Inefficient**: Loads entire datasets even for channel-wise operations
3. **C Extension Integration**: Current fastranc integration is primitive

## Proposed Architecture

### Core Principles

1. **Pipeline-Based Processing**: Declarative workflows using composable steps
2. **Plugin Architecture**: Easy registration of new processing steps
3. **MNE-First Design**: Leverage MNE's Raw, Epochs, and Evoked objects maximally
4. **Lazy Evaluation**: Compute only when needed, enable parallel execution
5. **Immutability by Default**: Processors create new objects rather than mutating
6. **Type Safety**: Full type hints for better IDE support and validation

### New Structure

```
src/facet/
├── core/
│   ├── __init__.py
│   ├── pipeline.py          # Pipeline execution engine
│   ├── processor.py         # Base processor interface
│   ├── context.py           # Processing context (replaces EEG object)
│   ├── registry.py          # Plugin registry
│   └── parallel.py          # Multiprocessing support
├── io/
│   ├── __init__.py
│   ├── loaders.py           # Data loading processors
│   ├── exporters.py         # Data export processors
│   └── formats.py           # Format-specific handlers
├── preprocessing/
│   ├── __init__.py
│   ├── filtering.py         # Filter processors
│   ├── resampling.py        # Resampling processors
│   ├── trigger_detection.py # Trigger finding processors
│   └── alignment.py         # Trigger alignment processors
├── correction/
│   ├── __init__.py
│   ├── aas.py              # Averaged artifact subtraction
│   ├── anc.py              # Adaptive noise cancellation
│   ├── pca.py              # PCA-based correction
│   ├── motion.py           # Motion-based correction
│   └── deeplearning.py     # DL-based correction
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metric processors
│   └── visualization.py    # Plotting processors
├── extensions/
│   ├── __init__.py
│   ├── native.py           # C extension loading utilities
│   └── fastranc/           # Fast RAN-C C extension
│       ├── __init__.py
│       ├── wrapper.py
│       └── fastranc.c
├── config/
│   ├── __init__.py
│   ├── schema.py           # Configuration validation
│   └── presets.py          # Preset configurations
├── utils/
│   ├── __init__.py
│   ├── validation.py       # Input validation
│   ├── logging.py          # Logging setup
│   └── mne_helpers.py      # MNE utility functions
└── legacy/
    └── (old API for backwards compatibility)
```

## New API Design

### Pipeline-Based API

```python
from facet import Pipeline
from facet.io import BIDSLoader, EDFExporter
from facet.preprocessing import TriggerDetector, UpSample, HighPassFilter
from facet.correction import AveragedArtifactSubtraction, AdaptiveNoiseCancellation
from facet.evaluation import SNRMetric, RMSMetric

# Define pipeline declaratively
pipeline = Pipeline([
    # Load data
    BIDSLoader(
        path="data/",
        subject="01",
        task="rest",
        bad_channels=["EKG", "EOG"]
    ),

    # Preprocessing
    HighPassFilter(freq=1.0),
    UpSample(factor=10),
    TriggerDetector(regex=r"\btrigger\b", auto_find_missing=True),

    # Correction
    AveragedArtifactSubtraction(
        window_size=30,
        correlation_threshold=0.975
    ),
    AdaptiveNoiseCancellation(filter_order=None),  # Auto

    # Evaluation
    SNRMetric(),
    RMSMetric(),

    # Export
    EDFExporter(path="output.edf")
])

# Execute pipeline
result = pipeline.run()

# Or run with multiprocessing for channel-wise operations
result = pipeline.run(parallel=True, n_jobs=4)
```

### Flexible API

```python
# Skip steps conditionally
pipeline = Pipeline([
    BIDSLoader(...),
    HighPassFilter(freq=1.0),
    ConditionalProcessor(
        condition=lambda ctx: ctx.metadata.get("needs_upsampling"),
        processor=UpSample(factor=10)
    ),
    # ...
])

# Use alternative algorithms
pipeline = Pipeline([
    # ...
    # Choose correction method based on data properties
    SwitchProcessor(
        selector=lambda ctx: "motion" if ctx.has_motion_data else "aas",
        cases={
            "aas": AveragedArtifactSubtraction(),
            "motion": MotionBasedCorrection()
        }
    ),
    # ...
])

# Compose processors
custom_preprocessing = SequenceProcessor([
    HighPassFilter(freq=0.5),
    Notch Filter(freqs=[50, 100]),
    UpSample(factor=10)
])

pipeline = Pipeline([
    BIDSLoader(...),
    custom_preprocessing,
    # ...
])
```

### Direct API (for scripts)

```python
from facet import FACET

# Simple fluent API
facet = (FACET()
    .load_bids("data/", subject="01", task="rest")
    .highpass(1.0)
    .upsample(10)
    .detect_triggers(r"\btrigger\b")
    .aas_correction(window_size=30)
    .anc()
    .downsample()
    .lowpass(70)
    .evaluate(metrics=["SNR", "RMS"])
    .export("output.edf"))

result = facet.run()
```

### Plugin System

```python
from facet.core import Processor, register_processor
from typing import Any
import numpy as np

@register_processor("custom_filter")
class MyCustomFilter(Processor):
    """Custom filtering processor."""

    name = "custom_filter"
    description = "My custom filtering algorithm"

    def __init__(self, cutoff: float = 50.0):
        self.cutoff = cutoff

    def validate(self, context: ProcessingContext) -> None:
        """Validate that prerequisites are met."""
        if not context.has_raw():
            raise ValueError("No raw data available")

    def process(self, context: ProcessingContext) -> ProcessingContext:
        """Apply processing."""
        raw = context.get_raw()
        # Apply custom filter
        filtered = self._apply_filter(raw, self.cutoff)
        return context.with_raw(filtered)

    def _apply_filter(self, raw, cutoff):
        # Implementation
        pass

# Use in pipeline
pipeline = Pipeline([
    BIDSLoader(...),
    MyCustomFilter(cutoff=60.0),
    # ...
])
```

### C Extension Integration

```python
from facet.extensions import CNativeProcessor

class FastANCProcessor(CNativeProcessor):
    """Fast ANC using C implementation."""

    library_name = "fastranc"
    function_name = "fastr_anc"

    def prepare_args(self, context):
        """Convert Python data to C-compatible format."""
        return {
            "reference": context.get_noise().ctypes,
            "data": context.get_raw().get_data(copy=False).ctypes,
            "order": self.filter_order,
            "mu": self.mu
        }

    def process_result(self, c_result, context):
        """Convert C result back to Python."""
        filtered = np.ctypeslib.as_array(c_result)
        return context.with_filtered_data(filtered)
```

### Multiprocessing Support

```python
# Automatic parallelization for channel-wise operations
pipeline = Pipeline([
    BIDSLoader(...),
    ParallelProcessor(
        processor=ChannelWiseCorrection(),
        n_jobs=4,
        backend="multiprocessing"  # or "threading", "dask"
    ),
    # ...
])

# Or use context manager
with pipeline.parallel(n_jobs=4):
    result = pipeline.run()
```

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] Implement `ProcessingContext` class
- [ ] Implement `Processor` base class and interface
- [ ] Implement `Pipeline` execution engine
- [ ] Implement `Registry` for plugin system
- [ ] Add validation framework
- [ ] Add logging infrastructure

### Phase 2: Migrate I/O (Week 2)
- [ ] Create loader processors (BIDS, EDF, etc.)
- [ ] Create exporter processors
- [ ] Ensure MNE compatibility
- [ ] Add format detection

### Phase 3: Migrate Preprocessing (Week 3)
- [ ] Migrate filtering operations
- [ ] Migrate resampling operations
- [ ] Migrate trigger detection
- [ ] Migrate trigger alignment
- [ ] Add missing trigger detection

### Phase 4: Migrate Correction Methods (Week 4-5)
- [ ] Migrate AAS algorithm
- [ ] Migrate ANC algorithm
- [ ] Migrate PCA algorithm
- [ ] Migrate motion-based correction
- [ ] Create deep learning processor wrapper

### Phase 5: Evaluation & Visualization (Week 5)
- [ ] Migrate evaluation metrics
- [ ] Create visualization processors
- [ ] Add result aggregation

### Phase 6: Parallel Processing (Week 6)
- [ ] Implement multiprocessing support
- [ ] Add channel-wise parallelization
- [ ] Optimize memory usage
- [ ] Add progress reporting

### Phase 7: C Extensions (Week 6-7)
- [ ] Refactor fastranc integration
- [ ] Create generic C extension framework
- [ ] Add automatic compilation detection
- [ ] Create fallback pure-Python implementations

### Phase 8: Polish & Documentation (Week 7-8)
- [ ] Create migration guide
- [ ] Update all examples
- [ ] Write comprehensive docs
- [ ] Add type hints everywhere
- [ ] Create preset configurations
- [ ] Add backwards compatibility layer

## Key Design Decisions

### 1. ProcessingContext vs MNE Raw

**Decision**: Use a `ProcessingContext` wrapper around MNE objects

**Rationale**:
- Provides metadata storage (triggers, parameters, history)
- Allows lazy evaluation
- Enables pipeline optimization
- Still exposes underlying MNE objects for advanced users

```python
class ProcessingContext:
    def __init__(self, raw: mne.io.Raw):
        self._raw = raw
        self._raw_original = raw.copy()
        self._metadata = {}
        self._history = []
        self._artifacts = {}

    def get_raw(self) -> mne.io.Raw:
        return self._raw

    def with_raw(self, raw: mne.io.Raw) -> 'ProcessingContext':
        """Create new context with updated raw (immutable)."""
        new_ctx = ProcessingContext(raw)
        new_ctx._raw_original = self._raw_original
        new_ctx._metadata = self._metadata.copy()
        new_ctx._history = self._history.copy()
        return new_ctx
```

### 2. Immutable vs Mutable Processing

**Decision**: Processors return new contexts (immutable by default)

**Rationale**:
- Safer for parallel processing
- Enables pipeline optimization
- Makes debugging easier (can inspect intermediate states)
- Optional in-place operations for memory efficiency

### 3. Plugin Discovery

**Decision**: Explicit registration over auto-discovery

**Rationale**:
- More predictable
- Faster startup
- Easier to debug
- Still easy to use with decorators

### 4. Multiprocessing Strategy

**Decision**: Use `multiprocessing` with spawn method, not fork

**Rationale**:
- Avoids issues with inherited state
- Works on all platforms (including macOS)
- Each worker gets clean Python environment
- Pass only serializable context data

```python
def _worker_process(processor_class, processor_config, context_data):
    """Worker function for multiprocessing."""
    # Reconstruct processor from config
    processor = processor_class(**processor_config)
    # Reconstruct context from serialized data
    context = ProcessingContext.from_dict(context_data)
    # Process
    result = processor.process(context)
    # Return serialized result
    return result.to_dict()
```

## Migration Path

### Backwards Compatibility

Keep old API working with deprecation warnings:

```python
# facet/legacy/facet.py
from facet import Pipeline
from facet.io import EDFLoader
# ... imports

class facet:
    """Legacy API - deprecated."""

    def __init__(self):
        warnings.warn(
            "The legacy facet API is deprecated. "
            "Use facet.Pipeline or facet.FACET instead.",
            DeprecationWarning
        )
        self._pipeline_steps = []

    def import_eeg(self, path, **kwargs):
        self._pipeline_steps.append(EDFLoader(path, **kwargs))
        return self

    def highpass(self, freq):
        self._pipeline_steps.append(HighPassFilter(freq))
        return self

    # ... other methods

    def run(self):
        pipeline = Pipeline(self._pipeline_steps)
        return pipeline.run()
```

### Example Migration

**Old Code**:
```python
f = facet()
f.import_eeg("data.edf")
f.highpass(1)
f.find_triggers(regex)
f.calc_matrix_aas()
f.remove_artifacts()
```

**New Code**:
```python
pipeline = Pipeline([
    EDFLoader("data.edf"),
    HighPassFilter(1),
    TriggerDetector(regex),
    AASCorrection()
])
result = pipeline.run()
```

## Success Criteria

1. **Modularity**: Each processor is <200 lines, single responsibility
2. **Testability**: 80%+ code coverage with unit tests
3. **Performance**: Multiprocessing achieves 3x+ speedup on 4 cores
4. **Usability**: New users can create pipelines in <5 minutes
5. **Extensibility**: Adding new processor takes <30 minutes
6. **Compatibility**: 100% of existing examples work with new API
7. **Documentation**: Every processor has examples and API docs

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing code | High | Maintain legacy API layer |
| Performance regression | Medium | Extensive benchmarking |
| Complexity overhead | Medium | Keep simple API alongside pipeline API |
| Multiprocessing bugs | Medium | Thorough testing, fallback to serial |
| MNE version incompatibility | Low | Test against multiple MNE versions |

## Next Steps

1. Review and approve this plan
2. Create feature branch: `feature/ai-refactoring`
3. Implement Phase 1 (core infrastructure)
4. Create proof-of-concept with one complete pipeline
5. Iterate based on feedback
6. Proceed with remaining phases
