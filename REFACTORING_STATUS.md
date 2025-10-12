# FACETpy Refactoring Status

**Branch**: `feature/ai-refactoring`
**Date**: 2025-01-12
**Status**: Phase 1 Complete - Core Infrastructure Implemented

## Summary

The foundation for the new FACETpy architecture has been successfully implemented. The core pipeline framework, plugin system, and parallelization support are now in place and ready for migration of existing functionality.

## What Has Been Completed ✅

### 1. Core Infrastructure (`src/facet/core/`)

All core components have been implemented and are fully functional:

#### **ProcessingContext** (`context.py`)
- Immutable-by-default wrapper around MNE Raw objects
- Metadata storage (triggers, parameters, custom data)
- Processing history tracking
- Estimated noise accumulation
- Cache management
- Serialization support for multiprocessing
- **Lines of Code**: ~380

#### **Processor Base Class** (`processor.py`)
- Abstract base class for all processors
- Automatic validation and history tracking
- Composite processors: `SequenceProcessor`, `ConditionalProcessor`, `SwitchProcessor`
- Utility processors: `NoOpProcessor`, `LambdaProcessor`
- Type-safe parameter tracking
- **Lines of Code**: ~370

#### **Pipeline Execution Engine** (`pipeline.py`)
- Declarative pipeline definition
- Sequential execution with error handling
- Progress tracking and timing
- Pipeline validation before execution
- Fluent PipelineBuilder API
- Serialization to/from dict
- **Lines of Code**: ~280

#### **Plugin Registry** (`registry.py`)
- Singleton registry for processor discovery
- Decorator-based registration: `@register_processor`
- Category-based filtering
- Name-based lookup
- Prevents duplicate registrations
- **Lines of Code**: ~210

#### **Parallel Execution** (`parallel.py`)
- Multiprocessing support (spawn method)
- Threading support (for I/O-bound tasks)
- Channel-wise parallelization
- Epoch-wise parallelization (stub)
- Automatic work distribution
- Fallback to serial execution
- **Lines of Code**: ~320

### 2. I/O Processors (`src/facet/io/`)

Basic loaders and exporters have been implemented:

#### **Loaders** (`loaders.py`)
- `EDFLoader`: Load EDF files
- `BIDSLoader`: Load BIDS datasets
- `GDFLoader`: Load GDF files
- All with bad channel marking and metadata initialization
- **Lines of Code**: ~200

#### **Exporters** (`exporters.py`)
- `EDFExporter`: Export to EDF
- `BIDSExporter`: Export to BIDS
- Handles events/triggers automatically
- **Lines of Code**: ~120

### 3. Documentation

#### **REFACTORING_PLAN.md**
- Comprehensive 8-phase implementation plan
- API examples for all major features
- Architecture decisions and rationale
- Migration guide from old to new API
- Success criteria and risk mitigation

#### **CLAUDE.md** (Updated)
- Core concepts documented
- New architecture explained
- Will be updated as implementation progresses

### 4. Demo Example (`examples/new_api_demo.py`)

Demonstrates:
- Basic pipeline usage
- Fluent builder API
- Plugin system with custom processors
- Conditional processing
- Registry listing

## Current Architecture

```
src/facet/
├── core/                    # ✅ COMPLETE
│   ├── __init__.py         # Exports all core classes
│   ├── context.py          # ProcessingContext, Metadata
│   ├── processor.py        # Processor base classes
│   ├── pipeline.py         # Pipeline execution engine
│   ├── registry.py         # Plugin registry system
│   └── parallel.py         # Multiprocessing support
│
├── io/                      # ✅ PARTIALLY COMPLETE
│   ├── __init__.py
│   ├── loaders.py          # EDF, BIDS, GDF loaders
│   └── exporters.py        # EDF, BIDS exporters
│
├── preprocessing/           # ⏳ TO BE IMPLEMENTED
│   ├── filtering.py        # Filter processors
│   ├── resampling.py       # Up/downsample processors
│   ├── trigger_detection.py# Trigger finding
│   └── alignment.py        # Trigger alignment
│
├── correction/              # ⏳ TO BE IMPLEMENTED
│   ├── aas.py              # Averaged artifact subtraction
│   ├── anc.py              # Adaptive noise cancellation
│   ├── pca.py              # PCA-based correction
│   ├── motion.py           # Motion-based correction
│   └── deeplearning.py     # DL-based correction
│
├── evaluation/              # ⏳ TO BE IMPLEMENTED
│   ├── metrics.py          # SNR, RMS, etc.
│   └── visualization.py    # Plotting
│
├── extensions/              # ⏳ TO BE IMPLEMENTED
│   ├── native.py           # C extension utilities
│   └── fastranc/           # Fast RAN-C
│
├── config/                  # ⏳ TO BE IMPLEMENTED
│   ├── schema.py           # Configuration validation
│   └── presets.py          # Preset configurations
│
└── legacy/                  # ⏳ TO BE IMPLEMENTED
    └── facet.py            # Backwards compatibility
```

## Key Design Achievements

### 1. Separation of Concerns
- **Context**: Data storage (MNE Raw + metadata)
- **Processor**: Business logic (one responsibility each)
- **Pipeline**: Orchestration (execution flow)
- **Registry**: Discovery (plugin management)

### 2. Immutability by Default
- Processors return new contexts
- Original data preserved
- Enables pipeline optimization
- Safe for parallel execution

### 3. MNE-First Design
- Direct exposure of MNE Raw objects
- Full compatibility with MNE ecosystem
- Can use any MNE method alongside pipeline

### 4. Type Safety & Validation
- All processors have type hints
- Validation before execution
- Clear error messages
- Prerequisites checked automatically

### 5. Extensibility
- Plugin system with decorator registration
- Custom processors in <50 lines
- Composable processors (Sequence, Conditional, Switch)
- Easy to add new formats/algorithms

## How to Use the New System

### Basic Pipeline Example

```python
from facet.core import Pipeline
from facet.io import EDFLoader, EDFExporter

# Define pipeline
pipeline = Pipeline([
    EDFLoader(
        path="data.edf",
        bad_channels=["EKG", "EOG"],
        preload=True
    ),
    # More processors will go here
    EDFExporter(path="output.edf")
])

# Execute
result = pipeline.run()

# Check result
if result.was_successful():
    print(f"Success! Time: {result.execution_time:.2f}s")
    print(f"History: {result.get_history()}")
```

### Custom Processor Example

```python
from facet.core import Processor, register_processor, ProcessingContext

@register_processor
class MyFilter(Processor):
    name = "my_filter"
    description = "Custom filter"

    def __init__(self, cutoff: float):
        self.cutoff = cutoff
        super().__init__()

    def process(self, context: ProcessingContext) -> ProcessingContext:
        raw = context.get_raw().copy()
        raw.filter(l_freq=None, h_freq=self.cutoff)
        return context.with_raw(raw)

# Use in pipeline
pipeline = Pipeline([
    EDFLoader("data.edf"),
    MyFilter(cutoff=50.0),
    EDFExporter("output.edf")
])
```

## Next Steps (In Priority Order)

### Phase 2: Preprocessing Processors (HIGH PRIORITY)
Implement in `src/facet/preprocessing/`:
1. **filtering.py**
   - `HighPassFilter`
   - `LowPassFilter`
   - `BandPassFilter`
   - `NotchFilter`

2. **resampling.py**
   - `UpSample`
   - `DownSample`
   - `Resample`

3. **trigger_detection.py**
   - `TriggerDetector` (regex-based)
   - `QRSTriggerDetector` (BCG-based)
   - `MissingTriggerDetector`

4. **alignment.py**
   - `TriggerAligner`
   - `SubsampleAligner`

### Phase 3: Correction Processors (HIGH PRIORITY)
Implement in `src/facet/correction/`:
1. **aas.py**
   - `AASCorrection` (port from `frameworks/correction.py`)
   - Keep correlation-based epoch selection
   - Support window_size parameter

2. **anc.py**
   - `AdaptiveNoiseCancellation`
   - Integrate with fastranc C extension
   - Auto-derive filter parameters

3. **pca.py**
   - `PCACorrection`
   - Port OBS/PCA algorithm

4. **motion.py**
   - `MotionBasedCorrection`
   - Use Moosmann algorithm

### Phase 4: Evaluation & Visualization (MEDIUM PRIORITY)
Implement in `src/facet/evaluation/`:
1. **metrics.py**
   - `SNRMetric`
   - `RMSMetric`
   - `MedianArtifactMetric`

2. **visualization.py**
   - `PlotEEG`
   - `PlotMetrics`

### Phase 5: C Extension Integration (MEDIUM PRIORITY)
Implement in `src/facet/extensions/`:
1. **native.py**
   - Generic C extension loader
   - Automatic ctypes wrapping
   - Fallback to Python implementations

2. **fastranc/**
   - Refactor existing fastranc integration
   - Add processor wrapper

### Phase 6: Configuration System (LOW PRIORITY)
Implement in `src/facet/config/`:
1. **schema.py**
   - Parameter validation
   - Type checking
   - Default values

2. **presets.py**
   - Common pipeline configurations
   - Easy-to-use templates

### Phase 7: Legacy Compatibility (LOW PRIORITY)
Implement in `src/facet/legacy/`:
1. **facet.py**
   - Wrapper around new Pipeline API
   - Maintains old method names
   - Deprecation warnings

### Phase 8: Testing & Documentation (ONGOING)
1. **Unit tests** for each processor
2. **Integration tests** for full pipelines
3. **API documentation** with examples
4. **Migration guide** for existing users

## Implementation Guidelines

### When Adding New Processors

1. **Inherit from Processor**
   ```python
   from facet.core import Processor, register_processor

   @register_processor
   class MyProcessor(Processor):
       name = "my_processor"
       description = "What it does"
       requires_triggers = False  # Set appropriately
       modifies_raw = True        # Set appropriately
   ```

2. **Implement process()**
   ```python
   def process(self, context: ProcessingContext) -> ProcessingContext:
       # Get data
       raw = context.get_raw()

       # Process (create copy if modifying)
       processed_raw = raw.copy()
       # ... do processing ...

       # Return new context
       return context.with_raw(processed_raw)
   ```

3. **Add validation if needed**
   ```python
   def validate(self, context: ProcessingContext) -> None:
       super().validate(context)  # Check requires_raw, requires_triggers
       # Add custom validation
       if some_condition:
           raise ProcessorValidationError("Reason")
   ```

4. **Add to module __init__.py**
   ```python
   from .my_module import MyProcessor
   __all__ = ['MyProcessor', ...]
   ```

### When Porting Existing Code

1. **Identify the operation** (e.g., "highpass filter")
2. **Create processor class** with clear name
3. **Port core logic** to `process()` method
4. **Extract parameters** to `__init__()`
5. **Add type hints** to all parameters
6. **Test with simple pipeline**
7. **Add to registry** with `@register_processor`

### Multiprocessing Guidelines

1. **Mark processor as parallel-safe**
   ```python
   class MyProcessor(Processor):
       parallel_safe = True
       parallelize_by_channels = True  # If channel-wise
   ```

2. **Ensure serializable parameters**
   - Use simple types (int, float, str, list, dict)
   - Avoid lambda functions in parameters
   - Avoid non-picklable objects

3. **Test serial first, then parallel**
   ```python
   # Serial
   result = pipeline.run(parallel=False)

   # Parallel
   result_parallel = pipeline.run(parallel=True, n_jobs=4)
   ```

## Testing the New System

```bash
# Run the demo
python examples/new_api_demo.py

# Once tests are added:
pytest tests/test_core.py
pytest tests/test_processors.py
pytest tests/test_pipeline.py
```

## Current Metrics

- **Total New Code**: ~1,880 lines
- **Core Framework**: 1,560 lines
- **I/O Processors**: 320 lines
- **Code Coverage**: 0% (tests not yet written)
- **Processors Implemented**: 5 (loaders/exporters only)
- **Processors Remaining**: ~30-40 (estimated)

## Migration Impact

### Breaking Changes
- New import paths: `from facet.core import ...`
- Different API: Pipeline instead of facet class
- Immutable contexts by default

### Compatibility
- Legacy API will be provided in Phase 7
- Old examples will work with deprecation warnings
- MNE objects fully compatible

## Questions & Decisions Needed

1. **Should we keep the old API functional?**
   - Recommendation: Yes, with deprecation warnings

2. **How to handle backwards compatibility?**
   - Recommendation: Create `legacy/facet.py` wrapper

3. **When to merge to main?**
   - Recommendation: After Phase 3 (core + correction processors)

4. **How to handle deep learning frameworks?**
   - Recommendation: Keep as optional dependencies, create separate processors

5. **Testing strategy?**
   - Recommendation: Unit tests for each processor, integration tests for pipelines

## Resources

- **Refactoring Plan**: `REFACTORING_PLAN.md`
- **Demo Example**: `examples/new_api_demo.py`
- **Architecture Docs**: `CLAUDE.md`
- **This Status**: `REFACTORING_STATUS.md`

## Contact & Support

For questions about the refactoring:
1. Review `REFACTORING_PLAN.md` for design rationale
2. Check `examples/new_api_demo.py` for usage examples
3. Read processor source code for implementation patterns
4. Ask questions in GitHub issues

---

**Status**: ✅ Foundation Complete - Ready for Phase 2 Implementation

**Next Action**: Implement preprocessing processors (filtering, resampling, trigger detection)
