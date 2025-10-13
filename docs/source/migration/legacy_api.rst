Legacy API Reference
====================

This document describes the legacy (v1.x) API for users who need to maintain
compatibility with older code or understand the differences between versions.

Overview
--------

FACETpy v1.x used a functional API with standalone functions. Version 2.0
introduces an object-oriented, pipeline-based architecture that is more
modular and extensible.

**We recommend migrating to the v2.0 API.** See :doc:`v2_migration_guide` for
a complete migration guide.

Legacy vs. Modern API
---------------------

Quick Comparison
~~~~~~~~~~~~~~~~

**Legacy (v1.x):**

.. code-block:: python

   from facet import correct_fmri_artifact

   corrected = correct_fmri_artifact(
       input_file="data.edf",
       output_file="corrected.edf",
       trigger_regex=r"\b1\b",
       window_size=30
   )

**Modern (v2.0):**

.. code-block:: python

   from facet.core import Pipeline
   from facet.io import EDFLoader, EDFExporter
   from facet.preprocessing import TriggerDetector
   from facet.correction import AASCorrection

   pipeline = Pipeline([
       EDFLoader(path="data.edf", preload=True),
       TriggerDetector(regex=r"\b1\b"),
       AASCorrection(window_size=30),
       EDFExporter(path="corrected.edf", overwrite=True)
   ])

   result = pipeline.run()

Legacy Functions
----------------

.. note::
   These functions are deprecated and will be removed in v3.0.
   Please migrate to the new API.

correct_fmri_artifact()
~~~~~~~~~~~~~~~~~~~~~~~

Main correction function (v1.x):

.. code-block:: python

   def correct_fmri_artifact(
       input_file: str,
       output_file: str,
       trigger_regex: str = r"\b1\b",
       window_size: int = 30,
       use_anc: bool = True,
       use_pca: bool = False,
       upsample_factor: int = 10,
       correlation_threshold: float = 0.975,
       **kwargs
   ) -> dict:
       """
       Correct fMRI artifacts in EEG data.

       Parameters
       ----------
       input_file : str
           Path to input EDF file
       output_file : str
           Path to output EDF file
       trigger_regex : str
           Regex pattern for trigger detection
       window_size : int
           AAS window size
       use_anc : bool
           Apply ANC correction
       use_pca : bool
           Apply PCA correction
       upsample_factor : int
           Upsampling factor
       correlation_threshold : float
           AAS correlation threshold

       Returns
       -------
       dict
           Results dictionary with metrics
       """

**Modern Equivalent:**

.. code-block:: python

   from facet import create_standard_pipeline

   pipeline = create_standard_pipeline(
       input_path=input_file,
       output_path=output_file,
       trigger_regex=trigger_regex,
       upsample_factor=upsample_factor,
       use_anc=use_anc,
       use_pca=use_pca
   )

   result = pipeline.run()

detect_triggers()
~~~~~~~~~~~~~~~~~

Trigger detection function (v1.x):

.. code-block:: python

   def detect_triggers(
       raw,
       regex: str = r"\b1\b",
       stim_channel: str = "auto"
   ) -> tuple:
       """
       Detect triggers in MNE Raw object.

       Parameters
       ----------
       raw : mne.io.Raw
           MNE Raw object
       regex : str
           Trigger pattern
       stim_channel : str
           Stimulus channel name

       Returns
       -------
       tuple
           (trigger_positions, artifact_length)
       """

**Modern Equivalent:**

.. code-block:: python

   from facet.preprocessing import TriggerDetector
   from facet.core import ProcessingContext

   detector = TriggerDetector(regex=regex, stim_channel=stim_channel)
   context = ProcessingContext(raw=raw)
   result = detector.execute(context)

   trigger_positions = result.get_triggers()
   artifact_length = result.metadata.artifact_length

apply_aas()
~~~~~~~~~~~

AAS correction function (v1.x):

.. code-block:: python

   def apply_aas(
       raw,
       triggers,
       artifact_length: int,
       window_size: int = 30,
       correlation_threshold: float = 0.975
   ):
       """
       Apply AAS correction.

       Parameters
       ----------
       raw : mne.io.Raw
           Input data
       triggers : array
           Trigger positions
       artifact_length : int
           Artifact length in samples
       window_size : int
           Window size
       correlation_threshold : float
           Correlation threshold

       Returns
       -------
       mne.io.Raw
           Corrected data
       """

**Modern Equivalent:**

.. code-block:: python

   from facet.correction import AASCorrection
   from facet.core import ProcessingContext, ProcessingMetadata

   metadata = ProcessingMetadata()
   metadata.triggers = triggers
   metadata.artifact_length = artifact_length

   context = ProcessingContext(raw=raw, metadata=metadata)
   aas = AASCorrection(
       window_size=window_size,
       correlation_threshold=correlation_threshold
   )
   result = aas.execute(context)

   corrected_raw = result.get_raw()

calculate_metrics()
~~~~~~~~~~~~~~~~~~~

Metrics calculation function (v1.x):

.. code-block:: python

   def calculate_metrics(
       corrected_raw,
       original_raw,
       triggers,
       artifact_length: int
   ) -> dict:
       """
       Calculate quality metrics.

       Parameters
       ----------
       corrected_raw : mne.io.Raw
           Corrected data
       original_raw : mne.io.Raw
           Original data
       triggers : array
           Trigger positions
       artifact_length : int
           Artifact length

       Returns
       -------
       dict
           Metrics dictionary
       """

**Modern Equivalent:**

.. code-block:: python

   from facet.evaluation import SNRCalculator, RMSCalculator
   from facet.core import ProcessingContext, ProcessingMetadata

   metadata = ProcessingMetadata()
   metadata.triggers = triggers
   metadata.artifact_length = artifact_length

   context = ProcessingContext(
       raw=corrected_raw,
       raw_original=original_raw,
       metadata=metadata
   )

   # Calculate metrics
   snr_calc = SNRCalculator()
   context = snr_calc.execute(context)

   rms_calc = RMSCalculator()
   context = rms_calc.execute(context)

   metrics = context.metadata.custom['metrics']

Migration Strategy
------------------

Wrapper Functions
~~~~~~~~~~~~~~~~~

Create wrapper functions for gradual migration:

.. code-block:: python

   def legacy_correct_fmri_artifact(input_file, output_file, **kwargs):
       """Legacy-compatible wrapper."""
       from facet import create_standard_pipeline

       # Map legacy parameters
       pipeline = create_standard_pipeline(
           input_path=input_file,
           output_path=output_file,
           trigger_regex=kwargs.get('trigger_regex', r"\b1\b"),
           upsample_factor=kwargs.get('upsample_factor', 10),
           use_anc=kwargs.get('use_anc', True),
           use_pca=kwargs.get('use_pca', False)
       )

       result = pipeline.run()

       # Return legacy-style result
       return {
           'success': result.success,
           'metrics': result.context.metadata.custom.get('metrics', {}),
           'time': result.execution_time
       }

Compatibility Layer
~~~~~~~~~~~~~~~~~~~

Create a compatibility module:

.. code-block:: python

   # facet/legacy.py
   """Legacy API compatibility layer."""

   import warnings

   def correct_fmri_artifact(*args, **kwargs):
       """Legacy function (deprecated)."""
       warnings.warn(
           "correct_fmri_artifact is deprecated. "
           "Use Pipeline API instead.",
           DeprecationWarning,
           stacklevel=2
       )
       return legacy_correct_fmri_artifact(*args, **kwargs)

Configuration Files
-------------------

Legacy Configuration
~~~~~~~~~~~~~~~~~~~~

v1.x used JSON configuration files:

.. code-block:: json

   {
       "input_file": "data.edf",
       "output_file": "corrected.edf",
       "trigger_regex": "\\b1\\b",
       "window_size": 30,
       "use_anc": true,
       "use_pca": false
   }

Modern Configuration
~~~~~~~~~~~~~~~~~~~~

v2.0 uses Python configuration:

.. code-block:: python

   # config.py
   from facet.core import Pipeline
   from facet.io import EDFLoader, EDFExporter
   from facet.preprocessing import TriggerDetector
   from facet.correction import AASCorrection, ANCCorrection

   def create_pipeline(config):
       """Create pipeline from configuration."""
       processors = [
           EDFLoader(path=config['input_file'], preload=True),
           TriggerDetector(regex=config['trigger_regex']),
           AASCorrection(window_size=config['window_size'])
       ]

       if config.get('use_anc', False):
           processors.append(ANCCorrection())

       processors.append(
           EDFExporter(path=config['output_file'], overwrite=True)
       )

       return Pipeline(processors)

   # Use it
   config = {
       'input_file': 'data.edf',
       'output_file': 'corrected.edf',
       'trigger_regex': r'\b1\b',
       'window_size': 30,
       'use_anc': True
   }

   pipeline = create_pipeline(config)
   result = pipeline.run()

Breaking Changes
----------------

API Changes
~~~~~~~~~~~

1. **Functions → Classes**
   - Functions replaced by Processor classes
   - Explicit pipeline construction

2. **Return Values**
   - Dictionary results → PipelineResult objects
   - Direct raw access → context.get_raw()

3. **Configuration**
   - JSON config → Python objects
   - Global settings → processor parameters

4. **Error Handling**
   - Mixed exceptions → specific exception types
   - Silent failures → explicit validation

5. **Data Flow**
   - Mutable operations → immutable contexts
   - In-place modification → new objects

Removed Features
~~~~~~~~~~~~~~~~

The following v1.x features have been removed:

- Global configuration file
- Automatic parameter tuning
- Built-in plotting functions (use MNE instead)
- Command-line interface (use Python API)

Renamed Modules
~~~~~~~~~~~~~~~

Module name changes:

- ``facet.correction`` → ``facet.correction`` (same)
- ``facet.utils`` → ``facet.core`` (restructured)
- ``facet.io.edf`` → ``facet.io`` (simplified)

Getting Help
------------

If you encounter migration issues:

1. Check the :doc:`v2_migration_guide`
2. Review :doc:`../getting_started/examples`
3. Open an issue on GitHub
4. Join our discussion forum

Deprecation Timeline
--------------------

- **v2.0** (2025-01) - Legacy API deprecated
- **v2.5** (2025-06) - Legacy warnings become errors
- **v3.0** (2026-01) - Legacy API removed

We recommend migrating as soon as possible to take advantage of new features
and improvements in v2.0.

See Also
--------

- :doc:`v2_migration_guide` - Step-by-step migration guide
- :doc:`../user_guide/architecture` - New architecture overview
- :doc:`../getting_started/quickstart` - Quick start guide
