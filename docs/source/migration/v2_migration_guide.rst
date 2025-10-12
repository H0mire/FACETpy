Migration Guide to v2.0
=======================

This guide helps you migrate from FACETpy v1.x to v2.0.

Overview of Changes
-------------------

Version 2.0 is a complete rewrite with a new modular architecture. The key changes are:

* **New API:** Pipeline-based instead of monolithic framework classes
* **Modular Design:** Individual processors instead of coupled methods
* **Plugin System:** Decorator-based registration for extensibility
* **MNE Integration:** Direct MNE object access instead of wrappers
* **Type Hints:** Full type annotations throughout
* **Parallel Processing:** Built-in multiprocessing support

.. warning::
   The v1.x API is **not backward compatible** with v2.0. You will need to update your code.

Quick Migration Examples
------------------------

Basic Correction
~~~~~~~~~~~~~~~~

**Old (v1.x):**

.. code-block:: python

   from facet.facet import facet

   f = facet("data.edf")
   f.detect_triggers(r"\b1\b")
   f.upsample(10)
   f.correction.calc_matrix_aas()
   f.correction.remove_artifacts()
   f.export("corrected.edf")

**New (v2.0):**

.. code-block:: python

   from facet import create_standard_pipeline

   pipeline = create_standard_pipeline(
       input_path="data.edf",
       output_path="corrected.edf",
       trigger_regex=r"\b1\b"
   )
   result = pipeline.run()

Custom Workflow
~~~~~~~~~~~~~~~

**Old (v1.x):**

.. code-block:: python

   from facet.facet import facet

   f = facet("data.edf")
   f.detect_triggers(r"\b1\b")
   f.upsample(10)
   f.correction.align_triggers(ref_trigger=0)
   f.correction.calc_matrix_aas(window_size=30)
   f.correction.remove_artifacts()
   f.correction.apply_ANC()
   f.downsample()
   f.export("corrected.edf")

**New (v2.0):**

.. code-block:: python

   from facet.core import Pipeline
   from facet.io import EDFLoader, EDFExporter
   from facet.preprocessing import TriggerDetector, UpSample, TriggerAligner, DownSample
   from facet.correction import AASCorrection, ANCCorrection

   pipeline = Pipeline([
       EDFLoader(path="data.edf", preload=True),
       TriggerDetector(regex=r"\b1\b"),
       UpSample(factor=10),
       TriggerAligner(ref_trigger_index=0),
       AASCorrection(window_size=30),
       ANCCorrection(),
       DownSample(factor=10),
       EDFExporter(path="corrected.edf")
   ])
   result = pipeline.run()

Accessing Results
~~~~~~~~~~~~~~~~~

**Old (v1.x):**

.. code-block:: python

   f = facet("data.edf")
   # ... processing ...
   raw = f._eeg.mne_raw
   triggers = f._eeg.loaded_triggers
   noise = f._eeg.estimated_noise

**New (v2.0):**

.. code-block:: python

   result = pipeline.run()
   raw = result.context.get_raw()
   triggers = result.context.get_triggers()
   noise = result.context.get_estimated_noise()

API Mapping
-----------

Framework Methods → Processors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Old (v1.x)
     - New (v2.0)
   * - ``facet.load_edf(path)``
     - ``EDFLoader(path=path)``
   * - ``facet.detect_triggers(regex)``
     - ``TriggerDetector(regex=regex)``
   * - ``facet.upsample(factor)``
     - ``UpSample(factor=factor)``
   * - ``facet.downsample()``
     - ``DownSample(factor=factor)``
   * - ``facet.filter(l_freq, h_freq)``
     - ``BandPassFilter(l_freq=l_freq, h_freq=h_freq)``
   * - ``facet.correction.align_triggers()``
     - ``TriggerAligner()``
   * - ``facet.correction.calc_matrix_aas()``
     - ``AASCorrection()``
   * - ``facet.correction.apply_ANC()``
     - ``ANCCorrection()``
   * - ``facet.correction.apply_PCA()``
     - ``PCACorrection()``
   * - ``facet.export(path)``
     - ``EDFExporter(path=path)``

Data Access
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Old (v1.x)
     - New (v2.0)
   * - ``f._eeg.mne_raw``
     - ``context.get_raw()``
   * - ``f._eeg.mne_raw_orig``
     - ``context.get_raw_original()``
   * - ``f._eeg.loaded_triggers``
     - ``context.get_triggers()``
   * - ``f._eeg.estimated_noise``
     - ``context.get_estimated_noise()``
   * - ``f._eeg.artifact_length``
     - ``context.get_artifact_length()``
   * - ``f._eeg.sfreq``
     - ``context.get_raw().info['sfreq']``

Key Differences
---------------

Immutability
~~~~~~~~~~~~

**v1.x:** Modifies data in-place

.. code-block:: python

   f = facet("data.edf")
   f.upsample(10)  # Modifies f._eeg.mne_raw in-place
   f.downsample()  # Modifies again

**v2.0:** Returns new contexts (immutable by default)

.. code-block:: python

   context = loader.execute(ProcessingContext())
   context2 = upsampler.execute(context)  # context unchanged
   context3 = downsampler.execute(context2)  # context2 unchanged

MNE Integration
~~~~~~~~~~~~~~~

**v1.x:** Wrapped MNE objects with custom accessors

.. code-block:: python

   f._eeg.mne_raw  # Custom wrapper

**v2.0:** Direct MNE object access

.. code-block:: python

   raw = context.get_raw()  # Returns mne.io.Raw directly
   raw.plot()  # All MNE methods available
   raw.filter(l_freq=0.5, h_freq=None)

Error Handling
~~~~~~~~~~~~~~

**v1.x:** Silent failures or cryptic errors

.. code-block:: python

   f.correction.calc_matrix_aas()  # Might fail silently

**v2.0:** Explicit validation and detailed errors

.. code-block:: python

   result = pipeline.run()
   if not result.success:
       print(f"Failed at: {result.failed_processor}")
       print(f"Error: {result.error}")

Migration Checklist
-------------------

1. **Update Imports**

   .. code-block:: python

      # Old
      from facet.facet import facet

      # New
      from facet import create_standard_pipeline
      from facet.core import Pipeline
      from facet.io import EDFLoader, EDFExporter
      # etc.

2. **Convert to Pipeline**

   Replace method calls with processor list:

   .. code-block:: python

      pipeline = Pipeline([
          # Your processors here
      ])

3. **Update Data Access**

   Replace ``f._eeg.*`` with ``context.get_*()``

4. **Handle Results**

   Check ``result.success`` and access ``result.context``

5. **Update Tests**

   Processors are independently testable now

Common Pitfalls
---------------

1. **Forgetting to Check Results**

   .. code-block:: python

      # Don't do this
      result = pipeline.run()
      raw = result.context.get_raw()  # May fail if result.success is False

      # Do this
      result = pipeline.run()
      if result.success:
          raw = result.context.get_raw()

2. **Expecting In-Place Modification**

   .. code-block:: python

      # v1.x behavior (modified in-place)
      f.upsample(10)  # f is modified

      # v2.0 behavior (returns new context)
      context = upsampler.execute(context)  # Must reassign!

3. **Not Using Preload**

   .. code-block:: python

      # May cause issues
      EDFLoader(path="data.edf", preload=False)

      # Recommended
      EDFLoader(path="data.edf", preload=True)

Getting Help
------------

If you encounter migration issues:

1. Check the :doc:`../getting_started/quickstart` for examples
2. Browse the :doc:`../api/core` for detailed API docs
3. Open an issue on GitHub
4. Join our community discussions

Benefits of Migrating
---------------------

The new architecture provides:

* ✓ **Cleaner code** - More readable and maintainable
* ✓ **Better errors** - Clear validation and error messages
* ✓ **Faster processing** - Built-in parallelization
* ✓ **More flexible** - Compose any workflow
* ✓ **Easier testing** - Independent processors
* ✓ **Better docs** - Comprehensive documentation
* ✓ **Future-proof** - Extensible plugin system

Example: Complete Migration
----------------------------

**v1.x Complete Script:**

.. code-block:: python

   from facet.facet import facet

   f = facet("subject_01.edf")
   f.detect_triggers(r"\b1\b")
   f.upsample(10)
   f.correction.align_triggers(ref_trigger=0)
   f.correction.calc_matrix_aas(window_size=30)
   f.correction.remove_artifacts()
   f.correction.apply_ANC()
   f.correction.apply_PCA(n_components=0.95)
   f.downsample()
   f.filter(l_freq=0.5, h_freq=None)
   f.export("subject_01_corrected.edf")
   print("Done!")

**v2.0 Equivalent:**

.. code-block:: python

   from facet.core import Pipeline
   from facet.io import EDFLoader, EDFExporter
   from facet.preprocessing import (
       TriggerDetector, UpSample, TriggerAligner, DownSample
   )
   from facet.correction import AASCorrection, ANCCorrection, PCACorrection
   from facet.preprocessing import HighPassFilter

   pipeline = Pipeline([
       EDFLoader(path="subject_01.edf", preload=True),
       TriggerDetector(regex=r"\b1\b"),
       UpSample(factor=10),
       TriggerAligner(ref_trigger_index=0),
       AASCorrection(window_size=30),
       ANCCorrection(),
       PCACorrection(n_components=0.95),
       DownSample(factor=10),
       HighPassFilter(freq=0.5),
       EDFExporter(path="subject_01_corrected.edf", overwrite=True)
   ])

   result = pipeline.run()

   if result.success:
       print("Done!")
   else:
       print(f"Failed: {result.error}")
