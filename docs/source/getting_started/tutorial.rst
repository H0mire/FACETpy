Tutorial
========

This tutorial walks you through a complete fMRI artifact correction workflow.

.. note::
   This tutorial assumes you have already installed FACETpy. If not, see :doc:`installation`.

Dataset
-------

For this tutorial, we'll use an example EEG dataset recorded during fMRI acquisition.

You can use your own data or download example data from: [link to example data]

Step 1: Load Your Data
-----------------------

First, let's load the EEG data:

.. code-block:: python

   from facet.io import Loader
   from facet.core import ProcessingContext

   # Load EEG file (EDF, GDF, etc.)
   loader = Loader(path="my_data.edf", preload=True)
   context = loader.execute(ProcessingContext())

   # Inspect the data
   raw = context.get_raw()
   print(f"Channels: {len(raw.ch_names)}")
   print(f"Sampling rate: {raw.info['sfreq']} Hz")
   print(f"Duration: {raw.times[-1]:.2f} seconds")

Step 2: Detect fMRI Triggers
-----------------------------

Next, we need to detect the fMRI volume acquisition triggers:

.. code-block:: python

   from facet.preprocessing import TriggerDetector

   # Detect triggers using regex pattern
   # Adjust the pattern to match your trigger markers
   detector = TriggerDetector(regex=r"\b1\b")
   context = detector.execute(context)

   # Check triggers
   triggers = context.get_triggers()
   print(f"Found {len(triggers)} triggers")
   print(f"Artifact length: {context.get_artifact_length()} samples")

.. tip::
   Common trigger patterns:

   * ``r"\b1\b"`` - Matches value "1"
   * ``r"TR"`` - Matches "TR" annotation
   * ``r"Volume"`` - Matches "Volume" annotation

Step 3: Upsample for Precision
-------------------------------

Upsampling improves alignment precision:

.. code-block:: python

   from facet.preprocessing import UpSample

   # Upsample by factor of 10
   upsampler = UpSample(factor=10)
   context = upsampler.execute(context)

   # Check new sampling rate
   raw = context.get_raw()
   print(f"New sampling rate: {raw.info['sfreq']} Hz")

Step 4: Align Triggers
-----------------------

Align triggers using cross-correlation:

.. code-block:: python

   from facet.preprocessing import TriggerAligner

   # Align to first trigger
   aligner = TriggerAligner(ref_trigger_index=0)
   context = aligner.execute(context)

   print("Triggers aligned")

Step 5: Apply AAS Correction
-----------------------------

The main correction algorithm:

.. code-block:: python

   from facet.correction import AASCorrection

   # Apply AAS
   aas = AASCorrection(
       window_size=30,              # Sliding window size
       correlation_threshold=0.975,  # Correlation threshold
       realign_after_averaging=True  # Realign to averaged artifacts
   )
   context = aas.execute(context)

   print("AAS correction applied")

Step 6: Optional Additional Corrections
----------------------------------------

For residual artifacts, apply ANC or PCA:

.. code-block:: python

   from facet.correction import ANCCorrection, PCACorrection

   # Adaptive Noise Cancellation
   anc = ANCCorrection(filter_order=5, hp_freq=1.0)
   context = anc.execute(context)

   # PCA (optional)
   pca = PCACorrection(n_components=0.95)
   context = pca.execute(context)

   print("Additional corrections applied")

Step 7: Downsample
------------------

Return to original sampling rate:

.. code-block:: python

   from facet.preprocessing import DownSample

   # Downsample back
   downsampler = DownSample(factor=10)
   context = downsampler.execute(context)

   print(f"Sampling rate: {context.get_raw().info['sfreq']} Hz")

Step 8: Apply Final Filter
---------------------------

Apply highpass filter to remove slow drifts:

.. code-block:: python

   from facet.preprocessing import HighPassFilter

   # Highpass filter
   hpf = HighPassFilter(freq=0.5)
   context = hpf.execute(context)

   print("Filtered")

Step 9: Evaluate Results
-------------------------

Calculate quality metrics:

.. code-block:: python

   from facet.evaluation import SNRCalculator, RMSCalculator, MetricsReport

   # Calculate metrics
   snr = SNRCalculator()
   context = snr.execute(context)

   rms = RMSCalculator()
   context = rms.execute(context)

   # Print report
   report = MetricsReport()
   context = report.execute(context)

Step 10: Export Corrected Data
-------------------------------

Finally, save the corrected data:

.. code-block:: python

   from facet.io import EDFExporter

   # Export
   exporter = EDFExporter(path="corrected.edf", overwrite=True)
   exporter.execute(context)

   print("Corrected data saved!")

Complete Pipeline
-----------------

Here's the complete workflow as a pipeline:

.. code-block:: python

   from facet.core import Pipeline
   from facet.io import Loader, EDFExporter
   from facet.preprocessing import (
       TriggerDetector, UpSample, TriggerAligner,
       DownSample, HighPassFilter
   )
   from facet.correction import AASCorrection, ANCCorrection
   from facet.evaluation import SNRCalculator, MetricsReport

   # Build pipeline
   pipeline = Pipeline([
       Loader(path="my_data.edf", preload=True),
       TriggerDetector(regex=r"\b1\b"),
       UpSample(factor=10),
       TriggerAligner(ref_trigger_index=0),
       AASCorrection(window_size=30, correlation_threshold=0.975),
       ANCCorrection(filter_order=5, hp_freq=1.0),
       DownSample(factor=10),
       HighPassFilter(freq=0.5),
       SNRCalculator(),
       MetricsReport(),
       EDFExporter(path="corrected.edf", overwrite=True)
   ], name="Complete fMRI Correction")

   # Run it
   result = pipeline.run()

   if result.success:
       print(f"✓ Completed in {result.execution_time:.2f}s")
   else:
       print(f"✗ Failed: {result.error}")

Parallel Processing
-------------------

Speed up processing with parallelization:

.. code-block:: python

   # Use all CPU cores
   result = pipeline.run(parallel=True, n_jobs=-1)

   # Or specify number of cores
   result = pipeline.run(parallel=True, n_jobs=4)

Visualization
-------------

Visualize the corrected data:

.. code-block:: python

   # Get corrected data
   corrected_raw = result.context.get_raw()

   # Plot using MNE
   corrected_raw.plot(duration=10.0, n_channels=30)

   # Compare with original
   original_raw = result.context.get_raw_original()
   original_raw.plot(duration=10.0, n_channels=30)

Next Steps
----------

* Explore :doc:`examples` for more use cases
* Learn about creating custom processors in the examples
* Check the :doc:`../api/core` reference for detailed API documentation
