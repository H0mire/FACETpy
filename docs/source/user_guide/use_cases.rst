Use Cases & Supported Artifacts
================================

This page gives an overview of the artifact types that FACETpy can handle and the
research scenarios in which the toolbox has been designed to be used.

.. contents:: Contents
   :local:
   :depth: 2


Supported Artifacts
-------------------

FACETpy targets artifacts that arise from the recording environment or from physiological
processes.  The table below lists the artifact types, their origin, and the correction
methods available in FACETpy.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Artifact
     - Origin
     - Correction methods
   * - **Gradient artifact (GA)**
     - MRI scanner gradient switching during simultaneous EEG-fMRI recordings
     - AAS, PCA
   * - **Ballistocardiogram (BCG)**
     - Cardiac-induced electrode movements in the static MRI magnetic field
     - AAS (after BCG detection), PCA
   * - **Cardioballistic / pulse artifact**
     - Pulsatile skin and vessel movement under electrodes
     - BCG detector + AAS
   * - **Motion artifact**
     - Head movement, cable pull, or electrode shift
     - Preprocessing filters, PCA
   * - **Power-line noise (50 / 60 Hz)**
     - Electrical mains interference
     - Notch filter (via MNE preprocessing step)
   * - **Muscle (EMG) artifact**
     - Jaw or neck muscle activity contaminating high-frequency EEG
     - High-frequency filtering, PCA
   * - **Amplifier saturation / DC drift**
     - Slow baseline drift or clipping in long recordings
     - Baseline correction, high-pass filter

.. note::

   Not every artifact class requires a dedicated FACETpy processor.  For standard
   spectral filtering (notch, bandpass) FACETpy delegates to MNE-Python's built-in
   filter methods that can be called inside a custom :class:`~facet.core.Processor`.
   See :doc:`custom_processors` for details.


Use Cases
---------


Simultaneous EEG-fMRI Research
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **primary use case** for which FACETpy was created.  EEG recorded inside an MRI
scanner is contaminated by strong, quasi-periodic gradient artifacts (GA) caused by the
rapidly switching magnetic field gradients used for image acquisition, and by
ballistocardiogram (BCG) artifacts driven by cardiac activity.

FACETpy addresses both in a single pipeline:

1. Detect volume triggers from annotations with :class:`~facet.preprocessing.TriggerDetector`.
2. Optionally upsample the signal with :class:`~facet.preprocessing.UpSample` for
   sub-sample trigger alignment.
3. Remove the gradient artifact with :class:`~facet.correction.AASCorrection`.
4. Detect R-peaks and remove the BCG artifact with a second AAS pass.
5. Export the corrected data with :class:`~facet.io.EDFExporter` or
   :class:`~facet.io.BIDSExporter`.

.. code-block:: python

   from facet import (
       Pipeline, Loader, EDFExporter,
       TriggerDetector, UpSample, DownSample,
       AASCorrection, BCGDetector,
   )

   pipeline = Pipeline([
       Loader(path="data.edf", preload=True),
       TriggerDetector(regex=r"\b1\b"),
       UpSample(factor=10),
       AASCorrection(window_size=30),                 # remove gradient artifact
       BCGDetector(),                                 # detect cardiac triggers
       AASCorrection(window_size=20, use_bcg=True),  # remove BCG artifact
       DownSample(factor=10),
       EDFExporter(path="corrected.edf", overwrite=True),
   ], name="EEG-fMRI Correction")

   result = pipeline.run()
   result.print_summary()


Multi-Subject / Multi-Session Batch Studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you have a BIDS dataset with many subjects and sessions, you can define one pipeline
and apply it to all runs automatically using :meth:`~facet.core.Pipeline.map`:

.. code-block:: python

   from facet import Pipeline, BIDSLoader, BIDSExporter, AASCorrection, TriggerDetector

   template = Pipeline([
       TriggerDetector(regex=r"\b1\b"),
       AASCorrection(window_size=30),
       BIDSExporter(output_root="derivatives/facetpy"),
   ])

   # collect contexts from a BIDS dataset
   from facet import BIDSLoader
   loader = BIDSLoader(bids_root="my_study/", task="rest")
   contexts = loader.load_all()

   batch_result = template.map(contexts)
   print(batch_result.summary())   # per-run SNR, RMS, timing

Metrics (SNR, RMS ratio) are collected per run and returned in a structured
:class:`~facet.core.BatchResult` object that can be exported to a CSV file for
downstream analysis.


Algorithm Comparison & Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FACETpy's evaluation module makes it straightforward to run several correction strategies
on the same dataset and compare them quantitatively.  This is particularly useful for
methodology papers that require SNR / RMS tables.

.. code-block:: python

   from facet import Pipeline, Loader, AASCorrection, PCACorrection
   from facet.evaluation import SNRCalculator, RMSCalculator, MetricsReport

   data = Loader(path="data.edf", preload=True)

   for CorrectionClass, label in [
       (AASCorrection(window_size=30), "AAS"),
       (PCACorrection(n_components=0.95), "PCA"),
   ]:
       result = Pipeline([data, CorrectionClass, SNRCalculator(), RMSCalculator()]).run()
       print(label, result.metrics)

   MetricsReport().compare(results)


Synthetic EEG Testing & Algorithm Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When real scanner data is not available — or when you need fully controlled ground truth —
:class:`~facet.misc.EEGGenerator` lets you synthesise EEG signals with configurable
artifact profiles:

.. code-block:: python

   from facet.misc import EEGGenerator

   gen = EEGGenerator(
       duration=300,          # seconds
       sfreq=5000,
       n_channels=32,
       artifact_type="gradient",
       tr=1.5,                # MRI repetition time in seconds
   )
   raw = gen.generate()      # returns mne.io.Raw

Use the generated data together with the normal pipeline processors to benchmark or tune
new correction algorithms before running on real data.


Clinical Feasibility Studies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EEG-fMRI is increasingly used in clinical settings — for example, to localise epileptic
foci during sleep MRI or to monitor anaesthesia depth.  In these scenarios, it is
essential to *preserve* pathological waveforms (interictal spikes, slow waves) while
removing scanner artifacts.

FACETpy supports this by:

* Keeping the full correction pipeline reproducible and auditable via
  :attr:`~facet.core.ProcessingContext.history`.
* Allowing conditional correction steps with
  :class:`~facet.core.ConditionalProcessor` so that artefact windows are skipped for
  clinical events.
* Providing per-epoch SNR metrics to quantify the quality of each corrected segment.

See :doc:`pipelines` for examples of conditional processing.


Teaching & Reproducible Science
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A FACETpy :class:`~facet.core.Pipeline` serialises the complete analysis as a Python
object that can be inspected, versioned, and shared with collaborators.  Each pipeline
step is documented in the processing history stored inside
:class:`~facet.core.ProcessingContext`, making the provenance of every result
transparent.

To share a complete, runnable analysis:

1. Commit the pipeline definition file and the raw data (or a BIDS dataset) to a
   repository.
2. Collaborators reproduce the results by running ``python pipeline_definition.py`` after
   installing FACETpy from PyPI.

No proprietary toolboxes, no manual GUI clicks — full reproducibility from a single
script.
