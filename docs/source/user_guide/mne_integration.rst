MNE-Python Integration
======================

FACETpy is designed to work directly with MNE-Python objects.
This page shows how to add FACETpy to an existing MNE-based research workflow
without rewriting your project structure.

When to Add FACETpy
-------------------

A practical pattern in existing MNE pipelines is:

1. Load data with your current MNE workflow.
2. Run FACETpy for MRI/BCG artifact correction.
3. Continue with MNE for ICA, epoching, statistics, and visualization.

Use FACETpy as one processing stage, not as a replacement for your full MNE stack.

Run FACETpy on an Existing MNE Raw
----------------------------------

If your project already creates a ``mne.io.Raw`` object, wrap it in a
:class:`facet.core.ProcessingContext` and run a FACETpy pipeline.

.. code-block:: python

   import mne
   from facet import (
       ProcessingContext,
       Pipeline,
       TriggerDetector,
       UpSample,
       TriggerAligner,
       AASCorrection,
       DownSample,
   )

   raw = mne.io.read_raw_fif("sub-01_task-rest_eeg.fif", preload=True)

   # Use copy() to keep your original Raw unchanged for comparisons.
   context = ProcessingContext(raw=raw.copy())

   correction = Pipeline([
       TriggerDetector(regex=r"\\b1\\b"),
       UpSample(factor=10),
       TriggerAligner(ref_trigger_index=0),
       AASCorrection(window_size=30),
       DownSample(factor=10),
   ], name="MRI Artifact Correction")

   result = correction.run(initial_context=context)
   if not result.success:
       raise RuntimeError(result.error)

   raw_corrected = result.context.get_raw()

Reuse Events From an Existing MNE Project
-----------------------------------------

If your triggers are already available as MNE events, you can pass them to FACETpy
via a context helper and skip trigger detection.

.. code-block:: python

   import mne
   from facet import ProcessingContext, Pipeline, UpSample, AASCorrection, DownSample

   raw = mne.io.read_raw_fif("sub-01_task-rest_eeg.fif", preload=True)
   events, event_id = mne.events_from_annotations(raw)

   mri_code = event_id["R128"]
   context = ProcessingContext(raw=raw.copy()).with_mne_events(
       events,
       event=mri_code,       # alternatively: event="R128"
       event_id=event_id,
       tr_seconds=2.0,       # required for AAS-style correction
   )

   correction = Pipeline([
       UpSample(factor=10),
       AASCorrection(window_size=30),
       DownSample(factor=10),
   ])

   result = correction.run(initial_context=context)
   raw_corrected = result.context.get_raw()

``ProcessingContext.with_mne_events(...)`` automatically converts absolute MNE sample
indices to FACETpy-relative trigger indices (accounts for ``raw.first_samp``).

Defaults for Optional Helper Fields
-----------------------------------

For ``ProcessingContext.with_mne_events(...)``:

- ``event=None``: allowed only if all events have the same event code.
- ``event_id=None``: only needed when ``event`` is passed as a string label.
- ``artifact_length=None`` and ``tr_seconds=None``:
  ``artifact_length`` defaults to the median sample distance between
  consecutive selected triggers.
- ``tr_seconds=None``: if provided, it is converted to samples and used as ``artifact_length``.
- ``store_event_id=True``: stores ``event_id`` in ``context.metadata.custom["event_id"]``.

For ``ProcessingContext.with_trigger_samples(...)``:

- ``samples_are_absolute=False``: trigger samples are interpreted as context-relative by default.
- ``artifact_length=None`` and ``tr_seconds=None``:
  ``artifact_length`` defaults to the median sample distance between
  consecutive triggers.
- ``custom=None``: no additional metadata entries are written.

Continue With MNE After Correction
----------------------------------

After correction, continue in your standard MNE analysis pipeline.

.. code-block:: python

   # Continue with your normal MNE workflow
   raw_corrected = result.context.get_raw()

   raw_corrected.filter(l_freq=1.0, h_freq=40.0)
   epochs = mne.Epochs(
       raw_corrected,
       events,
       event_id=event_id,
       tmin=-0.2,
       tmax=0.8,
       preload=True,
   )

   evoked = epochs.average()
   evoked.plot()

BIDS-Centered Workflows
-----------------------

If your project is BIDS-based, you can keep everything in BIDS using
:class:`facet.io.BIDSLoader` and :class:`facet.io.BIDSExporter`.

.. code-block:: python

   from facet import Pipeline, BIDSLoader, TriggerDetector, AASCorrection, BIDSExporter

   pipeline = Pipeline([
       BIDSLoader(root="./bids", subject="01", session="01", task="rest"),
       TriggerDetector(regex=r"\\b1\\b"),
       AASCorrection(window_size=30),
       BIDSExporter(root="./bids_derivatives/facetpy", subject="01", session="01", task="rest"),
   ])

   result = pipeline.run()

Recommendations for Research Projects
-------------------------------------

- Keep FACETpy correction in one dedicated pipeline stage for reproducibility.
- Store FACETpy metrics from ``result.context.metadata.custom`` next to your MNE outputs.
- Use ``raw.copy()`` before correction when you need before/after comparisons.
- If you provide triggers manually, always set ``artifact_length`` consistently with your TR/slice timing.
- For teams, version-control your pipeline definitions and processor parameters together with analysis scripts.
