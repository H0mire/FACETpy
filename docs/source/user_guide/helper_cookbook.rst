Helper Cookbook
===============

This page collects concise recipes for helper APIs that are easy to miss in the
long-form guides.

Quick File Conversion (`facet.load` / `facet.export`)
-----------------------------------------------------

Use top-level convenience helpers when you only need loading/exporting and no
explicit pipeline object.

.. code-block:: python

   import facet

   ctx = facet.load("./examples/datasets/NiazyFMRI.edf", preload=True)
   ctx = ctx | facet.HighPassFilter(freq=1.0)
   facet.export(ctx, "./outputs/NiazyFMRI_cleaned.set", overwrite=True)

Manual Trigger Injection (`with_trigger_samples`)
-------------------------------------------------

If you already have trigger samples from an external system, inject them
without running ``TriggerDetector``.

.. code-block:: python

   from facet import load

   ctx = load("data.edf", preload=True)
   ctx = ctx.with_trigger_samples(
       [1000, 2000, 3000, 4000],
       tr_seconds=2.0,
       trigger_regex="manual",
   )

MNE Events Bridge (`with_mne_events`)
-------------------------------------

Convert existing MNE events to FACETpy trigger metadata, including automatic
``first_samp`` handling.

.. code-block:: python

   import mne
   from facet import ProcessingContext

   raw = mne.io.read_raw_fif("sub-01_task-rest_eeg.fif", preload=True)
   events, event_id = mne.events_from_annotations(raw)

   ctx = ProcessingContext(raw=raw).with_mne_events(
       events,
       event="R128",
       event_id=event_id,
       tr_seconds=2.0,
   )

Batch Processing (`Pipeline.map` + `BatchResult.summary_df`)
-------------------------------------------------------------

Use one pipeline for many files and collect per-file metrics in one table.

.. code-block:: python

   from facet import Pipeline, TriggerDetector, AASCorrection, SNRCalculator

   pipeline = Pipeline([
       TriggerDetector(regex=r"\b1\b"),
       AASCorrection(window_size=30),
       SNRCalculator(),
   ])

   batch = pipeline.map([
       "sub-01.edf",
       "sub-02.edf",
       "sub-03.edf",
   ], keep_raw=False)

   batch.print_summary()
   df = batch.summary_df  # pandas.DataFrame (or None if pandas is unavailable)

Result Convenience (`PipelineResult.metric`, `metrics_df`, `plot`, `release_raw`)
-----------------------------------------------------------------------------------

Access metrics and plots without digging through nested metadata dictionaries.

.. code-block:: python

   result = pipeline.run(initial_context=ctx)
   result.print_summary()

   snr = result.metric("snr")
   flat_metrics = result.metrics_df

   # Optional quick visual check
   result.plot(channel="Fp1", start=5.0, duration=10.0)

   # Free memory after extracting metrics
   result.release_raw()

Context Metric Shortcut (`ProcessingContext.get_metric`)
--------------------------------------------------------

For conditional branches, ``get_metric`` keeps logic readable.

.. code-block:: python

   from facet import ConditionalProcessor, PCACorrection

   def needs_extra_cleanup(ctx):
       return ctx.get_metric("snr", float("inf")) < 10

   maybe_pca = ConditionalProcessor(
       condition=needs_extra_cleanup,
       processor=PCACorrection(n_components=0.95),
   )

Trigger QA (`TriggerExplorer` / `InteractiveTriggerExplorer`)
--------------------------------------------------------------

Inspect candidate triggers before correction when marker quality is uncertain.

.. code-block:: python

   from facet import Pipeline, Loader, AASCorrection
   from facet.preprocessing import TriggerExplorer

   pipeline = Pipeline([
       Loader(path="data.edf", preload=True),
       TriggerExplorer(),
       AASCorrection(window_size=30),
   ])

   result = pipeline.run()
