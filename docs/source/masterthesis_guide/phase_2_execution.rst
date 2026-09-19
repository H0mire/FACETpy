Phase-2 execution
=================

Use this guide to compare a deployment model with FARM and the processed
uncorrected recording. Read
:doc:`../thesis_reference/phase_2_pipeline_deployment` for the scientific scope
and known failures.

Prepare matched correction arms
-------------------------------

Complete :doc:`quickstart`, obtain the original Niazy EDF recording and materialize
the selected deployment artifact. Check its identity in
:doc:`checkpoint_selection`. The example uses ``deployment_demucs``.

Run from the checkout root and choose a new output directory. The arm writer
refuses to overwrite an existing arm.

.. code-block:: console

   uv run python tools/pipeline_demo/run_arms.py \
     --experiment deployment_demucs --baseline farm --baseline uncorrected \
     --edf /path/to/NiazyFMRI.edf --out-dir /path/to/replay/phase2/arms \
     --device cpu
   uv run python tools/pipeline_demo/measure_arms.py \
     --arm-dir /path/to/replay/phase2/arms \
     --out /path/to/replay/phase2/residual_metrics.csv

All arms use the retained pipeline settings. The uncorrected arm still passes
through shared preprocessing and cleanup. Each NPZ stores EEG in microvolts,
channel order, sampling frequency, elapsed time and window-local triggers.
Its JSON sidecar records the input hash, artifact identity and device.

The saved window spans 25–160 seconds of the original recording. By default,
measurement uses 4.5–135 seconds within that window, or 29.5–160 seconds of the
original recording. The sample-step diagnostic uses Fp1 when present; the tool
records the actual channel. Add ``--save-fif`` to the arm command when a later
analysis needs the full corrected recording.

Read the outputs
----------------

The CSV contains periodic artifact residuals, seam diagnostics and ratios to
FARM. The adjacent ``residual_metrics.meta.json`` records the measurement scope.
Read these together using :doc:`reading_results`. A low periodic residual does
not establish that the waveform is continuous or that neural events survive.

Compare the same arm and scope in the catalogued pipeline tables. The primary
DHCT-GAN arm and its later selected context variant have separate identities.
IC-U-Net and ST-GNN retain their invalid-output findings. D4PM has no completed
Phase-2 pipeline result.

A separate tensor-holdout check
-------------------------------

The Phase-1 evaluator also accepts available Phase-2 artifacts. Such a command
measures the unified tensor holdout, not the pipeline protocol above. The retained
``phase2_holdout_replay`` is a separate CPU replay; it does not recover the missing
original Figure-27 CSV. Keep these records and labels separate.

Record new runs using :doc:`run_environment`. For paired spike-injection tests,
follow the protocol described in :doc:`phase_3_execution`.
