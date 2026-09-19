Phase-1 execution
=================

Use this guide to evaluate recorded weights on the saved proof-fit holdout.
The scientific scope is described in
:doc:`../thesis_reference/phase_1_unified_holdout`.

Prepare the inputs
------------------

1. Complete :doc:`quickstart` and choose a Phase-1 experiment in :doc:`catalog`.
   The example below uses ``holdout_demucs``.
2. Place the original proof-fit bundle below your data root at its catalogued
   relative path. Keep the saved split at
   ``masterthesis_guide/datasets/proof_fit/splits/holdout_v1_indices.json``.
   Its 166 indices define the holdout; do not draw a new random split.
3. Materialize the experiment's LFS artifact for the chosen device. See
   :doc:`checkpoint_selection` for selection rules and identity checks.

Run the recorded weights
------------------------

Run from the checkout root. Replace the example paths with local paths and use a
new output location for each replay.

.. code-block:: console

   uv run python -m masterthesis_guide.examples.evaluate_model holdout_demucs \
     --data-root /path/to/local/thesis-data --device cpu \
     --out /path/to/replay/phase1/demucs.json

The evaluator loads the saved windows, applies the model's input packing and
subtracts the predicted artifact from the noisy centre epoch. It writes the
reference-based metrics to the requested JSON file. It does not train the model.
For Demucs, compare that file with
:download:`the recorded metrics <../../../masterthesis_guide/experiments/phase_1/holdout_demucs/metrics.json>`.
Check sample count, channels and sampling frequency before comparing scores.

Repeat with another Phase-1 experiment ID to evaluate another variant. D4PM
requires its source checkpoint and reverse sampler. DenoiseMamba rebuilds its
model from source weights. Their execution time and dependencies can differ
from those of a traced export.

Interpret the replay
--------------------

Use :doc:`reading_results` to read the metrics and report differences with an
explicit tolerance. Save the selected artifact ID, hash and environment using
:doc:`run_environment`. A completed command does not by itself establish agreement
with the recorded result or change the catalog's verification status.

For retraining, use the experiment's ``prepare_training`` and ``train`` commands
in :doc:`catalog`. Retraining creates new weights. The evaluator above continues
to select the catalogued historical artifact, so it will not evaluate a newly
trained checkpoint merely because that checkpoint exists on disk.
