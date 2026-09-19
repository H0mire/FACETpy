Read model results
==================

Start with the experiment ID. A model family can have several variants, datasets,
objectives and checkpoints. Its name alone does not identify a result.

Files and their sources
-----------------------

Each model's ``results/index.rst`` lists the experiments assigned to that exact
variant. The same list appears below the variant's diagrams on its model page.
The ``manifest.json`` records associations, artifact records and commands.
Paths in the manifest are relative to the repository root.

Each experiment folder contains:

* ``hyperparameters.yaml``: a view of recorded model, data and training settings,
  plus the source and reproduction-configuration path. Consult the full source
  configuration for checkpoint and export settings.
* ``evaluation.json``: run-owned evidence, explicit table-row selections and
  links to shared collections. Named comparison arms retain their own scope.
* ``training_curve.svg``: unsmoothed epoch losses, when an epoch history exists.
  An absent history stays absent; a final loss alone cannot supply a curve.

These are generated views of retained records, not new experiments. Source paths
and hashes allow each view to be traced back to its evidence. A shared table is
not assigned row by row unless an explicit association supports that selection.
Non-finite numbers appear as ``null`` in generated JSON; consult the unchanged
source to see the original value. Null does not mean zero.

Read status fields separately
-----------------------------

.. list-table:: Independent status fields
   :header-rows: 1
   :widths: 24 28 48

   * - Field
     - Values
     - Meaning
   * - Scientific outcome
     - ``valid``, ``invalid``, ``unavailable``
     - The catalog's assessment of the recorded result. Invalid findings remain
       evidence of failure; unavailable results supply no score.
   * - Replay verification
     - ``not_run``, ``verified``, ``blocked``
     - Whether the stated reproduction check has been performed. Read its scope
       and reason; a historical valid result can still have ``not_run`` here.
   * - Evaluation availability
     - ``recorded_evidence``, ``shared_collections_only``, ``unavailable``
     - Whether a run-owned record or only a shared collection is associated.
       Recorded evidence can include a manifest, not just a numeric score.
   * - Parameter provenance
     - ``original_resolved``, ``reproduction_config``, ``reconstructed``,
       ``partial_training_record``, ``unavailable``
     - How the settings were obtained. A reconstruction remains a reconstruction
       even if the resulting model runs.

An empty model result index means no experiment is assigned in the catalog. It
does not establish that the model was never trained or that no other files exist.

Training loss and evaluation metrics
------------------------------------

Training and validation curves show the objective used by that run. Check the
loss definition, target, normalization and weights before comparing values.
The validation minimum need not identify the exported state; see
:doc:`checkpoint_selection`.

.. list-table:: Read metrics within their protocol
   :header-rows: 1
   :widths: 30 70

   * - Measurement
     - Interpretation
   * - Phase-1 clean SNR improvement
     - The change in reference-based SNR after correction, in dB. Higher values
       mean less error relative to that clean target on the saved holdout.
   * - Phase-1 clean MSE and MAE
     - Error against the clean target. Lower values mean closer agreement.
       MSE uses squared input units; MAE uses input units.
   * - Phase-1 RMS recovery ratio
     - Mean corrected-to-clean standard-deviation ratio across window/channel
       pairs. The target is 1, not 0. It measures amplitude agreement.
   * - Pipeline ``ga_rest_uv``
     - Periodic artifact residual in microvolts within the recorded window.
       Lower is better for this measure, but does not establish neural preservation.
   * - Pipeline ``naht_ratio``
     - Seam diagnostic. The retained tool flags ratios at least 1.8 as suspicious
       and at least 2.5 as invalid seams. Inspect the waveform as well.
   * - Weg-A selection error
     - Reconstruction error on the selection split. Sharing microvolt units with
       a pipeline residual does not make the two measurements comparable.
   * - Spike preservation
     - Read the reference, event selection and denominator with each value.
       Labelled holdout cases and paired injected recordings are separate tests.

A compact comparison record
---------------------------

Report the experiment ID, artifact hash, dataset and split, protocol, metric,
units, window/channel scope and verification status together. Retain negative
findings and early-phase evidence used by the thesis. Compare results only where
these conditions match, and state any remaining differences.

For execution, use :doc:`legacy_execution`, :doc:`phase_1_execution`,
:doc:`phase_2_execution` or :doc:`phase_3_execution`.
