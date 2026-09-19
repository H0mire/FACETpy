Select and identify checkpoints
===============================

The experiment catalog defines the association between a result and its weights.
Use that association before relying on a filename, file date or model family.

Best, last and exported states
------------------------------

* A best checkpoint is selected by a monitored metric and its direction, such as
  minimum validation loss. Record the metric, epoch and selection rule.
* A last checkpoint contains the state saved at the last completed epoch. It can
  differ from the best state, including when early stopping ends training.
* An export contains the state passed to the exporter in an inference format.
  An export filename alone does not prove that the best state was loaded first.

The current training CLI exports its in-memory model after ``Trainer.fit()``;
that path does not automatically reload the best checkpoint. The retained grid
runner separately selects an epoch checkpoint by the validation-loss value in
its filename. Historical artifacts still require their own source records;
current code alone cannot prove which state produced an old export.

How guide commands select an artifact
-------------------------------------

``masterthesis_guide.reproduce.selected_artifact`` applies these rules:

1. Use ``inference_artifact`` when the experiment names one explicitly. Otherwise,
   consider only that experiment's associated artifacts.
2. For the base D4PM and DenoiseMamba loaders, restrict candidates to source
   checkpoints.
3. On non-CUDA devices, prefer a candidate marked ``cpu_export``. Then prefer an
   export over a checkpoint and the ``evaluated`` role over other roles. Equal
   priorities retain catalog order; this is not a search for the lowest loss.
4. Reject missing files and LFS pointers, then check the selected file's size and
   SHA-256 against the catalog before loading it.

The function does not download weights. After materializing the required LFS
objects, inspect a selection from the checkout root:

.. code-block:: python

   from masterthesis_guide.reproduce import load_catalog, selected_artifact

   catalog = load_catalog()
   artifact_id, path = selected_artifact("holdout_demucs", catalog, device="cpu")
   record = catalog["artifacts"][artifact_id]
   print(artifact_id, path)
   print(record["sha256"], record["role"], record.get("identity"))

What a hash proves
------------------

A matching hash establishes that the present bytes match the catalog. It cannot
prove that those bytes produced a historical table if the original table did not
record them. Keep ``identity_note``, ``selection_note`` and inferred associations
with any reported result.

For example, ``deployment_demucs`` has an export association inferred from the
frozen loader's selection rule. Its hash is known, but the old pipeline table did
not record an artifact checksum. Phase-0 diagnostic and delivery checkpoints
also remain separate; see :doc:`../thesis_reference/phase_0_legacy`.

For a new replay, retain the selected artifact ID and hash, device, epoch and
selection evidence where available. Mark an unknown epoch or export origin as
unknown. Do not assign the lowest point on a nearby training curve to an export
without evidence that they share that state. See :doc:`run_environment`.
