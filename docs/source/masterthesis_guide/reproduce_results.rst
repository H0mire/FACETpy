Reproduce recorded results
==========================

Choose a scope
--------------

* A smoke run checks imports, tensor shapes and correction plumbing with small
  fixtures. It establishes function, not thesis-result equivalence.
* One experiment uses its exact configuration, data version, saved split and
  evaluated checkpoint. Fetch only its artifacts.
* A full thesis replay also includes earlier phases, invalid results, grid trials,
  construction records, engineering measurements and figure inputs.

A valid scientific outcome does not imply that a rerun has been verified. The
catalog records outcome, artifact availability and verification separately.

Training
--------

Create a resolved configuration in an output directory. Original records under
``provenance`` remain unchanged. The resolver replaces dataset paths with explicit
local inputs and gives new training outputs their own directory.

.. code-block:: console

   uv run python -m masterthesis_guide.reproduce config \
     run8_demucs_lr0_001_ic64_sisdr0_s42 \
     --data-root /path/to/local/thesis-data \
     --output-dir /path/to/reproduction/demucs \
     --out /path/to/reproduction/demucs.yaml --device cpu
   uv run facet-train fit --config /path/to/reproduction/demucs.yaml

Retraining can produce different weights because of backend and numerical
variation. It is not a substitute for evaluating the recorded checkpoint.

Evaluation
----------

The Phase-1 example uses the saved holdout indices and retained metric functions.
Later phases require their own full-pipeline and spike-preservation protocols.
Do not apply a Phase-1 evaluator and label its output as a Phase-3 result.

.. literalinclude:: ../../../masterthesis_guide/examples/evaluate_model.py
   :language: python

For a matched FARM comparison, use the same recording, alignment, crop, cleanup
and output sampling frequency for both correction arms:

.. literalinclude:: ../../../masterthesis_guide/examples/compare_with_farm.py
   :language: python

Recorded inputs and source gaps
-------------------------------

The experiment index links executable configurations and original resolved
records. The catalog also names canonical evidence tables and artifact hashes.
An inferred artifact association or missing original configuration remains
explicit. A reconstructed configuration must be labelled as such; its existence
cannot fill a provenance gap by itself.

All source fields retain their original language and values. Maintained
explanations are English; immutable machine records are evidence.

Pipeline tables and injected spikes
-----------------------------------

Use the experiment IDs from the evidence index. The following command creates
matched primary model, FARM and uncorrected arms. It preserves the recorded
25–160-second window, channel order and microvolt output convention.

.. code-block:: console

   uv run python tools/pipeline_demo/run_arms.py \
     --experiment deployment_demucs --baseline farm --baseline uncorrected \
     --edf /path/to/NiazyFMRI.edf --out-dir /path/to/replay/arms --device cpu
   uv run python tools/pipeline_demo/measure_arms.py \
     --arm-dir /path/to/replay/arms --out /path/to/replay/residual_metrics.csv

For spike preservation, use ``inject_spikes.py`` to create the injected recording
and truth record. Run the same correction arms on the original and injected
recordings, then pass both directories and the truth file to
``measure_spike_preservation.py``. Its percentages relative to the nominal
injection and processed uncorrected arm are separate columns. Do not mix them.
Use ``--help`` on these entry points for the explicit inputs.

For the locked tensor holdout, ``compare_phase3_spike_aware.py`` reads the original
and retrained model artifacts and exact configuration. It reports the full
holdout separately from labelled spike cases. ``eval_run6_spike_preservation.py``
and ``paired_spike_comparison.py`` preserve the per-example and event-level
protocol when that comparison is required.

Grid search and dataset construction
------------------------------------

Resolve a recorded configuration before starting a new grid. Pass it with
``grid_search_run7.py --base-config``; use ``--dry-run`` to inspect generated
settings without training. The proof-fit grid also needs ``--edf`` for pipeline
scoring. The Weg-A grid uses its selection split and excludes the locked holdout.
The two scoring protocols are not interchangeable.

``derive_wega_loss_weights.py`` takes ``--config-dir`` containing resolved
``nested_gan.yaml``, ``vit_spectrogram.yaml``, ``demucs.yaml`` and ``dhct_gan.yaml``.
It writes a measurement with ``--out`` and can write separate adjusted copies
with ``--write-configs``. It does not edit the catalogued historical configurations.

Dataset construction starts with the proof-fit builder under
``examples/dataset_building`` or the retained Weg-A tools under
``tools/dataset_building``. Preserve each dataset's original construction settings,
metadata and stored split. Supply the VEPISET directory explicitly when selecting
real annotated spikes. The current tools do not supply or download those sources.

Figures and historical predictions
----------------------------------

The generated evidence index gives each retained figure's generator and command.
Diagram generators write SVG and PNG files under ``output/thesis_figures``;
PNG previews require ``rsvg-convert`` from librsvg. Table-based plots read the
canonical CSV files. Generated outputs stay outside versioned evidence until
reviewed as a distinct new result.

Phase-1 signal panels use fourteen original ``predicted_artifact.npy`` arrays.
These were tracked in the preserved source snapshot, so they are not necessarily
present in the ignored-file ZIP. The catalog records their Git source, SHA-256,
size and relative destination. Restore them to an external prediction root,
then pass that root as ``--prediction-root``. For one array:

.. code-block:: console

   mkdir -p /path/to/predictions/output/model_evaluations/demucs/holdout_v1
   git show 21d689da73eb93cdc3af8b48babed76b7271cc4e:output/model_evaluations/demucs/holdout_v1/predicted_artifact.npy > /path/to/predictions/output/model_evaluations/demucs/holdout_v1/predicted_artifact.npy

The plotting command checks these bytes against the catalog before drawing them.
Original embedded images are retained where an editable generator was not
recovered. The index states those limits and distinguishes a new replay from
an original measurement.

Optional GPU workers
--------------------

The retained fleet runs independent training jobs, one per GPU. It does not
implement distributed training of one model. Start from ``workers.example.yaml``
and keep personal worker settings in the ignored local file. The queue's default
state belongs to the current checkout; ``--state`` can select another explicit
location.

The sync helper excludes datasets and selected LFS artifacts. Place the exact
required inputs on a worker separately and use a resolved configuration with that
worker's paths. Run, sync, fetch and dispatch commands contact remote workers only
when explicitly invoked. No remote operation is needed for local validation,
normal tests or documentation builds.
