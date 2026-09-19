Phase-3 execution
=================

Phase 3 contains several protocols. Choose the experiment and dataset before
choosing a command. See :doc:`../thesis_reference/phase_3_grid_search` for the
recorded comparisons and their limits.

Replay one proof-fit trial
--------------------------

Complete :doc:`quickstart`. Obtain the original EDF and materialize the selected
trial's checkpoint. This example evaluates one recorded trial; it does not claim
that the trial is the best setting for every metric.

.. code-block:: console

   uv run python tools/pipeline_demo/run_arms.py \
     --experiment run8_demucs_lr0_001_ic64_sisdr0_s42 \
     --baseline farm --baseline uncorrected \
     --edf /path/to/NiazyFMRI.edf --out-dir /path/to/replay/phase3/arms \
     --device cpu
   uv run python tools/pipeline_demo/measure_arms.py \
     --arm-dir /path/to/replay/phase3/arms \
     --out /path/to/replay/phase3/residual_metrics.csv

Use the trial's explicit row association in :doc:`catalog` or its model's
``results/`` folder to find the recorded comparison. Keep its seed, capacity,
learning rate and objective with the score. See :doc:`phase_2_execution` for the
pipeline's windows and output files.

Prepare a new search
--------------------

Resolve a recorded configuration into a new output location first:

.. code-block:: console

   uv run python -m masterthesis_guide.reproduce config \
     run8_demucs_lr0_001_ic64_sisdr0_s42 \
     --data-root /path/to/local/thesis-data \
     --output-dir /path/to/replay/phase3/training \
     --out /path/to/replay/phase3/demucs.yaml --device cpu
   uv run python tools/training/grid_search_run7.py screen \
     --family demucs --dataset proof_fit \
     --base-config /path/to/replay/phase3/demucs.yaml \
     --edf /path/to/NiazyFMRI.edf --out-root /path/to/replay/phase3/grid \
     --device cpu --pipeline-device cpu --dry-run

The dry run writes planned settings without training or scoring them. Inspect
those settings before removing ``--dry-run``. The full search can take substantial
time; a limited trial is useful for checking execution but cannot stand for the
completed grid. Use the tool's ``--help`` for confirmation seeds and trial limits.
Record all attempted settings, including failures. Inspect the output JSON
``zeilen`` entries: the grid command can exit successfully even when every trial
failed. A zero exit code alone does not establish a completed search.

The retained proof-fit search contains complete grids for three families and
only two incomplete DHCT-GAN trials. Running the current grid generator does not
turn that historical evidence into a completed four-family comparison.

Weg-A selection and locked holdout
----------------------------------

Use a Weg-A experiment's own configuration and dataset for a Weg-A search.
``--dataset wega`` selects a different scoring path; it does not convert a
proof-fit configuration into a Weg-A configuration. Selection uses its selection
split. Keep the locked holdout out of model selection.

For the later checkpoint comparison, the retained evaluator takes an explicit
baseline TorchScript export and a retrained source checkpoint. For Demucs,
resolve ``spike_aware_demucs`` with the configuration command above, writing
``/path/to/replay/weg-a-resolved.yaml``. Materialize both LFS inputs below.
The baseline is recorded as ``comparison_baseline_artifact``; the retrained
checkpoint is ``inference_artifact``. Both match the hashes in the original
comparison report.

.. code-block:: console

   uv run python tools/evaluation/compare_phase3_spike_aware.py \
     --config /path/to/replay/weg-a-resolved.yaml \
     --dataset /path/to/locked-weg-a.npz \
     --baseline artifacts/exports/masterthesis/phase_3/demucs_deployment_edition/spike_aware_demucs/comparison_baseline.ts \
     --checkpoint artifacts/checkpoints/masterthesis/phase_3/demucs_deployment_edition/spike_aware_demucs/epoch0002_val_loss13.2668.pt \
     --output-dir /path/to/replay/phase3/locked-holdout --device cpu

Use the original ``weg_a_v10_locked_1ch`` bundle for this Demucs example.
For another family, use its paired artifact IDs, dataset and configuration from
the catalog. The script evaluates ``example_split == 2`` and reports the full
holdout and labelled spike cases separately. Both training data and objective changed between these
models; this comparison cannot isolate the effect of the spike-loss weight.

Pipeline spike preservation
---------------------------

The injected-recording test is a separate protocol:

1. Use ``tools/pipeline_demo/inject_spikes.py`` to create the injected EDF and its
   truth record from the original recording.
2. Run the same correction arms on the original and injected recordings, with
   identical artifacts and pipeline settings, in separate output directories.
3. Use ``tools/pipeline_demo/measure_spike_preservation.py`` with those directories
   and the truth record. Use ``--help`` for the required input paths.

The nominal injection and processed uncorrected signal define different
percentage denominators. The retained test injects into all EEG channels and
measures at Fp1. It is not the labelled tensor-holdout test above.

Use :doc:`reading_results` and :doc:`run_environment` when reporting either test.
