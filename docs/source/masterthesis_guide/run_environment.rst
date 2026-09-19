Record the run environment
==========================

Keep the environment with each new run so another reader can explain a mismatch.
Use a new output directory. Preserve historical records unchanged.

Read historical records conservatively
--------------------------------------

A resolved configuration records requested settings, such as ``device: cuda``
and a seed. It does not by itself identify the GPU model, driver or complete
software environment. A summary may record elapsed training time without
covering data preparation, export or evaluation.

For example, the retained ``holdout_demucs`` configuration requests CUDA and seed
42. Its summary records 52 epochs, best epoch 49 and about 405.7 seconds of
training. These fields do not establish the original GPU model or a full runtime
benchmark. Inspect the
:download:`configuration <../../../masterthesis_guide/experiments/phase_1/holdout_demucs/provenance/facet_train_config.resolved.json>`
and :download:`summary <../../../masterthesis_guide/experiments/phase_1/holdout_demucs/provenance/summary.json>`
for their exact values and scope.

The current repository lockfile describes the current dependency set. It does
not reconstruct an absent historical environment. The original Phase-0 environment
remains unresolved; see :doc:`legacy_execution`.

Capture a new run
-----------------

Keep these items beside the resolved configuration, logs and outputs:

* Repository commit and any uncommitted patch; Python version, installed package
  versions and a copy or hash of the lockfile.
* Operating system, CPU, RAM, actual accelerator model and backend. For CUDA,
  include the driver and PyTorch CUDA version. Record CPU thread settings too.
* Exact command, input dataset version and hash, saved split and artifact hash.
* Seeds for splitting, shuffling, initialization and augmentation where known;
  deterministic settings and any source of randomness left uncontrolled.
* UTC start and end times; separate durations for training and evaluation.
  State whether loading, preprocessing, export and warm-up are included.
* Result protocol, output paths, comparison tolerance and observed differences.

A single training seed does not prove that all random operations were controlled.
Matching seeds also does not establish identical results across devices or library
versions. Report what was set and what was measured.

Use this small sidecar as a starting point. Fill fields from the actual run;
``null`` means not recorded. This is a suggested record, not a configuration
accepted by ``facet-train``.

.. code-block:: yaml

   experiment: holdout_demucs
   record_kind: new_replay
   command: null
   source_commit: null
   source_patch: null
   python: null
   package_versions_file: null
   lockfile_sha256: null
   hardware:
     os: null
     cpu: null
     ram_bytes: null
     accelerator: null
     backend: null
     driver: null
     thread_settings: null
   inputs:
     dataset_sha256: null
     split_sha256: null
     artifact_id: null
     artifact_sha256: null
   randomness:
     seeds: null
     deterministic_settings: null
     uncontrolled_sources: null
   timing:
     started_utc: null
     finished_utc: null
     elapsed_seconds: null
     scope: null
   comparison:
     reference_record: null
     protocol: unified_holdout
     tolerance: null
     observed_difference: null
     conclusion: not_checked

Existing output coverage
------------------------

Training writes a resolved configuration, summary and epoch history. Pipeline
arms record the input and artifact hashes, device and protocol in JSON sidecars;
their NPZ files include elapsed pipeline time. The measurement command adds a
metric-scope sidecar. The simple Phase-1 evaluator writes metrics only.

These outputs cover different parts of the record above. Add missing environment
and comparison details beside them. A successful command or a new sidecar does
not automatically change the historical catalog's verification state.
