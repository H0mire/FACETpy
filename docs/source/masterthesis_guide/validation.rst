Guide validation
================

The thesis-guide instructions were checked on 19 September 2026 using the local
original data and retained model artifacts. The machine used an Apple M4 Pro,
48 GiB RAM, Python 3.13.11 and PyTorch 2.11.0 on CPU.

This record separates complete example runs from short functional checks. It does
not mark every historical experiment as numerically reproduced. The catalog's
historical verification fields remain unchanged.

Executed checks
---------------

.. list-table:: Coverage and observed outputs
   :header-rows: 1
   :widths: 27 73

   * - Area
     - Check
   * - Inputs and configuration
     - Verified all 7 dataset hashes and 211 artifact hashes. Resolved all 210
       training configurations, imported their factories and checked their data
       paths. Checked 198 inference selections.
   * - Quickstart and FARM comparison
     - Ran both examples on the original EDF with deployment Demucs. Their saved
       Demucs signals were identical. All four saved FIF outputs, including the
       adapted Phase-0 run, had 30 EEG channels, a status channel, 2,048 Hz and
       finite samples.
   * - Phase 0
     - Ran the legacy checkpoint through the current adapter and full pipeline.
       This checks adapted execution, not the original legacy environment.
   * - Phase 1
     - Evaluated Demucs on all 166 saved holdout windows. The SNR improvement was
       31.295431 dB, matching the recorded value. The largest absolute difference
       across 20 comparable metric fields was about 1.3e-12.
   * - Phase 2
     - Generated Demucs, FARM and uncorrected pipeline arms. Demucs produced
       3.819 microvolts of periodic residual and a seam ratio of 1.367, matching
       the stored table. Its separate tensor-holdout SNR improvement also matched
       the retained CPU replay: 28.379042 dB.
   * - Phase 3
     - Ran the named Demucs trial through the full pipeline. Its periodic residual
       was 1.890 microvolts and its seam ratio was 1.452, matching the recorded
       values at the CSV's precision.
   * - Locked holdout
     - Compared the original and retrained Demucs artifacts on all 4,110 locked
       examples. Dataset and model hashes matched the historical report. The
       retrained model's CPU RMSE differed from the recorded CUDA value by about
       0.0021 microvolts. Execution passed; results were not bit-identical.
   * - Training and search
     - Completed one training epoch on 60 real proof-fit channel examples and
       wrote a checkpoint, history and export. Generated 27 proof-fit and 108
       Weg-A settings in dry runs and exercised confirmation. One short grid
       trial completed training, export and full-pipeline scoring; the seam check
       correctly rejected its undertrained output.
   * - Spike preservation
     - Injected four spikes into the original EDF, ran matched injected and
       original arms, and measured preservation at Fp1. The output retained both
       percentage denominators.
   * - Figures and generated views
     - Ran the referenced figure generators, restored and hash-checked all 14
       original prediction arrays, and checked 15 generated PNGs. All 13 model
       variants in the single-window plot ran. Verified 44 model diagram pairs
       and result views with 213 experiments and 210 epoch curves.
   * - Regression checks
     - The full suite passed 1,028 tests. Its two data-dependent skips also passed
       when rerun against the external original data. Compiled 213 documentation
       Python snippets. The Sphinx build and local HTML link checks passed.

The loss-scaling tool ran for all four families with two sampled examples and
wrote adjusted configuration copies. This checks its operation; it does not
reproduce the original 256-example measurement. The GPU fleet's local submit,
status and cancel operations passed in an isolated queue. No job was dispatched.

Corrections made during validation
----------------------------------

* Spike injection and single-channel dataset derivation now accept external input
  paths when writing provenance. Regression tests cover both cases.
* Three original comparison-baseline exports are now retained in Git LFS. Their
  hashes match the historical reports. They remain distinct from the retrained
  checkpoints selected for inference.
* The Nested GAN and ViT spike-comparison records now name the single-channel
  dataset identified by their original comparison hashes.
* The build instructions request the optional Sphinx dependencies and explain the
  local macOS compiler selection. The grid guide explains why a successful
  process exit must be checked against each trial's recorded status.

Limits
------

The exact original Phase-0 environment remains unresolved. Full historical
retraining, complete screening and confirmation runs, and full dataset rebuilding
from source recordings and VEPISET annotations were not performed. Dataset-tool
coverage consists of entry-point checks and synthetic regression tests.

Remote GPU execution, network Git/LFS acquisition and CUDA/MPS numerical
agreement were not tested. Local LFS objects were used. The named examples above
define execution coverage; checking every configuration does not mean that every
catalogued model was trained or evaluated again.

The :download:`machine-readable validation record <../../../masterthesis_guide/provenance/guide-validation-2026-09-19.json>`
contains commands, input hashes, measurements and source hashes. Its path tokens
``${REPO}``, ``${DATA_ROOT}`` and ``${VALIDATION_OUTPUT}`` name the checkout, external
data root and separate test-output directory. New test outputs did not replace
historical evidence.

Prediction archival follow-up (20 September 2026)
-------------------------------------------------

The fourteen original Phase-1 prediction arrays are now retained under
``artifacts/predictions/phase_1/`` through Git LFS. Each array matches the size and
SHA-256 already recorded in the catalog. Each contains 166 holdout windows, 30
channels and 512 samples, with finite values throughout.

The Phase-1 signal figure was generated from a separate copy of the current
source files and predictions without a ``.git`` directory. The proof-fit dataset
remained an explicit external input. No historical commit or prediction folder
was used. Regression tests cover absent files, LFS pointers, changed prediction
bytes and incomplete catalog associations. Generated review figures remain
outside the versioned evidence.
