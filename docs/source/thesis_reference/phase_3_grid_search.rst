Phase 3 — Hyperparameter search and independent-reference evaluation
====================================================================

The completed base grid covers 27 settings for each of Nested GAN, Demucs and
ViT-Spectrogram. DHCT-GAN contributes two incomplete trials; it is not a completed
fourth grid. Preserve each trial's settings and evaluation weights, including
poor results, so that the ranking can be checked.

.. csv-table:: Recorded Phase-2 and selected Phase-3 comparison
   :file: ../../../masterthesis_guide/results/table_phase3_before_after/metrics.csv
   :header-rows: 1

Later experiments use the locked Weg-A data and a revised objective. Keep their
results separate from the proof-fit grid. Changing both the data and objective
prevents an isolated causal claim about the spike-loss weight.

The labelled held-out spike examples and the pipeline's four 100-microvolt injections into all EEG channels, measured at Fp1, are different tests. They must retain separate denominators, metric
protocols and result records. Improved artifact suppression alone does not show
that a model preserves neural events.

The catalog retains the base grid, Weg-A grid, selected later retraining runs and
compact comparison evidence. Acquisition can select one experiment without
removing other trial weights from the repository's LFS set.

Comparison boundaries
---------------------

The Weg-A grid table reports reconstruction error on its selection split. The
proof-fit grid reports residual gradient artifact after pipeline correction.
Both are expressed in microvolts, but they measure different quantities. The
thesis's unnumbered Weg-A plot and following negative finding remain associated
with their own dataset and metric; their values do not form a matched before/after
pipeline comparison.

The original Figure 34 is retained in the evidence index. Its caption says
"full dataset", while the plotted title specifies Fp1 from 29.5 to 160 seconds.
Use the narrower scope shown in the figure. The original spectral estimator
settings were not recovered, so the image alone cannot establish exact numerical
reproduction of that spectrum.

For inputs, commands and output checks, see
:doc:`../masterthesis_guide/phase_3_execution`.
