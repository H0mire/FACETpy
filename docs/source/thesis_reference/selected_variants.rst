Selected thesis variants by model family
========================================

Read each result together with its dataset, target, preprocessing, checkpoint and
metric protocol. The same architecture can have different results under different
conditions. Those conditions must remain visible.

Code is organized by family. Experiments are organized by phase. The base,
deployment and any evaluated V2 variants remain separate where their contracts
differ. Experimental paper-accurate editions state an implementation goal; their
names do not establish agreement with a source paper.

.. list-table:: Comparison scopes
   :header-rows: 1
   :widths: 12 44 44

   * - Phase
     - Question
     - Limit
   * - 0
     - Can a small model reproduce an AAS-derived artifact target?
     - Same-recording fitting is not independent correction validation.
   * - 1
     - How do the models compare on one fixed proof-fit holdout?
     - Tensor holdout quality does not establish full-pipeline quality.
   * - 2
     - Do deployed models behave correctly inside the correction pipeline?
     - Discontinuous output is invalid even if its residual looks small.
   * - 3
     - How do selected families respond to tuning and revised training data?
     - Joint changes to the dataset and objective do not isolate either effect.

For exact settings and availability, use :doc:`../masterthesis_guide/catalog`.
No model or result is excluded solely because it comes from an earlier phase.
