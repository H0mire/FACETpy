Phase 1 — Unified holdout comparison
====================================

Phase 1 evaluates fourteen neural model variants and two AAS baselines using a
common recorded holdout. The proof-fit bundle contains 833 examples; the saved
20-percent split contains 166 indices. The index file, not a rounded description
of the percentage, defines the evaluation set.

Every model receives the same selected examples through its own recorded packing
and normalization. Some models consume only the centre epoch, others the complete
context, and the multichannel models require the recorded montage order. Compare
model-reference contracts before interpreting a shared dataset shape as a shared
input shape.

The best reported holdout SNR improvement is approximately 31.3 dB for Demucs.
This is a tensor holdout result. It does not establish that its later deployment
variant behaves correctly inside the complete correction pipeline.

The catalog retains all evaluated models, their exports or source checkpoints,
individual metrics and the aggregate ranking. CPU-specific exports are distinct
artifacts unless their bytes are identical. D4PM needs its reverse sampler;
DenoiseMamba's original CPU path loads a Python model from its state dictionary.

See :doc:`../masterthesis_guide/reproduce_results` and
:doc:`../masterthesis_guide/catalog` for commands and the saved split.

For inputs, commands and output checks, see
:doc:`../masterthesis_guide/phase_1_execution`.
