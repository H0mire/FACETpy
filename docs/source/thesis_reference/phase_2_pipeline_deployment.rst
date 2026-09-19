Phase 2 — Pipeline deployment and failure diagnosis
===================================================

Phase 2 examines the models inside the correction pipeline. The reusable pipeline
records the trigger offset, crop, filtering, upsampling, alignment and cleanup
settings. Changing one of these steps changes the experiment.

Deployment variants use an explicit recovered-clean objective and reconstruction
wrapper. Base-model holdout scores and deployment scores must not share a label
that suggests the same weights or protocol.

Validity and missing results
----------------------------

IC-U-Net and ST-GNN showed discontinuous pipeline output and were flagged as
invalid. Their numbers remain available as evidence of this failure. D4PM has no
completed Phase-2 pipeline result. Training curves and configuration records do
not supply that missing result.

Some original pipeline tables recorded a model name without an artifact checksum.
Where an association comes from the frozen loader's selection rule, the catalog
labels it as inferred. Hashing the file now verifies its present identity; it does
not retrospectively prove which bytes produced an old table.

The training curves and pipeline export can belong to different runs. They retain
separate experiment records unless the original records establish a shared run.
See :doc:`../masterthesis_guide/catalog` for these associations.

Primary arms and selected variants
----------------------------------

Figure 30 ranks the primary Phase-2 arms. Its DHCT-GAN uses one epoch; the later
selected seven-epoch DHCT variant belongs to a separate experiment. Keep their
residuals and waveforms under those respective identities.

The original Figure-27 image is retained, but its source CSV was not found. A new
CPU replay of all thirteen available primary deployment models on the 166 saved
holdout windows is stored as ``phase2_holdout_replay``. It uses global clean-signal
SNR improvement. This replay is labelled separately and does not replace the
missing original CSV or prove its exact values.

For inputs, commands and output checks, see
:doc:`../masterthesis_guide/phase_2_execution`.
