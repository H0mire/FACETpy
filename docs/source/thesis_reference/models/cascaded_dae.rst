Cascaded DAE
============

A denoising autoencoder (DAE) compresses a signal and learns to reconstruct a target.
Here, two stages estimate the artifact. The first makes an initial estimate. The second
receives the remaining signal and estimates the artifact left in it. FACETpy adds the
two predictions before subtraction.

Family and origin
-----------------

**Architecture:** Fully connected denoising autoencoder.

**Origin:** FACETpy baseline.

This baseline grew from the earlier FACETpy autoencoder prototype. No dedicated external
paper is assigned to this implementation. Its lineage is recorded in the retained code
and Phase-0 evidence.

FACETpy adaptation
------------------

The current model processes one epoch of one EEG channel. The migrated Phase-0 model is
a separate implementation: both of its autoencoders receive the same flattened
multichannel input. It is therefore a parallel sum, despite the historical name. It must
not be described as the current residual cascade.

Implementation and input
------------------------

The main package is ``facet.models.masterthesis.cascaded_dae``. The recorded adapter contract is
**single epoch, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base and deployment variants have separate factories. Select their input
packing, output type and normalization together with the checkpoint. A dataset
may store seven epochs even when a single-epoch adapter uses only the centre.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

The seven-epoch configuration in thesis Figure 8 and Table 5 belongs to
``deployment_cascaded_context_dae`` (batch size 64). Select that experiment for
those references. The single-epoch ``deployment_cascaded_dae`` arm (batch size
128) is a separate result. Both remain available; see :doc:`context_dae`.

Variants and lineage
--------------------

* :ref:`masterthesis/cascaded_dae/deployment <diagram-masterthesis-cascaded-dae-deployment>`:
  Two fully connected stages predict an artifact from one channel and one epoch. Stage
  two receives the signal after the first estimate is subtracted. The deployment
  wrapper normalizes input, restores output units and can remove the predicted epoch
  mean. Its objective scores recovered clean EEG.

* :ref:`masterthesis/cascaded_dae <diagram-masterthesis-cascaded-dae>`:
  Two fully connected stages predict an artifact from one channel and one epoch. Stage
  two receives the signal after the first estimate is subtracted.

* :ref:`masterthesis/legacy_cascaded_dae <diagram-masterthesis-legacy-cascaded-dae>`:
  Migrated Phase-0 adapter: two fully connected autoencoders receive the same
  flattened multichannel epoch, and their predictions are added. The adapter changes
  segmentation and resampling; it does not reproduce the original FACETpy 0.1.0
  environment.

Source-paper record
-------------------

This is an internal FACETpy baseline or extension. No dedicated external
source paper is assigned. The implementation and linked thesis experiments
record its lineage; the model name alone is not a literature attribution.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-cascaded-dae` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
