Context DAE
===========

Context DAE uses neighbouring epochs to estimate the artifact in the centre epoch. Two
autoencoders work in sequence. After the first prediction, the model subtracts that
estimate from the centre slot of the context. The second stage then estimates the
remaining artifact.

Family and origin
-----------------

**Architecture:** Fully connected context autoencoder.

**Origin:** FACETpy extension of Cascaded DAE.

This is an internal extension of :doc:`cascaded_dae`. It adds temporal context to the
residual cascade. No separate source paper is claimed.

FACETpy adaptation
------------------

The recorded input contains seven epochs from one channel. Both stages return only the
centre-epoch artifact. Neighbouring epochs supply context; they are not additional
output channels. Context length and channel count describe different axes.

Implementation and input
------------------------

The main package is ``facet.models.masterthesis.cascaded_context_dae``. The recorded adapter contract is
**seven epochs, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base and deployment variants have separate factories. Select their input
packing, output type and normalization together with the checkpoint. A dataset
may store seven epochs even when a single-epoch adapter uses only the centre.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Variants and lineage
--------------------

* :ref:`masterthesis/cascaded_context_dae/deployment <diagram-masterthesis-cascaded-context-dae-deployment>`:
  Two fully connected stages read seven epochs from one channel and predict the centre
  artifact. The second stage reads context with the first centre estimate removed. The
  deployment wrapper normalizes input, restores output units and can remove the
  predicted epoch mean. Its objective scores recovered clean EEG.

* :ref:`masterthesis/cascaded_context_dae <diagram-masterthesis-cascaded-context-dae>`:
  Two fully connected stages read seven epochs from one channel and predict the centre
  artifact. The second stage reads context with the first centre estimate removed.

Source-paper record
-------------------

This is an internal FACETpy baseline or extension. No dedicated external
source paper is assigned. The implementation and linked thesis experiments
record its lineage; the model name alone is not a literature attribution.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-cascaded-context-dae` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
