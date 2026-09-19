SepFormer
=========

SepFormer encodes a waveform and divides the learned features into overlapping chunks.
Attention first relates samples within each chunk, then relates chunks to one another. A
mask and decoder turn these features back into a waveform.

Family and origin
-----------------

**Architecture:** Dual-path attention source separator.

**Origin:** Speech separation.

Subakan et al. introduced SepFormer for speech separation [Subakan2021]_. FACETpy uses a
compact adaptation to estimate an EEG artifact.

FACETpy adaptation
------------------

The model receives concatenated epochs from one channel and returns the centre artifact.
Its attention chunks lie in learned feature space; they are not the seven input epochs.
Model capacity and loss are selected for the EEG experiment. A scale-invariant loss
alone does not constrain the absolute artifact amplitude needed for subtraction.

Implementation and input
------------------------

The main package is ``facet.models.masterthesis.sepformer``. The recorded adapter contract is
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

* :ref:`experimental/paper_accurate/sepformer <diagram-experimental-paper-accurate-sepformer>`:
  This experimental variant revises the mask network and residual connections around
  the attention stacks. It keeps a compact, single-target EEG configuration rather
  than the original speech experiment.

* :ref:`masterthesis/sepformer/deployment <diagram-masterthesis-sepformer-deployment>`:
  Dual-path attention connects samples within and between learned feature chunks. A
  mask and decoder recover the centre-epoch artifact from one channel's context. The
  deployment wrapper normalizes input, restores output units and can remove the
  predicted epoch mean. Its objective scores recovered clean EEG.

* :ref:`masterthesis/sepformer <diagram-masterthesis-sepformer>`:
  Dual-path attention connects samples within and between learned feature chunks. A
  mask and decoder recover the centre-epoch artifact from one channel's context.

Source-paper record
-------------------

Sources: [Subakan2021]_. See :doc:`../model_references` for full references.

A source citation explains the design lineage. It does not establish that a
FACETpy checkpoint reproduces the source paper's training or results.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-sepformer` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
