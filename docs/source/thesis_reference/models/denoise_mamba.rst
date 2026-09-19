DenoiseMamba
============

A state-space layer carries a learned state through a sequence. DenoiseMamba combines
this temporal processing with local convolutions. The FACETpy variants use different
state-space blocks and must be identified by their full package names.

Family and origin
-----------------

**Architecture:** Convolutional and state-space model.

**Origin:** EEG artifact removal.

Chen et al. introduced DenoiseMamba for EEG artifact removal [Chen2025]_. The thesis
base implementation is a flat Mamba-1-style reconstruction. It does not implement the
paper's full U-shaped ConvSSD design.

FACETpy adaptation
------------------

The base model projects one channel into features, applies selective state-space blocks,
and predicts the artifact. The experimental edition uses an encoder-decoder with ConvSSD
blocks, which combine convolution and structured state-space duality (SSD). It replaces
a spatial scan with forward and reverse temporal scans for the single-channel task. Its
default target is clean EEG; artifact prediction is configurable.

Implementation and input
------------------------

The main package is ``facet.models.masterthesis.denoise_mamba``. The recorded adapter contract is
**single epoch, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base and deployment variants have separate factories. Select their input
packing, output type and normalization together with the checkpoint. A dataset
may store seven epochs even when a single-epoch adapter uses only the centre.

The Phase-1 CPU path rebuilds the model from its state dictionary because the
original traced state-space scan contains a CUDA device. Select the source
checkpoint for that protocol.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Variants and lineage
--------------------

* :ref:`experimental/paper_accurate/denoise_mamba <diagram-experimental-paper-accurate-denoise-mamba>`:
  This experimental variant uses a U-shaped ConvSSD encoder-decoder and Mamba-2-style
  state-space processing. It predicts clean EEG by default and can be configured to
  predict artifacts.

* :ref:`masterthesis/denoise_mamba/deployment <diagram-masterthesis-denoise-mamba-deployment>`:
  A flat stack of local convolution and Mamba-1-style selective state-space blocks
  predicts a single-channel artifact. This is the thesis reconstruction, not the full
  published ConvSSD architecture. The deployment wrapper normalizes input, restores
  output units and can remove the predicted epoch mean. Its objective scores recovered
  clean EEG.

* :ref:`masterthesis/denoise_mamba <diagram-masterthesis-denoise-mamba>`:
  A flat stack of local convolution and Mamba-1-style selective state-space blocks
  predicts a single-channel artifact. This is the thesis reconstruction, not the full
  published ConvSSD architecture.

Source-paper record
-------------------

Sources: [Chen2025]_. See :doc:`../model_references` for full references.

A source citation explains the design lineage. It does not establish that a
FACETpy checkpoint reproduces the source paper's training or results.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-denoise-mamba` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
