ViT-Spectrogram
===============

This model converts EEG into a time-frequency representation with the short-time Fourier
transform (STFT). It splits the spectrogram into patches and uses attention to predict
the centre region from context. An inverse STFT reconstructs the waveform.

Family and origin
-----------------

**Architecture:** Spectrogram Transformer with masked reconstruction.

**Origin:** Image recognition and masked image reconstruction.

The patch encoder draws on Vision Transformer (ViT) [Dosovitskiy2021]_. Masked
autoencoding (MAE) supplies the reconstruction idea [He2022]_. The EEG spectrogram,
centre-epoch mask and correction target are FACETpy adaptations.

FACETpy adaptation
------------------

The base model predicts clean magnitude and reuses the noisy phase. Its adapter derives
artifact by subtracting the clean prediction from the noisy centre epoch. The deployment
variant instead learns a complex mask, so it can change both magnitude and phase. The
experimental MAE edition encodes only visible patches and uses a separate decoder; it
retains magnitude-based waveform reconstruction.

Implementation and input
------------------------

The main package is ``facet.models.masterthesis.vit_spectrogram``. The recorded adapter contract is
**seven epochs, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base and deployment variants have separate factories. Select their input
packing, output type and normalization together with the checkpoint. A dataset
may store seven epochs even when a single-epoch adapter uses only the centre.

The Phase-1 model predicts clean EEG. Its adapter subtracts this prediction
from the demeaned noisy centre epoch to obtain the artifact. The deployment
variant returns an artifact estimate through its own reconstruction wrapper.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Variants and lineage
--------------------

* :ref:`experimental/paper_accurate/vit_spectrogram <diagram-experimental-paper-accurate-vit-spectrogram>`:
  This experimental variant encodes visible patches and reconstructs masked patches
  with a separate MAE decoder. It retains magnitude prediction and noisy phase for
  waveform reconstruction.

* :ref:`masterthesis/vit_spectrogram/deployment <diagram-masterthesis-vit-spectrogram-deployment>`:
  This deployment variant replaces magnitude-only prediction with a complex
  spectrogram mask. It can change phase and magnitude, reconstructs a waveform, and
  returns the centre artifact in input units. It trains against recovered clean EEG.

* :ref:`masterthesis/vit_spectrogram <diagram-masterthesis-vit-spectrogram>`:
  A patch Transformer reconstructs clean spectrogram magnitude and combines it with
  noisy phase. The adapter derives the centre artifact from the reconstructed clean
  waveform.

Source-paper record
-------------------

Sources: [Dosovitskiy2021]_, [He2022]_. See :doc:`../model_references` for full references.

A source citation explains the design lineage. It does not establish that a
FACETpy checkpoint reproduces the source paper's training or results.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-vit-spectrogram` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
