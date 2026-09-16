ViT-Spectrogram
===============

A spectrogram model inspired by ViT and masked autoencoding. The deployment variant reconstructs the centre epoch through a complex mask and inverse STFT.

Implementation and input
------------------------

The family is owned by ``facet.models.masterthesis.vit_spectrogram``. The recorded adapter contract is
**seven epochs, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base implementation and ``deployment`` variant retain separate factories.
The deployment wrapper changes the objective and normalization; the catalog
selects its recorded configuration and artifact. A seven-epoch dataset does not
mean that every model consumes all seven epochs: single-epoch adapters select
the centre epoch.

The Phase-1 model predicts clean EEG. Its adapter subtracts this prediction
from the demeaned noisy centre epoch to obtain the artifact. The deployment
variant returns an artifact estimate through its own reconstruction wrapper.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-vit-spectrogram` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
