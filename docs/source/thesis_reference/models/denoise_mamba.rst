DenoiseMamba
============

A waveform model that combines convolutional features with selective state-space blocks.

Implementation and input
------------------------

The family is owned by ``facet.models.masterthesis.denoise_mamba``. The recorded adapter contract is
**single epoch, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base implementation and ``deployment`` variant retain separate factories.
The deployment wrapper changes the objective and normalization; the catalog
selects its recorded configuration and artifact. A seven-epoch dataset does not
mean that every model consumes all seven epochs: single-epoch adapters select
the centre epoch.

The Phase-1 CPU path rebuilds the model from its state dictionary because the
original traced state-space scan contains a CUDA device. Select the source
checkpoint for that protocol.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-denoise-mamba` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
