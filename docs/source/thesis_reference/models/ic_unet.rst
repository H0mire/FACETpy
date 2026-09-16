IC-U-Net
========

An ICA-inspired multichannel U-Net adaptation.

Implementation and input
------------------------

The family is owned by ``facet.models.masterthesis.ic_unet``. The recorded adapter contract is
**seven concatenated epochs, 30 channels in the recorded order**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base implementation and ``deployment`` variant retain separate factories.
The deployment wrapper changes the objective and normalization; the catalog
selects its recorded configuration and artifact. A seven-epoch dataset does not
mean that every model consumes all seven epochs: single-epoch adapters select
the centre epoch.

Phase-2 pipeline output was flagged as invalid because of discontinuities.
Its recorded numerical residual must not be interpreted as successful correction.
The evidence and weights are retained so that the failure remains inspectable.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Source-paper record
-------------------

Chuang et al. (2022), IC-U-Net. The citation is retained from the thesis source records.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-ic-unet` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
