Nested GAN
==========

A nested time-frequency and time-domain adversarial model.

Implementation and input
------------------------

The family is owned by ``facet.models.masterthesis.nested_gan``. The recorded adapter contract is
**seven epochs, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base implementation and ``deployment`` variant retain separate factories.
The deployment wrapper changes the objective and normalization; the catalog
selects its recorded configuration and artifact. A seven-epoch dataset does not
mean that every model consumes all seven epochs: single-epoch adapters select
the centre epoch.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Source-paper record
-------------------

Yang et al. (2025), End-to-End EEG Artifact Removal Method via Nested Generative Adversarial Network. The citation is retained from the thesis source records.
