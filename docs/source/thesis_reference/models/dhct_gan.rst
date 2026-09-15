DHCT-GAN
========

A family of adversarial models. The single-epoch implementation and the context-aware V2 implementation have separate factories.

Implementation and input
------------------------

The family is owned by ``facet.models.masterthesis.dhct_gan``. The recorded adapter contract is
**single epoch for the base implementation; seven epochs for V2, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base implementation and ``deployment`` variant retain separate factories.
The deployment wrapper changes the objective and normalization; the catalog
selects its recorded configuration and artifact. A seven-epoch dataset does not
mean that every model consumes all seven epochs: single-epoch adapters select
the centre epoch.

The V2 implementation is under ``dhct_gan.v2``. It remains part of the thesis
collection because the thesis evaluates it. The two incomplete Phase-3 DHCT
trials do not constitute a completed 27-configuration tuning study.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Source-paper record
-------------------

Cai et al. (2025), DHCT-GAN. The citation is retained from the thesis source records.
