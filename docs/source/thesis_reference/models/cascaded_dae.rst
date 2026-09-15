Cascaded DAE
============

A supervised cascaded autoencoder baseline developed within FACETpy.

Implementation and input
------------------------

The family is owned by ``facet.models.masterthesis.cascaded_dae``. The recorded adapter contract is
**single epoch, one channel**. Input packing, demeaning and reconstruction are part of the
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

The seven-epoch configuration in thesis Figure 8 and Table 5 belongs to
``deployment_cascaded_context_dae`` (batch size 64). Select that experiment for
those references. The single-epoch ``deployment_cascaded_dae`` arm (batch size
128) is a separate result. Both remain available; see :doc:`context_dae`.
