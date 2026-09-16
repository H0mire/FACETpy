D4PM
====

A conditional diffusion model for artifact estimation. The implemented conditional path differs from the dual-branch architecture described by the source paper.

Implementation and input
------------------------

The family is owned by ``facet.models.masterthesis.d4pm``. The recorded adapter contract is
**single epoch, one channel**. Input packing, demeaning and reconstruction are part of the
experiment; a family name alone does not identify them.

The base implementation and ``deployment`` variant retain separate factories.
The deployment wrapper changes the objective and normalization; the catalog
selects its recorded configuration and artifact. A seven-epoch dataset does not
mean that every model consumes all seven epochs: single-epoch adapters select
the centre epoch.

Phase 1 uses the Python training module and its reverse-diffusion sampler.
The original TorchScript stub does not contain that sampler. Phase 2 has no
completed pipeline result for D4PM; preserved training records do not establish
successful deployment.

Experiments and evidence
------------------------

Use :doc:`../../masterthesis_guide/catalog` to select the phase, exact variant,
configuration and weights. :doc:`../selected_variants` explains how comparisons
differ. The family reference is not a claim of validated paper fidelity.

Architecture diagrams
---------------------

See :ref:`diagram-masterthesis-d4pm` for the compact overview and detailed implementation diagram.
The :doc:`model diagram gallery </model_diagrams>` also lists the separate variants.
