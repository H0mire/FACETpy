# D4PM — experimental paper_accurate variant

Conditional diffusion model. This experimental variant adds continuous noise-level and artifact-class conditioning. An optional clean branch supports joint posterior sampling; enable it explicitly in the configuration.

## Origin and role

Origin: EEG artifact removal.

Sources: [Shao2025](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#shao2025).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/d4pm.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.paper_accurate.d4pm.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
