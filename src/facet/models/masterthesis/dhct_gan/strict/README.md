# DHCT-GAN strict

Dual-branch convolution and attention model. Independent CNN/Transformer branches return clean, noise and fused outputs for adversarial training. An optional electrode-attention bridge and configurable decoder extend the source design; identify both in any reported result.

## Origin and role

Origin: EEG artifact removal.

Sources: [Cai2025](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#cai2025).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/dhct_gan.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.dhct_gan.strict.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
