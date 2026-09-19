# DPAE — deployment

Dual-pathway convolutional autoencoder. Two convolutional pathways combine fine and coarse temporal features to predict a single-channel artifact. The deployment wrapper normalizes input, restores output units and can remove the predicted epoch mean. Its objective scores recovered clean EEG.

## Origin and role

Origin: EEG denoising.

Sources: [Xiong2024](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#xiong2024).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/dpae.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.dpae.deployment.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
