# DenoiseMamba — deployment

Convolutional and state-space model. A flat stack of local convolution and Mamba-1-style selective state-space blocks predicts a single-channel artifact. This is the thesis reconstruction, not the full published ConvSSD architecture. The deployment wrapper normalizes input, restores output units and can remove the predicted epoch mean. Its objective scores recovered clean EEG.

## Origin and role

Origin: EEG artifact removal.

Sources: [Chen2025](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#chen2025).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/denoise_mamba.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.denoise_mamba.deployment.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
