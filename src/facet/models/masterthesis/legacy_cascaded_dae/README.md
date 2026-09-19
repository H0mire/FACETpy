# Legacy Cascaded DAE

Fully connected denoising autoencoder. Migrated Phase-0 adapter: two fully connected autoencoders receive the same flattened multichannel epoch, and their predictions are added. The adapter changes segmentation and resampling; it does not reproduce the original FACETpy 0.1.0 environment.

## Origin and role

Origin: Duffy et al. (2020), gradient artifact correction in simultaneous EEG-fMRI; adapted in FACETpy.

Source: [Duffy et al. (2020)](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#duffy2020). The family reference explains the FACETpy adaptations.

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/cascaded_dae.html) explains the family, adaptations and limits.

[Phase-0 execution guide](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/legacy_execution.html) distinguishes the original evidence from the migrated adapter.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
