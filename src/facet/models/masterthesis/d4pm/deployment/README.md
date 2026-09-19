# D4PM — deployment

Conditional diffusion model. This deployment variant normalizes the training pair and adds a recovered-waveform objective at configured diffusion timesteps. Inference still requires an iterative sampler. The retained record contains no completed Phase-2 pipeline result.

## Origin and role

Origin: EEG artifact removal.

Sources: [Shao2025](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#shao2025).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/d4pm.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.d4pm.deployment.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
