# Demucs — experimental paper_accurate variant

Waveform encoder-decoder with recurrent processing. This experimental variant adds input-length alignment and optional internal resampling. It still predicts an EEG artifact and uses the FACETpy experiment configuration.

## Origin and role

Origin: Music source separation.

Sources: [Defossez2019](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#defossez2019).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/demucs.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.paper_accurate.demucs.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
