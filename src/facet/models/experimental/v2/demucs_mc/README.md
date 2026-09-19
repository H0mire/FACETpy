# Multichannel Demucs

Waveform encoder-decoder with electrode attention. This experimental Demucs extension combines a target electrode and its neighbours through electrode attention, then predicts the target-channel centre artifact.

## Origin and role

Origin: FACETpy extension of Demucs.

Sources: [Defossez2019](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#defossez2019).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/multichannel_demucs.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.v2.demucs_mc.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
