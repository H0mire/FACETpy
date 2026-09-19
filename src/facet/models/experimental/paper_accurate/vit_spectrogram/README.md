# ViT Spectrogram — experimental paper_accurate variant

Spectrogram Transformer with masked reconstruction. This experimental variant encodes visible patches and reconstructs masked patches with a separate MAE decoder. It retains magnitude prediction and noisy phase for waveform reconstruction.

## Origin and role

Origin: Image recognition and masked image reconstruction.

Sources: [Dosovitskiy2021](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#dosovitskiy2021) · [He2022](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#he2022).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/vit_spectrogram.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.paper_accurate.vit_spectrogram.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
