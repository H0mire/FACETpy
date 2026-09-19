# IC-U-Net — experimental paper_accurate variant

Multichannel U-Net denoising autoencoder. This experimental variant defaults to a sensor-space U-Net with clean-EEG output and a learned upsampling decoder. Frozen ICA is optional rather than part of the default inference path.

## Origin and role

Origin: EEG artifact removal using ICA-derived training pairs.

Sources: [Chuang2022](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#chuang2022).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/ic_unet.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.paper_accurate.ic_unet.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
