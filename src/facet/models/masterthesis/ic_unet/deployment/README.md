# IC-U-Net — deployment

Multichannel U-Net denoising autoencoder. A multichannel U-Net operates between frozen ICA and inverse-ICA transforms and predicts the centre-epoch artifact. The in-model ICA transforms are a FACETpy adaptation. The deployment wrapper normalizes input, restores output units and can remove the predicted epoch mean. Its objective scores recovered clean EEG.

## Origin and role

Origin: EEG artifact removal using ICA-derived training pairs.

Sources: [Chuang2022](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#chuang2022).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/ic_unet.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.ic_unet.deployment.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
