# ViT Spectrogram

Spectrogram Transformer with masked reconstruction. A patch Transformer reconstructs clean spectrogram magnitude and combines it with noisy phase. The adapter derives the centre artifact from the reconstructed clean waveform.

## Origin and role

Origin: Image recognition and masked image reconstruction.

Sources: [Dosovitskiy2021](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#dosovitskiy2021) · [He2022](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#he2022).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/vit_spectrogram.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.vit_spectrogram.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
