# ST-GNN

Spatiotemporal graph convolution network. Temporal convolutions and graph filters predict multichannel artifacts on a fixed electrode graph. Channel order and montage belong to the trained model contract.

## Origin and role

Origin: Traffic forecasting and EEG graph modelling.

Sources: [Yu2018](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#yu2018) · [Defferrard2016](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#defferrard2016) · [Wagh2020](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#wagh2020).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/st_gnn.html) explains the family, adaptations and limits.

Factories: `facet.models.masterthesis.st_gnn.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
