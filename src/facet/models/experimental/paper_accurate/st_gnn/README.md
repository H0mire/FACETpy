# ST-GNN — experimental paper_accurate variant

Spatiotemporal graph convolution network. This experimental variant uses linear-sigmoid temporal gates, a graph bottleneck, layer normalization and spherical electrode distances. It reconstructs EEG waveforms rather than forecasting traffic or classifying recordings.

## Origin and role

Origin: Traffic forecasting and EEG graph modelling.

Sources: [Yu2018](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#yu2018) · [Defferrard2016](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#defferrard2016) · [Wagh2020](https://facetpy.readthedocs.io/en/latest/thesis_reference/model_references.html#wagh2020).

[Model reference](https://facetpy.readthedocs.io/en/latest/thesis_reference/models/st_gnn.html) explains the family, adaptations and limits.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.paper_accurate.st_gnn.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
