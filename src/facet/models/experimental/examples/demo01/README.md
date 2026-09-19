# Context CNN demo

A small context convolution network for synthetic spike-artifact examples. It stacks epochs as features and predicts the centre artifact. It is a teaching example, separate from the fully connected Context DAE and the thesis result models.

## Origin and role

Origin: FACETpy teaching example.

Developed within FACETpy; no dedicated source paper is assigned to this implementation.

[Model reference](https://facetpy.readthedocs.io/en/latest/user_guide/deep_learning.html) explains this teaching model and its training interface.

This package is experimental. Its name does not establish paper fidelity or equivalence to a thesis result.

Factories: `facet.models.experimental.examples.demo01.training`.

Select the configuration, preprocessing, output type and checkpoint together in the [experiment catalog](https://facetpy.readthedocs.io/en/latest/masterthesis_guide/catalog.html). Large weights are retrieved through Git LFS.

## Results

[Recorded results](results/index.rst) lists this variant's experiments, hyperparameters, training curves and evaluations. [Manifest](results/manifest.json) records source and checkpoint associations. Unassigned results are marked explicitly.

## Architecture diagrams

[Compact overview](diagrams/overview.svg) · [Detailed architecture](diagrams/architecture.svg).
The overview fits within half an A4 page at its native print size.
[Editable diagram content](diagrams/architecture.json) links the implementation sources.
