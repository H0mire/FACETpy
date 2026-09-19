# Master’s thesis guide

Start with [the evidence index](INDEX.rst) to find the experiment behind a thesis figure or table.
The English [Sphinx guide](../docs/source/masterthesis_guide/index.rst) explains setup and reproduction.

- Model implementations: `src/facet/models/masterthesis/` and `src/facet/models/experimental/`.
- Selected weights: `artifacts/`, tracked through Git LFS.
- Original Phase-1 predictions: `artifacts/predictions/phase_1/`, tracked through Git LFS.
- Large datasets: external; paths and checksums are recorded in `catalog.yaml`.
- Run examples from the repository root with `python -m masterthesis_guide.examples.<name>`.

Original measurements and run records are evidence. Generate new outputs into a separate directory.
