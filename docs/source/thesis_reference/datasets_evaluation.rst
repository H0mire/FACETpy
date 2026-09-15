Datasets and evaluation protocol
================================

Large recordings and tensor bundles remain external. The repository keeps their
available metadata, lineage, saved split records and experiment associations.
Use an explicit local data root; examples do not assume the original author's
filesystem paths.

Preserve each recorded version. In particular, the original proof-fit bundle,
Weg-A v7 and the later locked v10 bundles are not interchangeable. Consult each
version's metadata for clean-source construction, artifact source, context shape,
guards, channel selection and sampling frequency. Do not infer those facts from
a shared family name or from a later construction document.

A SHA-256 of a metadata file identifies that record, not the whole dataset. Dataset
binary hashes and split hashes must be labelled separately. Where a dataset
carries its split internally, retain that split when preparing another storage
format. A seed alone does not identify the original set of examples.

Metrics retain their recorded units and signal state. A residual before the final
low-pass filter and one after it are different measurements. Likewise, AAS-derived
clean targets and independently constructed clean targets test different claims.

See :doc:`../masterthesis_guide/quickstart` for local path configuration and
:doc:`../masterthesis_guide/catalog` for the dataset records.
