Where in FACETpy: code provenance
=================================

The published thesis snapshot is identified by the annotated
`masterwork tag <https://github.com/H0mire/FACETpy/tree/refs/tags/masterwork>`_.
The ``masterwork`` branch also provides this thesis collection. Resolve the tag
to its full commit hash and check out the recorded snapshot:

.. code-block:: console

   git rev-parse 'refs/tags/masterwork^{commit}'
   git checkout --detach refs/tags/masterwork

Model families retain their original
module identity in the catalog alongside the migrated Python module. Original
resolved configurations and compact run records remain unchanged under
``provenance``; executable configurations contain the migrated imports.

The migration manifest records source and destination paths and available hashes.
A source snapshot identifies the retained code. It does not establish the exact
training revision of a run whose original metadata did not record one.

Scientific outcome, artifact availability and reproduction verification are
separate fields. Missing provenance remains an explicit gap. A reconstructed
configuration must be identified as reconstructed and must not replace an
original record.

The fourteen cited model-reference titles, document paths and inventoried anchors
are protected by a reference fixture. A successful local documentation build
checks those generated pages; it does not publish them to Read the Docs.

Catalog contract and verification
---------------------------------

``catalog.yaml`` uses schema version 1. The small validator checks unique IDs,
repository paths, experiment/model/dataset associations, selected artifact
ownership, scientific outcome, verification state, required reasons, LFS pointer
contents and protected original figure hashes. ``validate --hashes`` also checks
all materialized weight bytes. Normal CI accepts valid LFS pointers and does not
fetch weights.

Original Phase-1 prediction arrays are also retained in Git LFS. Their catalog
records preserve the source paths and commit as provenance. Plotting reads the
current repository paths and verifies the recorded bytes; it does not need the
historical source checkout.

The verification records distinguish fixed-input model relocation, two real
Phase-1 holdout windows, synthetic complete adapter correction, and the new full
Phase-2 holdout replay. Each record states its own input scope and tolerance.
Passing an adapter comparison does not verify an unperformed training run or a
historical full-pipeline result.

The tool-disposition manifest records the owner or removal reason for every
original tool. Intermediate spike-aware weights that were not used by the
completed comparisons remain in the external originals; their exclusion record
preserves their hashes. Evaluated best checkpoints and the corresponding
pipeline exports remain selected in LFS.
