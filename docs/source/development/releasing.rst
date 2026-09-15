Releasing FACETpy
=================

The package is named ``facetpy`` and imports as ``facet``. The repository's
``publish.yml`` workflow publishes version tags through PyPI Trusted Publishing.
Configure the publisher for the repository and workflow before the first release.

Before a release, update the version in ``pyproject.toml`` and
``src/facet/__init__.py``. Run the tests, build the wheel and source distribution,
and inspect their contents. Check that neither distribution includes thesis
weights or large datasets.

.. code-block:: console

   uv sync --locked
   uv run pytest
   uv build --out-dir /path/to/release-artifacts
   uvx twine check /path/to/release-artifacts/facetpy-*

After approval to publish, create and push the matching ``vX.Y.Z`` tag. Verify the
workflow result and install the published version in a clean environment. Package
publication, Git LFS object publication and Read the Docs deployment are separate
operations. A successful package release does not publish thesis weights.
