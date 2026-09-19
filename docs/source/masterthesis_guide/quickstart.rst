Quickstart
==========

Two ways to use the project
---------------------------

An installed ``facetpy`` package accepts explicit model and artifact paths. It does
not require this repository, its thesis catalog or the original author's dataset
folders. Install the ``pytorch`` extra for the PyTorch models.

The repository guide additionally resolves experiment IDs, configurations and
recorded artifacts. Run its commands from the checkout root. Use ``uv sync`` to
install the project's development environment and ``uv run build-fastranc`` to
compile the native correction extension where required.

.. code-block:: console

   uv sync
   uv run build-fastranc
   uv run python -m masterthesis_guide.reproduce validate

If the compiler fails on macOS, see :doc:`troubleshooting`. To build the local
documentation, request the optional documentation dependencies explicitly:

.. code-block:: console

   uv run --extra docs sphinx-build -W -b html docs/source docs/build/html

Data and weights
----------------

Set ``FACETPY_ARTIFACT_DIR`` to a local directory containing the original relative
paths recorded by the catalog. A guide command can instead take ``--data-root``.
Neither option downloads a dataset or changes its split.

.. code-block:: console

   export FACETPY_ARTIFACT_DIR=/path/to/local/thesis-data

Selected checkpoints and exports live under ``artifacts/`` through Git LFS. A
pointer file is not a model. The resolver reports a missing binary before model
loading. Selected retrieval avoids downloading the complete thesis artifact set.

The original Phase-1 prediction arrays also live in Git LFS, under
``artifacts/predictions/phase_1/``. See :doc:`reproduce_results` for the targeted
download and plotting command. These predictions require no old source checkout.

For a separately published checkout, ``GIT_LFS_SKIP_SMUDGE=1 git clone ...`` skips
automatic materialization. Then request only the paths needed by the chosen
experiment with ``git lfs pull --include="<catalogued artifact path>" --exclude=""``.
These commands require that the corresponding LFS objects have actually been
published and that the user has access. The local integration itself uploads no
objects and does not establish public availability.

Run an experiment
-----------------

Find an ID in :doc:`catalog`, resolve its data and weights, then run the example:

.. code-block:: console

   uv run python -m masterthesis_guide.examples.run_model deployment_demucs \
     --edf /path/to/NiazyFMRI.edf --out /path/to/results/demucs_raw.fif

.. literalinclude:: ../../../masterthesis_guide/examples/run_model.py
   :language: python

A synthetic input can test shape and pipeline behavior. It does not reproduce a
published score. Use the original data, saved split and evaluated artifact for a
numerical comparison.

Acquisition scopes
------------------

* For a smoke check, use synthetic fixtures and the declared backend extra. No
  research recording or historical checkpoint is required.
* For one recorded experiment, obtain only its catalogued data version, split,
  configuration and evaluated artifact. Materialize the needed LFS paths.
* For a complete thesis replay, obtain the full selected LFS set, every referenced
  external dataset and the figure-specific signal/prediction inputs. Include
  negative and earlier-phase results. The catalog's genuine source gaps still
  apply; a full download cannot recreate absent provenance.

Dataset hosting is not established by this local integration. Local originals,
the source Git snapshot and the ignored-file archive have different coverage.
Keep all three until a separately approved distribution has been verified.
