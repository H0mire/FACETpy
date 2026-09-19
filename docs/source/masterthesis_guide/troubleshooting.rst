Troubleshooting
===============

macOS compiler selection
------------------------

If ``build-fastranc`` fails because the selected Xcode installation requires
license setup, an existing Command Line Tools installation can provide the
compiler. When ``/Library/Developer/CommandLineTools`` is installed, select it for
this command:

.. code-block:: console

   DEVELOPER_DIR=/Library/Developer/CommandLineTools uv run build-fastranc

This sets the toolchain for one invocation. It does not accept an Xcode license
or change the system-wide developer-directory setting.

Missing model artifact
----------------------

Check the experiment's artifact list and materialize the selected LFS object.
The resolver distinguishes a missing file from an LFS pointer. Copying a pointer
to another directory does not provide model weights. Run the catalog validator
with ``--hashes`` to verify local binary contents; this reads all selected files.

Missing dataset
---------------

Provide ``--data-root`` or set ``FACETPY_ARTIFACT_DIR``. Preserve the catalogued
relative path, original data version and saved split. A similarly named newer
bundle is not an equivalent substitute.

Device-specific export
----------------------

Some original traces contain CUDA device constants. Select the recorded CPU
export for CPU or MPS where one is available. DenoiseMamba's Phase-1 path rebuilds
from source weights; D4PM needs its reverse-diffusion sampler rather than the
original TorchScript stub.

Channel or shape mismatch
-------------------------

Use the exact family variant and input contract. Multichannel models require the
recorded montage order. A state-dictionary mismatch must be resolved through the
correct factory and configuration, not by silently ignoring unmatched parameters.

A score differs
---------------

Check the checkpoint identity, saved indices, target construction, units,
normalization, trigger offset, filtering and metric definition. Record the actual
comparison and its tolerance. Do not loosen a tolerance to make a migration pass
or compare different signal states under the same result label.

Index or documentation is stale
-------------------------------

.. code-block:: console

   uv run python -m masterthesis_guide.reproduce index
   uv run python -m masterthesis_guide.reproduce index --check
   uv run --extra docs sphinx-build -W -b html docs/source docs/build/html

The documentation build and ordinary tests do not download weights or recordings.
The ``docs`` extra installs Sphinx and its theme; plain ``uv sync`` does not include
them. Open ``docs/build/html/index.html`` after the build.
