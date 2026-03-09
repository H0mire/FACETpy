Installation
============

Requirements
------------

* Python 3.11, 3.12, or 3.13
* MNE-Python 1.10.2
* NumPy 2.1.3
* SciPy >= 1.15.3
* scikit-learn >= 1.4.2
* `Poetry <https://python-poetry.org>`_ >= 1.8 (contributors only)

Install from PyPI (Recommended)
-------------------------------

For normal usage, install FACETpy from PyPI. This is sufficient for running
your own scripts and the repository examples.

.. code-block:: bash

   pip install facetpy

The package name on PyPI is ``facetpy``; import it in Python as ``facet``.

Full Setup from Source (Contributing, Example Datasets/Scripts, Early Access to features)
-----------------------------------------------------------------------------------------

FACETpy is managed with Poetry.

Unix (macOS/Linux)
~~~~~~~~~~~~~~~~~~

For contributors on Unix-like systems, the quickest setup is the bootstrap URL:

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/H0mire/facetpy/main/scripts/bootstrap.sh | sh
   cd facetpy

You can also run the installer script in an existing clone:

.. code-block:: bash

   ./scripts/install.sh

Other platforms (including Windows)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For contributors outside Unix-like shells, use Poetry directly:

.. code-block:: bash

   git clone https://github.com/H0mire/facetpy.git
   cd facetpy
   poetry install --no-interaction

The Unix ``./scripts/install.sh`` script:

* checks for Python 3.11/3.12/3.13
* checks whether Poetry is installed
* prompts to install Poetry if missing
* runs ``poetry install --no-interaction``

The bootstrap script:

* clones FACETpy into ``./facetpy``
* runs ``./scripts/install.sh`` inside that clone

Manual Poetry setup (contributors):

.. code-block:: text

   poetry install --no-interaction

Run contributor commands inside the Poetry environment (Unix and Windows):

.. code-block:: text

   poetry run pytest

Optional extras (Unix and Windows):

.. code-block:: text

   poetry install -E deeplearning   # TensorFlow-based extras
   poetry install -E notebooks      # Jupyter notebook support
   poetry install -E gui            # PyQt6 GUI components
   poetry install -E docs           # Sphinx + MyST documentation toolchain
   poetry install -E all            # all optional dependencies

Development Installation
------------------------

The standard ``poetry install`` includes development dependencies such as
``pytest`` and ``ruff``. Documentation tooling (``sphinx``, ``myst-parser``)
is installed via ``-E docs``.

Optional Components
-------------------

C Extension for ANC
~~~~~~~~~~~~~~~~~~~

Strong recommendation: compile the FastRANC C extension once after installing.
ANC runs significantly faster with it.

Without Poetry:

.. code-block:: bash

   python -m facet.build

With Poetry:

.. code-block:: bash

   poetry run build-fastranc

This will create:

* ``libfastranc.so`` (Linux)
* ``libfastranc.dylib`` (macOS)
* ``fastranc.dll`` (Windows)

If the C extension is not compiled, ``ANCCorrection`` falls back to a slower
Python implementation and the rest of the toolbox still works.

Verify Installation
-------------------

To verify your installation:

.. code-block:: python

   import facet
   print(facet.__version__)

This should print the version number without errors.

Troubleshooting
---------------

MNE-Python Issues
~~~~~~~~~~~~~~~~~

If you encounter issues with MNE-Python installation:

.. code-block:: bash

   pip install --upgrade mne

See `MNE installation guide <https://mne.tools/stable/install/index.html>`_ for platform-specific instructions.

Import Errors
~~~~~~~~~~~~~

If you get import errors, reinstall dependencies:

.. code-block:: bash

   pip install --upgrade --force-reinstall facetpy

For contributor/source environments, run:

.. code-block:: bash

   poetry install --no-interaction
