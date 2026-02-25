Installation
============

Requirements
------------

* Python 3.11 or 3.12
* MNE-Python 1.10.2
* NumPy 2.1.3
* SciPy >= 1.15.3
* scikit-learn >= 1.4.2
* `Poetry <https://python-poetry.org>`_ >= 1.8 (package manager)

Install from Source
-------------------

FACETpy is managed with Poetry. The quickest setup is the bootstrap URL:

.. code-block:: bash

   curl -fsSL https://raw.githubusercontent.com/H0mire/facetpy/main/scripts/bootstrap.sh | sh
   cd facetpy

This clones FACETpy into ``./facetpy`` and runs the installer.

You can also clone first and run the local installer:

.. code-block:: bash

   git clone https://github.com/H0mire/facetpy.git
   cd facetpy
   ./scripts/install.sh

The script:

* checks for Python 3.11/3.12
* checks whether Poetry is installed
* prompts to install Poetry if missing
* runs ``poetry install --no-interaction``

Manual installation:

.. code-block:: bash

   poetry install

Run commands inside the Poetry environment:

.. code-block:: bash

   poetry run pytest

Optional extras:

.. code-block:: bash

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

For faster Adaptive Noise Cancellation, compile the C extension once after installing:

.. code-block:: bash

   poetry run build-fastranc

This will create:

* ``libfastranc.so`` (Linux)
* ``libfastranc.dylib`` (macOS)
* ``fastranc.dll`` (Windows)

If the C extension is not compiled, ``ANCCorrection`` is unavailable but the rest of
the toolbox works normally.

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

   poetry install
