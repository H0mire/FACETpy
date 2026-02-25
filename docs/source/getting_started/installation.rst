Installation
============

Requirements
------------

* Python 3.11 or 3.12
* pip (ships with every Python installer — no extra package manager required)

The package metadata follows PEP 517/518 so ``pip install .`` resolves and
installs all dependencies automatically.

Install from Source
-------------------

Clone the repository and install with pip:

.. code-block:: bash

   git clone https://github.com/H0mire/facetpy.git
   cd facetpy

   # create and activate an isolated environment (recommended)
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate

   pip install .

Optional extras
~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install ".[deeplearning]"   # TensorFlow-based extras
   pip install ".[notebooks]"      # Jupyter notebook support
   pip install ".[gui]"            # PyQt6 GUI components
   pip install ".[docs]"           # Sphinx + MyST documentation toolchain
   pip install ".[all]"            # all optional dependencies

Conda environment
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n facetpy python=3.12 -y
   conda activate facetpy
   pip install .

Development Installation
------------------------

For an editable install that picks up source changes immediately, plus
the test and lint toolchain:

.. code-block:: bash

   pip install -e ".[dev]"

Alternatively, use the explicit requirements files:

.. code-block:: bash

   pip install -r requirements-dev.txt
   pip install -e .

Run tests directly:

.. code-block:: bash

   pytest

.. note::

   **Poetry users** — the ``pyproject.toml`` is still fully Poetry-compatible.
   Running ``poetry install`` continues to work as before.

Optional Components
-------------------

C Extension for ANC
~~~~~~~~~~~~~~~~~~~

For faster Adaptive Noise Cancellation, compile the C extension once after installing:

.. code-block:: bash

   python -m facet.build

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

   pip install -e .
