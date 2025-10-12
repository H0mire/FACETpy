Installation
============

Requirements
------------

* Python 3.8 or higher
* MNE-Python >= 1.0.0
* NumPy >= 1.20.0
* SciPy >= 1.7.0
* scikit-learn >= 1.0.0

Install from PyPI
-----------------

The easiest way to install FACETpy is via pip:

.. code-block:: bash

   pip install facetpy

This will install FACETpy and all required dependencies.

Install from Source
-------------------

To get the latest development version:

.. code-block:: bash

   git clone https://github.com/your-org/facetpy.git
   cd facetpy
   pip install -e .

Development Installation
------------------------

For development, install with additional dependencies:

.. code-block:: bash

   git clone https://github.com/your-org/facetpy.git
   cd facetpy
   pip install -e ".[dev]"

This includes:

* pytest - for running tests
* sphinx - for building documentation
* black - for code formatting
* mypy - for type checking

Optional Components
-------------------

C Extension for ANC
~~~~~~~~~~~~~~~~~~~

For faster Adaptive Noise Cancellation, compile the C extension:

.. code-block:: bash

   cd src/facet/helpers
   python build.py

This will create:

* ``libfastranc.so`` (Linux)
* ``libfastranc.dylib`` (macOS)
* ``fastranc.dll`` (Windows)

If the C extension is not available, FACETpy will automatically fall back to a Python implementation.

Verify Installation
-------------------

To verify your installation:

.. code-block:: python

   import facet
   print(facet.__version__)
   print(facet.list_processors())

This should print the version number and list of available processors.

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

If you get import errors, ensure all dependencies are installed:

.. code-block:: bash

   pip install -r requirements.txt

Permission Errors
~~~~~~~~~~~~~~~~~

On Linux/macOS, if you encounter permission errors:

.. code-block:: bash

   pip install --user facetpy

Or use a virtual environment:

.. code-block:: bash

   python -m venv facetpy_env
   source facetpy_env/bin/activate  # On Windows: facetpy_env\Scripts\activate
   pip install facetpy
