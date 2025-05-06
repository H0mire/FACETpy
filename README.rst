
.. image:: docs/source/_static/logo.png
   :align: center
   :width: 300px

FACET Python Toolbox
###################

.. image:: https://readthedocs.org/projects/facetpy/badge/?version=latest
    :target: https://facetpy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

A powerful Python toolbox for correcting EEG artifacts in simultaneous EEG-fMRI recordings using Averaged Artifact Subtraction (AAS).

Features
--------
- Import EEG data from various formats including BIDS and EDF
- Advanced artifact detection and correction using AAS
- Comprehensive evaluation framework
- Flexible processing pipeline
- Built on top of MNE-Python

Documentation
------------
Full documentation is available at `<https://facetpy.readthedocs.io/>`_

Installation
-----------
We recommend using Poetry 1.4+ for installation:

.. code-block:: bash

   # Install Poetry if needed
   conda install -c conda-forge poetry=1.4

   # Install project dependencies
   poetry install

   # Activate virtual environment
   poetry shell

Documentation Development
-----------------------
To build and edit the documentation:

1. Install Sphinx dependencies:

   .. code-block:: bash

      poetry install --with docs

2. Navigate to docs directory:

   .. code-block:: bash
   
      cd docs

3. Build documentation:

   .. code-block:: bash

      # Build HTML docs
      make html

      # Auto-rebuild on changes
      make livehtml

The compiled documentation will be available in ``docs/build/html/``.
