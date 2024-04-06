Getting started: Averaged Artifact Subtraction (AAS) Correction of EDF EEG Data
===============================================================================

Introduction
------------

This document provides an overview of applying Averaged Artifact Subtraction (AAS) to EEG data using the FACET Tool. AAS is a technique used to calculate average artifacts, such as gradient induced artifacts, from EEG recordings, thereby improving the quality of the data for analysis if removed.

Prerequisites
-------------

Before applying AAS, ensure you have the following:

- MNE-Python installed in your environment
- Having installed the rest of the necessary packages.
- An EEG dataset imported with the analyis Framework

Installation of necessary packages can be done using pip:

.. code-block:: bash
   # Optional: Create a virtual environment
   pip install virtualenv
   python -m venv venv
   # On Windows
   venv\Scripts\activate.bat
   # On Unix or MacOS
   source venv/bin/activate

   # Mandatory: Install the necessary packages
   pip install -r requirements.txt

Loading Your EEG Data
---------------------

.. note::
   
   This tutorial is based on the default/stock FACET object implementation. The FACET Object is intented to be individualized to the user's needs.
   To access all functionalities (if not already provided) of the FACET Tool, either edit/expand the `src/FACET/Facet.py` file or directly access the frameworks by e.g. `f.get_correction()` for the correction framework instance.

To begin, load your EEG dataset into FACET object:

.. code-block:: python
   
   from src.FACET.Facet import Facet
   #It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
   #Begin Configuration Block
   # Path to your EEG file
   file_path = 'path/to/your/eeg_file.edf'
   # Event Regex assuming using stim channel
   event_regex = r'\b1\b'
   # Upsampling factor
   upsample_factor = 10
   #End Configuration Block

   # Loading the EEG data by creating a FACET object and importing the EEG data
   f = FACET()
   f.import_EEG(file_path, upsample_factor)

Preprocessing
-------------
If you have not already preprocessed your EEG data, you may want to perform some preprocessing steps before applying AAS. 
This can include filtering and resampling.

.. note::

   The stock FACET object provides a `pre_processing` method, which is a general predefined collection of preprocessing steps. 
   This can and should be individualized to the user's needs.
   `f.pre_processing()`

.. code-block:: python

	f.highpass(1)
	f.upsample() # upsampling factor must be specified on importing the EEG data

Finding triggers
----------------

Before applying AAS, you need to specify which events in your EEG data will be used as triggers.
This is done using the `find_triggers` method, which takes an event regex as an argument.
The event regex is a regular expression that matches the events you want to use as triggers.


.. code-block:: python

   f.find_triggers(event_regex)

Applying Averaged Artifact Subtraction
--------------------------------------

Before you can remove artifacts you must calculate the average artifact. This is done 
Once your triggers are specified, apply AAS to correct for artifacts:

.. code-block:: python

   f.calc_matrix_AAS()

.. important::

   This only calculates the average artifact. To remove the average artifact from the EEG data, you must call the `f.remove_artifacts` method.

Removing Artifacts
------------------

After calculating the average artifact, you can remove the average artifact from the EEG data:

.. code-block:: python

   f.remove_artifacts()

Further Processing
------------------

After removing artifacts, you can proceed with further EEG data processing, such as filtering, and downsampling.

.. note::

   The stock FACET object provides a `post_processing` method, which is a general predefined collection of postprocessing steps. 
   Again, this can and should be individualized to the user's needs.
   `f.post_processing()`

.. code-block:: python

   # Example: Applying a low-pass filter
   f.downsample() # downsampling by upsample factor
   f.lowpass(50)

Plotting the Processed EEG Data
-------------------------------
If you want to visualize the processed EEG data, you can use the `plot_EEG` method.

.. code-block:: python

   f.plot_EEG()

Exporting the Processed EEG Data
--------------------------------
After processing your EEG data, you may want to export the processed data to a file.
This can be done using the `export_EEG` method, which takes the file path as an argument.

.. code-block:: python

   f.export_EEG('path/to/your/processed_eeg_file.edf')

Conclusion
----------

Applying Averaged Artifact Subtraction (AAS) is crucial for preparing EEG data for analysis by reducing noise and artifacts. This documentation outlined the steps to apply AAS using FACET, from loading your EEG data to applying the AAS correction.

For more detailed information on processing EEG data with MNE-Python, refer to the official MNE-Python documentation.
