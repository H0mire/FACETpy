Averaged Artifact Subtraction (AAS) Correction in EEG Data
===========================================================

Introduction
------------

This document provides an overview of applying Averaged Artifact Subtraction (AAS) to EEG data using the MNE-Python library. AAS is a technique used to reduce noise and artifacts, such as eye blinks or heartbeats, from EEG recordings, thereby improving the quality of the data for analysis.

Prerequisites
-------------

Before applying AAS, ensure you have the following:

- MNE-Python installed in your environment
- An EEG dataset imported with the analyis Framework

Installation of MNE-Python can be done using pip:

.. code-block:: bash

   pip install mne

Loading Your EEG Data
---------------------

To begin, load your EEG dataset into an MNE `Raw` object:

.. code-block:: python

   import mne

   # Path to your EEG file
   file_path = 'path/to/your/eeg_file.edf'
   # Event Regex assuming using stim channel
   event_regex = r'\b1\b'

   # Loading the EEG data
   f = new FACET()
   f.import_EEG(file_path)

Applying Averaged Artifact Subtraction
--------------------------------------

Once your data is loaded, apply AAS to correct for artifacts:

.. code-block:: python

   f.find_triggers(event_regex)
   f.applyAAS()
   f.remove_artifacts()


Further Processing
------------------

After applying AAS, you can proceed with further EEG data processing, such as filtering, epoching, and analysis.

.. code-block:: python

   # Example: Applying a band-pass filter
   f.lowpass()

Conclusion
----------

Applying Averaged Artifact Subtraction (AAS) is crucial for preparing EEG data for analysis by reducing noise and artifacts. This documentation outlined the steps to apply AAS using MNE-Python, from loading your EEG data to applying the AAS correction.

For more detailed information on processing EEG data with MNE-Python, refer to the official MNE-Python documentation.
