Getting started: Full Averaged Artifact Subtraction (AAS) Correction of BIDS EEG Data
====================================================================================

Introduction
------------

This document explains how to use the FACET framework to full extend. It tells how to import EEG data from BIDS, apply alignment, and apply Averaged Artifact Subtraction (AAS) to correct for artifacts. This document assumes you have already installed FACET and have a basic understanding of how to use it.

Prerequisites
-------------

Before applying AAS, ensure you have the following:

- MNE-Python installed in your environment.
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
   # Path to your BIDS dataset
   bids_path = "F:\EEG Datasets\openneuro\FMRIWITHMOTION"
   # Event Regex assuming using stim channel
   event_regex = r'\b1\b'
   # Upsampling factor
   upsample_factor = 10
   # Assuming you want to use the same path for exporting the data
   export_bids_path = bids_path
   # Some settings regarding your BIDS dataset
   subject = "01"
   session = "01"
   task = "eegfmriNF"

   # Now some settings for the AAS
   window_size = 25
   relative_window_position = -0.5   
   artifact_to_trigger_offset_in_seconds = -6
   moosmann_motion_threshold = 0.8
   event_id_description_pairs={'trigger':1}
   regex_trigger_annotation_filter = r'\bYour Trigger Tag\b' # Annotations with the description 'trigger' are considered as triggers
   unwanted_bad_channels = ['EKG', 'EMG', 'EOG', 'ECG'] # Channels with these names are considered as bad channels and not considered in the processing
   evaluation_measures=["SNR", "RMS", "RMS2", "MEDIAN"] # Evaluation measures to be used for the evaluation of the AAS
   #End Configuration Block

   # Loading the EEG data by creating a FACET object and importing the EEG data
   f = FACET()
   f.import_EEG(path=bids_path,fmt="bids",upsampling_factor=upsampling_factor,artifact_to_trigger_offset=artifact_to_trigger_offset_in_seconds, bads=unwanted_bad_channels, subject=subject, session=session,task=task)


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
   f.prepare_ANC() # prepare the ANC for the AAS

Finding triggers
----------------

Before applying AAS, you need to specify which events in your EEG data will be used as triggers.
This is done using the `find_triggers` method, which takes an event regex as an argument.
The event regex is a regular expression that matches the events you want to use as triggers.


.. code-block:: python

   f.find_triggers(event_regex)

Aligning Triggers
-----------------

Before applying AAS, you neet to align the triggers so it matches the Slice Gradient artifact.

.. code-block:: python

   reference_trigger = 0
   f.align_slices(0)

Applying Averaged Artifact Subtraction
--------------------------------------

Before you can remove artifacts you must calculate the average artifact. This is done 
Once your triggers are specified, apply AAS to correct for artifacts:

.. code-block:: python

   f.apply_AAS()

.. important::

   This only calculates the average artifact. To remove the average artifact from the EEG data, you must call the `f.remove_artifacts` method.

Removing Artifacts
------------------

After calculating the average artifact, you can remove the average artifact from the EEG data:

.. code-block:: python

   f.remove_artifacts()

Further Processing
------------------

After removing artifacts, you can proceed with further EEG data processing, such as filtering, adaptive noise cancelation, and downsampling.

.. note::

   The stock FACET object provides a `post_processing` method, which is a general predefined collection of postprocessing steps. 
   Again, this can and should be individualized to the user's needs.
   `f.post_processing()`

.. code-block:: python

   # Example: Applying a low-pass filter
   f.downsample() # downsampling by upsample factor
   f.lowpass(50)
   f.apply_ANC() # apply the ANC to the EEG data

Plotting the Processed EEG Data
-------------------------------
If you want to visualize the processed EEG data, you can use the `plot_EEG` method.

.. code-block:: python

   f.plot_EEG()

Evaluating the Processed EEG Data
---------------------------------
If you want to evaluate the processed EEG data, you add the eeg data to the evaluation framework and call the `evaluate` method.

.. code-block:: python

   f.add_to_evaluate(f.get_EEG(), name="Corrected EEG")
   f.evaluate(plot=true, measures=evaluation_measures)

Exporting the Processed EEG Data
--------------------------------
After processing your EEG data, you may want to export the processed data to a file.
This can be done using the `export_EEG` method, which takes the file path as an argument.

.. code-block:: python

   f.export_EEG(event_id=event_id_description_pairs, path=export_bids_path, fmt="bids")

Conclusion
----------

Applying Averaged Artifact Subtraction (AAS) is crucial for preparing EEG data for analysis by reducing noise and artifacts. This documentation outlined the steps to apply AAS using FACET, from loading your EEG data to applying the AAS correction.

For more detailed information on processing EEG data with MNE-Python, refer to the official MNE-Python documentation.
