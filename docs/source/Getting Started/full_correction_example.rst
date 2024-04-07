Getting Started: Full Averaged Artifact Subtraction (AAS) Correction of BIDS EEG Data
=====================================================================================

Introduction
------------

This document explains how to use the facet framework to its full extent. It describes how to import EEG data with a BIDS structure, apply alignment, and apply Averaged Artifact Subtraction (AAS) to correct for artifacts. This document assumes you have already installed facet and have a basic understanding of how to use it.

Prerequisites
-------------

Before applying AAS, ensure you have the following:

- MNE-Python installed in your environment.
- The rest of the necessary packages installed.
- An EEG dataset imported with the analysis Framework

You can install the necessary packages using pip:

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
   
   This tutorial is based on the default/stock facet object implementation. The facet Object is intended to be individualized to the user's needs.
   To access all functionalities (if not already provided) of the facet Tool, either edit/expand the `src/facet/facet.py` file or directly access the frameworks by e.g. `f.get_correction()` for the correction framework instance.

To begin, load your EEG dataset into a facet object:

.. code-block:: python
   
   from src.facet.facet import facet
   # It is advised to add a configuration block here, to keep an overview of the settings used for the analysis.
   # Begin Configuration Block
   # Path to your BIDS dataset
   bids_path = "F:\EEG Datasets\openneuro\FMRIWITHMOTION"
   # Annotations with the description 'Your Trigger Tag' are considered as triggers
   regex_trigger_annotation_filter = r'\bYour Trigger Tag\b'
   # Upsampling factor
   upsample_factor = 10
   # Assuming you want to use the same path for exporting the data
   export_bids_path = bids_path
   # Some settings regarding your BIDS dataset
   subject = "01"
   session = "01"
   task = "eegfmriNF"

   # Now some settings for the AAS
   window_size = 30
   relative_window_position = -0.5   
   artifact_to_trigger_offset_in_seconds = -0.005
   event_id_description_pairs={'trigger':1}
   unwanted_bad_channels = ['EKG', 'EMG', 'EOG', 'ECG'] # Channels with these names are considered as bad channels and not considered in the processing
   evaluation_measures=["SNR", "RMS", "RMS2", "MEDIAN"] # Evaluation measures to be used for the evaluation of the AAS
   # End Configuration Block

   # Loading the EEG data by creating a facet object and importing the EEG data
   f = facet()
   f.import_eeg(path=bids_path,fmt="bids",upsampling_factor=upsampling_factor,artifact_to_trigger_offset=artifact_to_trigger_offset_in_seconds, bads=unwanted_bad_channels, subject=subject, session=session,task=task)

.. important::

   Please make sure that the path to your BIDS dataset is correct and that the dataset is structured according to the BIDS standard.
   Also, make sure that you have enough memory available to load the dataset and conside that the upscaled dataset will require more memory. So choose the upsampling factor wisely.

Preprocessing
-------------
If you have not already preprocessed your EEG data, you may want to perform some preprocessing steps before applying AAS. 
This can include filtering and resampling.

.. note::

   The stock facet object provides a `pre_processing` method, which is a general predefined collection of preprocessing steps. 
   This can and should be individualized to the user's needs.
   `f.pre_processing()`

.. code-block:: python

   f.highpass(1)
   f.upsample() # upsampling factor must be specified when importing the EEG data

Finding Triggers
----------------

Before applying AAS, you need to specify which events in your EEG data will be used as triggers.
This is done using the `find_triggers` method, which takes an event Regular Expression as an argument.
The event regex is a Regular Expression that matches the events you want to use as triggers.

`find_triggers` automatically detects if the Dataset contains Annotations or a Stim Channel and uses the provided regex to find the triggers.
If your Dataset contains Annotations, the provided Regular Expression should contain the annotation description you want to use as triggers.
If your Dataset contains a Stim Channel (Channel that contains event information), the provided regex should contain the event_id you want to use as triggers. e.g. \b1\b for event_id 1.

.. code-block:: python

   f.find_triggers(event_regex)
   f.find_missing_triggers()

.. note::

   If there are triggers missing, you can either add them with the `add_trigger` method or detect them automatically with the `find_missing_triggers` method.

Aligning Triggers
-----------------

Before applying AAS, you need to align the triggers so they match the Slice Gradient artifact.

.. code-block:: python

   reference_trigger = 0
   f.align_triggers(reference_trigger)

Applying Averaged Artifact Subtraction
--------------------------------------

After preprocessing your EEG data and aligning the triggers, you can apply Averaged Artifact Subtraction (AAS) to remove artifacts from the EEG data.
AAS includes the following steps:

Calculating Averaged Artifact Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you can remove artifacts, you must calculate the average artifact matrix. This can be done 
after your triggers are loaded.

.. code-block:: python

   f.calc_matrix_aas()

.. important::

   This only calculates the average artifact matrix. To calculate and remove the average artifact from the EEG data, you must call the `f.remove_artifacts` method.
   If you need the calculated average artifact itself (not the matrix), you can calculate it with `f.get_correction().calc_avg_artifact()`. But this is automatically done when calling `f.remove_artifacts()`.

Removing Artifacts
^^^^^^^^^^^^^^^^^^

After calculating the average artifact, you can remove the average artifact from the EEG data:

.. code-block:: python

   f.remove_artifacts()

With that, the AAS correction is done. You can now proceed with further processing of the EEG data.

Further Processing
------------------

After removing artifacts, you can proceed with further EEG data processing, such as filtering, adaptive noise cancellation, and downsampling.

.. note::

   The stock facet object provides a `post_processing` method, which is a general predefined collection of postprocessing steps. 
   Again, this can and should be individualized to the user's needs.
   `f.post_processing()`

.. code-block:: python

   # Example: Applying a low-pass filter
   f.downsample() # downsampling by upsample factor
   f.lowpass(70)
   f.apply_ANC() # apply the ANC to the EEG data. This may take some time. If you want keep track of the progress, you can set the logger level to DEBUG

Plotting the Processed EEG Data
-------------------------------
If you want to visualize the processed EEG data, you can use the `plot_eeg` method.

.. code-block:: python

   f.plot_eeg()

Evaluating the Processed EEG Data
---------------------------------
If you want to evaluate the processed EEG data, you can add the EEG data to the evaluation framework and call the `evaluate` method.

.. code-block:: python

   f.add_to_evaluate(f.get_eeg(), name="Corrected EEG")
   results = f.evaluate(plot=true, measures=evaluation_measures)
   print(results) # Print the evaluation results if you want to see detailed figures

Exporting the Processed EEG Data
--------------------------------
After processing your EEG data, you may want to export the processed data to a file.
This can be done using the `export_eeg` method, which takes the file path as an argument.

.. code-block:: python

   f.export_eeg(event_id=event_id_description_pairs, path=export_bids_path, fmt="bids")

Conclusion
----------

Applying Averaged Artifact Subtraction (AAS) is crucial for preparing EEG data for analysis by reducing noise and artifacts. This documentation outlined the steps to apply AAS using facet, from loading your EEG data to applying the AAS correction.

For more detailed information on processing EEG data with MNE-Python, refer to the official MNE-Python documentation.