Analytics Framework Module
==========================

.. module:: Analytics
   :synopsis: Provides methods for importing, exporting, and analyzing EEG data.

.. moduleauthor:: Janik Michael Müller <email@example.com>

Version: 1.0
Date: 15.02.2024

.. currentmodule:: FACET.Frameworks.Analytics

.. autoclass:: Analytics_Framework
   :members:
   :undoc-members:
   :show-inheritance:

   The ``Analytics_Framework`` class is designed to analyise the eeg dataset. It offers functionalities for importing EEG data from different formats, exporting data in the BIDS format, finding triggers in the EEG signals, and other analysis tasks.

   Constructor
   -----------

   .. automethod:: __init__

   Methods
   -------

   .. automethod:: export_as_bids
   .. automethod:: import_from_bids
   .. automethod:: import_EEG
   .. automethod:: export_EEG
   .. automethod:: find_triggers
   .. automethod:: get_mne_raw
   .. automethod:: get_mne_raw_orig
   .. automethod:: get_eeg
   .. automethod:: plot_EEG
   .. automethod:: print_analytics

   Private Methods
   ---------------

   These methods are intended for internal use within the class to support its public methods.

   .. automethod:: _try_to_get_events
   .. automethod:: _derive_art_length
   .. automethod:: _derive_anc_hp_params
   .. automethod:: _filter_annotations
