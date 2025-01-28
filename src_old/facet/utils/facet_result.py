from facet.helpers.eeg_metadata_mapper import eeg_to_dict

import mne
import numpy as np


class FACETResult:
    """
    Class to store the results of FACET processing.
    Contains the original, processed and noise EEG data.
    """
    def __init__(self):
        """Initialize empty FACET result object."""
        self._mne_orig = None
        self._mne_processed = None 
        self._mne_noise = None

    @property
    def mne_orig(self):
        """The original unprocessed MNE Raw object."""
        return self._mne_orig

    @mne_orig.setter 
    def mne_orig(self, value):
        self._mne_orig = value

    @property
    def mne_processed(self):
        """The processed MNE Raw object."""
        return self._mne_processed

    @mne_processed.setter
    def mne_processed(self, value):
        self._mne_processed = value

    @property
    def mne_noise(self):
        """The estimated noise MNE Raw object."""
        return self._mne_noise

    @mne_noise.setter
    def mne_noise(self, value):
        self._mne_noise = value

    def get_metadata(self, name, mne_raw=None):
        """
        Get metadata from the specified MNE Raw object's info dictionary.

        Parameters
        ----------
        name : str
            Name of the metadata field to retrieve
        mne_raw : mne.io.Raw, optional
            MNE Raw object to get metadata from.
            If not provided, uses the processed data (self.mne_processed)

        Returns
        -------
        any
            The metadata value if found, None if not found

        Raises
        ------
        AttributeError
            If no MNE Raw object exists to get metadata from
        KeyError 
            If the metadata field does not exist
        """
        
        raw = mne_raw if mne_raw is not None else self.mne_processed
        if raw is None:
            raise AttributeError("No MNE Raw object exists to get metadata from")
            
        try:
            return raw.info['temp']['facet'][name]
        except KeyError:
            raise KeyError(f"Metadata field '{name}' not found")

    @classmethod
    def from_facet_object(cls, facet_obj):
        """
        Create a FACETResult from a FACET object.

        Parameters
        ----------
        facet_obj : FACET
            The FACET object to create the result from.

        Returns
        -------
        FACETResult
            The result object containing the processed data.
        """
        result = cls()
        
        # Store the processed raw data
        result.mne_processed = facet_obj._eeg.mne_raw.copy()
        
        # Store the original raw data
        result.mne_original = facet_obj._eeg.mne_raw_orig.copy()
        
        # Map EEG metadata to the info dictionary
        metadata = eeg_to_dict(facet_obj._eeg)
        
        # Store metadata in the info dictionary under 'facet' key
        if 'temp' not in result.mne_processed.info:
            result.mne_processed.info['temp'] = {}
        result.mne_processed.info['temp']['facet'] = metadata
        
        if 'temp' not in result.mne_original.info:
            result.mne_original.info['temp'] = {}
        result.mne_original.info['temp']['facet'] = metadata

        # Generate the noise raw object
        result.mne_noise = cls._generate_noise_raw(cls,facet_obj)
        
        return result
    
    def _generate_noise_raw(self, facet_obj):
        """
        Generate the noise raw object from the FACET object.

        Creates a new MNE Raw object from scratch with the noise data
        to ensure clean metadata and avoid potential side effects from copying.

        Parameters
        ----------
        facet_obj : FACET
            The FACET object containing the estimated noise data

        Returns
        -------
        mne.io.Raw
            New MNE Raw object containing the noise data
        """


        info = mne.create_info(
            ch_names=facet_obj._eeg.mne_raw.ch_names,
            sfreq=facet_obj._eeg.mne_raw.info['sfreq'],
            ch_types=['eeg'] * len(facet_obj._eeg.mne_raw.ch_names)
        )
        noise_data = np.vstack(facet_obj._eeg.estimated_noise)
        noise_raw = mne.io.RawArray(noise_data, info)

        if 'temp' not in noise_raw.info:
            noise_raw.info['temp'] = {}
        noise_raw.info['temp']['facet'] = eeg_to_dict(facet_obj._eeg)

        return noise_raw