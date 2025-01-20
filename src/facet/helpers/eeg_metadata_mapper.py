import numpy as np

def eeg_to_dict(eeg):
    """
    Maps EEG object attributes to a dictionary for storage in mne.info['facet'].
    
    Parameters:
        eeg (facet.eeg_obj.EEG): The EEG object to map
        
    Returns:
        dict: Dictionary containing the EEG metadata
    """
    metadata = {
        'artifact_to_trigger_offset': eeg.artifact_to_trigger_offset, # The position of the artifact relative to the trigger
        'last_trigger_search_regex': eeg.last_trigger_search_regex, # The regex used to search for triggers
        'upsampling_factor': eeg.upsampling_factor, # The factor by which the data is upsampled
        'time_first_artifact_start': eeg.time_first_artifact_start, # The start time of the first artifact
        'time_last_artifact_end': eeg.time_last_artifact_end, # The end time of the last artifact
        'data_time_start': eeg.data_time_start, # The start time of the data. 
        'data_time_end': eeg.data_time_end, # The end time of the data. 
        'artifact_length': eeg.artifact_length,  # The length of the artifact in samples
        'anc_hp_frequency': eeg.anc_hp_frequency, # The high pass frequency of the ANC
        'obs_hp_frequency': eeg.obs_hp_frequency, # The high pass frequency of the optimal basis set
        'obs_exclude_channels': eeg.obs_exclude_channels, # The channels to exclude from the optimal basis set
        'time_acq_padding_left': eeg.time_acq_padding_left, # The padding time before the acquisition
        'time_acq_padding_right': eeg.time_acq_padding_right, # The padding time after the acquisition
        '_tmin': eeg._tmin, # Time relative to the trigger marking the start of the artifact
        '_tmax': eeg._tmax # Time relative to the trigger marking the end of the artifact
    }
    
    # Convert numpy arrays and other non-serializable types to lists
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            metadata[key] = value.tolist()
            
    return metadata

def dict_to_eeg(metadata_dict, eeg):
    """
    Maps dictionary from mne.info['facet'] back to EEG object attributes.
    
    Parameters:
        metadata_dict (dict): Dictionary containing the EEG metadata
        eeg (facet.eeg_obj.EEG): The EEG object to update
        
    Returns:
        facet.eeg_obj.EEG: Updated EEG object
    """
    for key, value in metadata_dict.items():
        if hasattr(eeg, key):
            # Convert lists back to numpy arrays if needed
            if isinstance(value, list) and key in ['obs_exclude_channels']:
                value = np.array(value)
            setattr(eeg, key, value)
            
    return eeg



