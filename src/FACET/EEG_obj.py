import numpy as np
from copy import deepcopy

class EEG:
    mne_raw: None #The MNE raw object storing the EEG data
    mne_raw_orig: None #Untouched MNE raw object storing the EEG data
    estimated_noise = None #The estimated noise of the EEG data used for the ANC
    _tmin= None
    _tmax= None
    artifact_to_trigger_offset: None
    loaded_triggers= None
    last_trigger_search_regex=None
    all_events= None
    triggers_as_events= None
    count_triggers= None
    upsampling_factor= None
    time_first_artifact_start= None
    time_last_artifact_end= None
    data_time_start= None
    data_time_end= None
    artifact_length= None
    artifact_duration= None
    volume_gaps= None
    BIDSPath= None

    #calculations
    anc_hp_frequency = None #The highpass frequency of the ANC
    anc_hp_filter_weights = None #The filter weights of the ANC
    anc_filter_order = None #The filter order of the ANC

    def __init__(self, mne_raw=None, mne_raw_orig=None, anc_hp_frequency=None, estimated_noise=None, artifact_to_trigger_offset=0.0, loaded_triggers=None, last_trigger_search_regex=None, all_events=None, triggers_as_events=None, count_triggers=None, upsampling_factor=None, time_first_artifact_start=None, time_last_artifact_end=None, data_time_start=None, data_time_end=None, artifact_length=0, artifact_duration=0, volume_gaps=None, BIDSPath=None):
        self.mne_raw = mne_raw
        self.mne_raw_orig = mne_raw_orig if mne_raw_orig is not None else mne_raw.copy() if mne_raw is not None else None
        self.anc_hp_frequency = anc_hp_frequency
        self.estimated_noise = estimated_noise
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.loaded_triggers = loaded_triggers
        self.last_trigger_search_regex = last_trigger_search_regex
        self.all_events = all_events
        self.triggers_as_events = triggers_as_events
        self.count_triggers = count_triggers
        self.upsampling_factor = upsampling_factor
        self.time_first_artifact_start = time_first_artifact_start
        self.time_last_artifact_end = time_last_artifact_end
        self.data_time_start = data_time_start
        self.data_time_end = data_time_end
        self.artifact_length = artifact_length
        self.artifact_duration = artifact_duration
        self.volume_gaps = volume_gaps
        self.BIDSPath = BIDSPath

        #calculations
        self._tmin = self.artifact_to_trigger_offset
        self._tmax = self.artifact_to_trigger_offset + self.artifact_duration

    def get_tmin(self):
        return self._tmin
    def get_tmax(self):
        return self._tmax
    def copy(self):
        copied = deepcopy(self)
        copied.mne_raw = self.mne_raw.copy()
        copied.mne_raw_orig = self.mne_raw_orig.copy()
        return copied