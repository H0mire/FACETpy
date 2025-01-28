import numpy as np
from copy import deepcopy


class EEG:
    mne_raw: None  # The MNE raw object storing the EEG data
    mne_raw_orig: None  # Untouched MNE raw object storing the EEG data
    estimated_noise = None  # The estimated noise of the EEG data used for the ANC
    _tmin = None
    _tmax = None
    artifact_to_trigger_offset: None
    loaded_triggers = None
    last_trigger_search_regex = None
    all_events = None
    upsampling_factor = None
    time_first_artifact_start = None
    time_last_artifact_end = None
    data_time_start = None
    data_time_end = None
    artifact_length = None
    volume_gaps = None
    slice_triggers = None
    BIDSPath = None
    acq_padding_left = None
    acq_padding_right = None

    # calculations
    anc_hp_frequency = None  # The highpass frequency of the ANC
    anc_hp_filter_weights = None  # The filter weights of the ANC
    anc_filter_order = None  # The filter order of the ANC
    ssa_hp_frequency = None  # The highpass frequency of the SSA
    obs_hp_frequency = 70  # The highpass frequency of the OBS/PCA
    obs_hp_filter_weights = None  # The filter weights of the ANC
    obs_exclude_channels = []

    def __init__(
        self,
        mne_raw=None,
        mne_raw_orig=None,
        anc_hp_frequency=None,
        estimated_noise=None,
        artifact_to_trigger_offset=0.0,
        loaded_triggers=None,
        last_trigger_search_regex=None,
        all_events=None,
        upsampling_factor=None,
        time_first_artifact_start=None,
        time_last_artifact_end=None,
        data_time_start=None,
        data_time_end=None,
        artifact_length=0,
        volume_gaps=None,
        BIDSPath=None,
        obs_hp_frequency=70,
    ):
        self.mne_raw = mne_raw
        self.mne_raw_orig = (
            mne_raw_orig
            if mne_raw_orig is not None
            else mne_raw.copy() if mne_raw is not None else None
        )
        self.anc_hp_frequency = anc_hp_frequency
        self.estimated_noise = estimated_noise
        self.artifact_to_trigger_offset = artifact_to_trigger_offset
        self.loaded_triggers = loaded_triggers
        self.last_trigger_search_regex = last_trigger_search_regex
        self.all_events = all_events
        self.upsampling_factor = upsampling_factor
        self.time_first_artifact_start = time_first_artifact_start
        self.time_last_artifact_end = time_last_artifact_end
        self.data_time_start = data_time_start
        self.data_time_end = data_time_end
        self.artifact_length = artifact_length
        self.volume_gaps = volume_gaps
        self.BIDSPath = BIDSPath
        self.obs_hp_frequency = obs_hp_frequency

        # calculations
        self._tmin = self.artifact_to_trigger_offset
        self._tmax = self.artifact_to_trigger_offset + self.artifact_duration
        self.time_acq_padding_left = self.artifact_duration
        self.time_acq_padding_right = self.artifact_duration

        # private attributes
        self._loaded_triggers_upsampled = None

    @property
    def triggers_as_events(self):
        if self.loaded_triggers is None:
            return None

        events = []
        for trigger in self.loaded_triggers:
            events.append((trigger, 0, 1))

        return np.array(events, dtype=np.int32)

    @property
    def count_triggers(self):
        if self.loaded_triggers is None:
            return 0
        return len(self.loaded_triggers)

    @property
    def smin(self):
        return int(np.ceil(self._tmin * self.mne_raw.info["sfreq"]))

    @property
    def smax(self):
        return self.smin + self.artifact_length

    @property
    def tmin(self):
        return self._tmin

    @property
    def tmax(self):
        return self._tmax

    @property
    def s_acq_padding_left(self):
        return int(np.ceil(self.time_acq_padding_left * self.mne_raw.info["sfreq"]))

    @property
    def s_acq_padding_right(self):
        return int(np.ceil(self.time_acq_padding_right * self.mne_raw.info["sfreq"]))

    @property
    def s_first_artifact_start(self):
        return int(np.ceil(self.time_first_artifact_start * self.mne_raw.info["sfreq"]))

    @property
    def s_last_artifact_end(self):
        return int(np.ceil(self.time_last_artifact_end * self.mne_raw.info["sfreq"]))

    @property
    def time_acq_start(self):
        return np.max([0, self.time_first_artifact_start - self.time_acq_padding_left])

    @property
    def time_acq_end(self):
        return np.min(
            [
                self.mne_raw.times[-1],
                self.time_last_artifact_end + self.time_acq_padding_right,
            ]
        )

    @property
    def s_acq_start(self):
        return np.max([0, self.s_first_artifact_start - self.s_acq_padding_left])

    @property
    def s_acq_end(self):
        return np.min(
            [self.mne_raw.n_times, self.s_last_artifact_end + self.s_acq_padding_right]
        )

    @property
    def artifact_duration(self):
        if (self.mne_raw is None) or (self.mne_raw.info["sfreq"] is None):
            return -1
        return self.artifact_length / (self.mne_raw.info["sfreq"] or 1)

    def copy(self):
        copied = deepcopy(self)
        copied.mne_raw = self.mne_raw.copy()
        copied.mne_raw_orig = self.mne_raw_orig.copy()
        return copied
