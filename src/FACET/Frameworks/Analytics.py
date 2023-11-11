import numpy as np
import mne, re
from scipy.stats import pearsonr

# import inst for mne python


class Analytics_Framework:
    def __init__(self):
        self._triggers = None
        self._num_triggers = None
        self._plot_number = 0

        self._eeg = {
            "raw": None, # MNE Raw Object
            "raw_orig": None, # MNE Raw Object unimpaired
            "tmin": None, # Relative Start Time of Artifact. Often equal to rel_trigger_pos. Often 0
            "tmax": None, # Relative End Time of Artifact. Often equal to tmin + duration_art
            "rel_trigger_pos": 0, # Relative Trigger Position. Often 0. Means that the trigger is at the beginning of the artifact
            "triggers": None, # Trigger Positions in Samples
            "events": None, # Mne Events 
            "num_triggers": None, # Number of Triggers
            "upsampling_factor": None, # Upsampling Factor
            "time_triggers_start": None, # Time of first Trigger. The start of the first slice of e. g. the fMRI Scan
            "time_triggers_end": None, # Time of last Trigger.  The start of the last slice of e.g. the fMRI Scan
            "time_start": None, # Often 0. Time of the first sample
            "time_end": None, # Time of the last sample
            "art_length": None, # Length of the artifact in samples
            "duration_art": None, # Length of the artifact in seconds
            "volume_gaps": None, # True if there are gaps between the slices of the fMRI Scan
        }

    def import_EEG(self, filename, rel_trig_pos=0, upsampling_factor=10):
        raw = mne.io.read_raw_edf(filename)
        raw.load_data()
        raw_orig = raw.copy()
        time_start = raw.times[0]
        time_end = raw.times[-1]
        self._eeg = {
            "raw": raw,
            "raw_orig": raw_orig,
            "tmin": None,
            "tmax": None,
            "rel_trigger_pos": rel_trig_pos,
            "triggers": None,
            "events": None,
            "num_triggers": None,
            "upsampling_factor": upsampling_factor,
            "time_triggers_start": None,
            "time_triggers_end": None,
            "time_start": time_start,
            "time_end": time_end,
            "art_length": None,
            "duration_art": None,
            "volume_gaps": None,
        }
        print(filename)
        return self._eeg

    def import_EEG_GDF(self, filename, rel_trig_pos=0, upsampling_factor=10):
        raw = mne.io.read_raw_gdf(filename)
        raw.load_data()
        raw_orig = raw.copy()
        time_start = raw.times[0]
        time_end = raw.times[-1]
        self._eeg = {
            "raw": raw,
            "raw_orig": raw_orig,
            "tmin": None,
            "tmax": None,
            "rel_trigger_pos": rel_trig_pos,
            "triggers": None,
            "events": None,
            "num_triggers": None,
            "upsampling_factor": upsampling_factor,
            "time_triggers_start": None,
            "time_triggers_end": None,
            "time_start": time_start,
            "time_end": time_end,
            "art_length": None,
            "duration_art": None,
            "volume_gaps": None,
        }

        print(filename)
        return self._eeg

    def export_EEG(self, filename):
        raw = self._eeg["raw"]
        raw.export(filename, fmt="edf", overwrite=True)

    def find_triggers(self, regex):
        raw = self._eeg["raw"]
        #TODO: Filter events by filtered annotations
        events = mne.events_from_annotations(raw)
        raw.add_events(events)
        # print(self._filterAnnotations(regex))
        annotations = self._filter_annotations(regex)
        positions = []
        for onset, duration, description in annotations:
            print(f"Onset: {onset}, Duration: {duration}, Description: {description}")
            positions.append(onset)

        triggers = positions
        num_triggers = len(positions)
        time_triggers_start = raw.times[self._triggers[0]]
        time_triggers_end = raw.times[self._triggers[-1]]
        self._eeg["triggers"] = triggers
        self._eeg["events"] = events
        self._eeg["num_triggers"] = num_triggers
        self._eeg["time_triggers_start"] = time_triggers_start
        self._eeg["time_triggers_end"] = time_triggers_end
        self._eeg["volume_gaps"] = False
        self._derive_art_length()
        self._eeg["tmin"] = self._eeg["rel_trigger_pos"]
        self._eeg["tmax"] = self._eeg["tmin"] + self._eeg["duration_art"] 


    # TODO: Implement better Structure
    def get_mne_raw(self):
        return self._eeg["raw"]

    def get_mne_raw_orig(self):
        return self._eeg["raw_orig"]
    
    def get_eeg(self):
        return self._eeg

    def find_triggers_with_events(self, regex, idx=0):
        raw = self._eeg["raw"]
        print(raw.ch_names)
        events = mne.find_events(raw, stim_channel="Status", initial_event=True)
        pattern = re.compile(regex)

        filtered_events = [event for event in events if pattern.search(str(event[2]))]
        filtered_positions = [event[idx] for event in filtered_events]
        _events = filtered_events
        triggers = filtered_positions
        num_triggers = len(filtered_positions)
        time_triggers_start = raw.times[triggers[0]]
        time_triggers_end = raw.times[triggers[-1]]
        self._eeg["triggers"] = triggers
        self._eeg["events"] = _events
        self._eeg["num_triggers"] = num_triggers
        self._eeg["time_triggers_start"] = time_triggers_start - self._eeg["rel_trigger_pos"]
        self._eeg["time_triggers_end"] = time_triggers_end
        self._eeg["volume_gaps"] = False
        self._derive_art_length()
        self._eeg["tmin"] = self._eeg["rel_trigger_pos"]
        self._eeg["tmax"] = self._eeg["tmin"] + self._eeg["duration_art"]

    def prepare(self):
        self._upsample_data()

    def plot_EEG(self):
        self._plot_number += 1
        self._raw.plot(title=str(self._plot_number), start=27)


    def _derive_art_length(self):
        d = np.diff(self._eeg["triggers"])  # trigger distances

        if self._eeg["volume_gaps"]:
            m = np.mean([np.min(d), np.max(d)])  # middle distance
            ds = d[d < m]  # trigger distances belonging to slice triggers
            # dv = d[d > m]  # trigger distances belonging to volume triggers

            # total length of an artifact
            self._eeg["art_length"] = np.max(ds)  # use max to avoid gaps between slices
        else:
            # total length of an artifact
            self._eeg["art_length"] = np.max(d)
            self._eeg["duration_art"] = self._eeg["art_length"] / self._eeg["raw"].info["sfreq"]

    def _filter_annotations(self, regex):
        """Extract specific annotations from an MNE Raw object."""
        eeg = self._eeg
        # initialize list to store results
        specific_annotations = []

        # compile the regular regex pattern
        pattern = re.compile(regex)

        # loop through each annotation in the raw object
        for annot in eeg["raw"].annotations:
            # check if the annotation description matches the pattern
            if pattern.search(annot["description"]):
                # if it does, append the annotation (time, duration, description) to our results list
                specific_annotations.append(
                    (annot["onset"], annot["duration"], annot["description"])
                )

        return specific_annotations
