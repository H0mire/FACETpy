import numpy as np
import mne, re
from scipy.stats import pearsonr

# import inst for mne python


class Correction_Framework:
    def __init__(self, Rel_Trig_Pos, Upsample):
        self._rel_trigger_pos = Rel_Trig_Pos
        self._triggers = None
        self._num_triggers = None
        self._upsample = Upsample
        self._plot_number = 0

        self._eeg = {
            "raw": None,
            "raw_orig": None,
            "tmin": None,
            "tmax": None,
            "rel_trigger_pos": Rel_Trig_Pos,
            "triggers": None,
            "num_triggers": None,
            "upsample": Upsample,
            "time_triggers_start": None,
            "time_triggers_end": None,
            "time_start": None,
            "time_end": None,
            "art_length": None,
            "duration_art": None,
            "volume_gaps": None,
        }

    def import_EEG(self, filename):
        raw = mne.io.read_raw_edf(filename)
        raw.load_data()
        raw_orig = raw.copy()
        time_start = raw.times[0]
        time_end = raw.times[-1]
        self._eeg = {
            "raw": None,
            "raw_orig": raw_orig,
            "tmin": None,
            "tmax": None,
            "rel_trigger_pos": self._rel_trigger_pos,
            "triggers": None,
            "num_triggers": None,
            "upsample": self._upsample,
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

    def import_EEG_GDF(self, filename):
        raw = mne.io.read_raw_gdf(filename)
        raw.load_data()
        raw_orig = raw.copy()
        time_start = raw.times[0]
        time_end = raw.times[-1]
        self._eeg = {
            "raw": None,
            "raw_orig": raw_orig,
            "tmin": None,
            "tmax": None,
            "rel_trigger_pos": self._rel_trigger_pos,
            "triggers": None,
            "num_triggers": None,
            "upsample": self._upsample,
            "time_triggers_start": None,
            "time_triggers_end": None,
            "time_start": time_start,
            "time_end": time_end,
            "art_length": None,
            "duration_art": None,
            "volume_gaps": None,
        }

        print(filename)

    def export_EEG(self, filename):
        self._eeg["raw"].export(filename, fmt="edf", overwrite=True)

    def find_triggers(self, regex):
        # self._raw.add_events(mne.events_from_annotations(self._raw))
        # print(self._filterAnnotations(regex))

        annotations = self._filter_annotations(regex)
        positions = []
        for onset, duration, description in annotations:
            print(f"Onset: {onset}, Duration: {duration}, Description: {description}")
            positions.append(onset)

        triggers = positions
        num_triggers = len(positions)
        time_triggers_start = self._raw.times[self._triggers[0]]
        time_triggers_end = self._raw.times[self._triggers[-1]]
        self._eeg["triggers"] = triggers
        self._eeg["num_triggers"] = num_triggers
        self._eeg["time_triggers_start"] = time_triggers_start
        self._eeg["time_triggers_end"] = time_triggers_end
        self._derive_art_length()

        print(positions)

    def cut(self):
        self._raw.crop(
            tmin=self._time_triggers_start,
            tmax=min(
                self._time_end,
                self._time_triggers_end
                + (
                    self._time_triggers_end
                    - self._raw.times[self._triggers[len(self._triggers) - 2]]
                ),
            ),
        )
        return

    # TODO: Implement better Structure
    def get_mne_raw(self):
        return self._eeg["raw"]

    def get_mne_raw_orig(self):
        return self._eeg["raw_orig"]

    def find_triggers_with_events(self, regex, idx=0):
        print(self._raw.ch_names)
        events = mne.find_events(self._raw, stim_channel="Status", initial_event=True)
        pattern = re.compile(regex)

        filtered_events = [event for event in events if pattern.search(str(event[2]))]
        filtered_positions = [event[idx] for event in filtered_events]
        _events = filtered_events
        triggers = filtered_positions
        num_triggers = len(filtered_positions)
        time_triggers_start = self._raw.times[self._triggers[0]]
        time_triggers_end = self._raw.times[self._triggers[-1]]
        self._eeg["triggers"] = triggers
        self._eeg["num_triggers"] = num_triggers
        self._eeg["time_triggers_start"] = time_triggers_start
        self._eeg["time_triggers_end"] = time_triggers_end
        self._derive_art_length()

    def prepare(self):
        self._upsample_data()

    def plot_EEG(self):
        self._plot_number += 1
        self._raw.plot(title=str(self._plot_number), start=27)


    def _derive_art_length(self):
        d = np.diff(self._triggers)  # trigger distances

        if self._volume_gaps:
            m = np.mean([np.min(d), np.max(d)])  # middle distance
            ds = d[d < m]  # trigger distances belonging to slice triggers
            # dv = d[d > m]  # trigger distances belonging to volume triggers

            # total length of an artifact
            self._art_length = np.max(ds)  # use max to avoid gaps between slices
        else:
            # total length of an artifact
            self._art_length = np.max(d)
            self._duration_art = self._art_length / self._raw.info["sfreq"]

    def _filter_annotations(self, regex):
        """Extract specific annotations from an MNE Raw object."""
        raw = self._raw
        # initialize list to store results
        specific_annotations = []

        # compile the regular regex pattern
        pattern = re.compile(regex)

        # loop through each annotation in the raw object
        for annot in raw.annotations:
            # check if the annotation description matches the pattern
            if pattern.search(annot["description"]):
                # if it does, append the annotation (time, duration, description) to our results list
                specific_annotations.append(
                    (annot["onset"], annot["duration"], annot["description"])
                )

        return specific_annotations
