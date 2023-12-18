import numpy as np
import mne, re, os
from mne_bids import BIDSPath, write_raw_bids, read_raw_bids
from scipy.stats import pearsonr
import numpy as np

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

    # a method to export the eeg data as a bids project using mne_bids
    def export_as_bids(self, event_id): #TODO: Add BIDSPath as parameter. Clean up
        if "line_freq" not in self._eeg["raw"].info or self._eeg["raw"].info["line_freq"] is None:
            line_freq = input("Please enter the line frequency: ")
            self._eeg["raw"].info["line_freq"] = float(line_freq)
        if "line_freq" not in self._eeg["raw_orig"].info or self._eeg["raw_orig"].info["line_freq"] is None:
            line_freq = input("Please enter the line frequency: ")
            self._eeg["raw_orig"].info["line_freq"] = float(line_freq)
        
        BIDSPathuncorrected = BIDSPath(
            subject="subjectid", session="sessionid", task="uncorrected", root="./bids_dir"
        )
        BIDSPathcorrected = BIDSPath(
            subject="subjectid", session="sessionid", task="corrected", root="./bids_dir"
        )
        print("Exporting Channels: "+str(self._eeg["raw"].ch_names))

        raw = self._eeg["raw"].copy()
        raw_orig = self._eeg["raw_orig"].copy()
        #drop stim channels
        stim_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
        raw.drop_channels([raw.ch_names[ch] for ch in stim_channels])
        raw_orig.drop_channels([raw_orig.ch_names[ch] for ch in stim_channels])

        if self._eeg["raw_orig"] is not None:
            write_raw_bids(raw=raw, bids_path=BIDSPathuncorrected, overwrite=True, format="EDF", allow_preload=True, events=self._eeg["events"], event_id=event_id)
        if self._eeg["raw"] is not None:
            write_raw_bids(raw=raw_orig, bids_path=BIDSPathcorrected, overwrite=True, format="EDF", allow_preload=True, events=self._eeg["events"], event_id=event_id)
    
    def import_from_bids(self, bids_path="./bids_dir", rel_trig_pos=0, upsampling_factor=10, bads=[]):
        bids_path_i = BIDSPath(subject="subjectid", session="sessionid", task="corrected", root=bids_path)
        raw = read_raw_bids(bids_path_i)
        raw.load_data()
        all_channels = raw.ch_names
        exclude = [item for i, item in enumerate(all_channels) if item in bads]
        raw.info["bads"] = exclude
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
            "BIDSPath": bids_path
        }
        print("Importing EEG with:")
        print("Channels " + str(raw.ch_names))
        print(f"Time Start: {time_start}s")
        print(f"Time End: {time_end}s")
        print(f"Number of Samples: {raw.n_times}")
        print(f"Sampling Frequency: {raw.info['sfreq']}Hz")
        print(bids_path)
        return self._eeg



    def import_EEG(self, filename, rel_trig_pos=0, upsampling_factor=10, fmt="edf", bads=[]):
        if fmt == "edf":
            raw = mne.io.read_raw_edf(filename)
        elif fmt == "gdf":
            raw = mne.io.read_raw_gdf(filename)
        else:
            raise ValueError("Format not supported")
        raw.load_data()

        all_channels = raw.ch_names
        exclude = [item for i, item in enumerate(all_channels) if item in bads]
        raw.info["bads"] = exclude
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
        print("Importing EEG with:")
        print("Channels " + str(raw.ch_names))
        print(f"Time Start: {time_start}s")
        print(f"Time End: {time_end}s")
        print(f"Number of Samples: {raw.n_times}")
        print(f"Sampling Frequency: {raw.info['sfreq']}Hz")
        print(filename)
        return self._eeg

    def export_EEG(self, filename):
        raw = self._eeg["raw"]
        raw.export(filename, fmt="edf", overwrite=True)

    def find_triggers(self, regex, idx=0):
        raw = self._eeg["raw"]
        stim_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
        events=[]
        filtered_events=[]
        

        if len(stim_channels) > 0:
            print("Stim-Kanäle gefunden:", [raw.ch_names[ch] for ch in stim_channels])
            events = mne.find_events(raw, stim_channel=raw.ch_names[stim_channels[0]], initial_event=True)
            pattern = re.compile(regex)
            filtered_events = [event for event in events if pattern.search(str(event[2]))]

        else:
            print("No Stim-Channels found.")
            print()
            events_obj = mne.events_from_annotations(raw)
            print(events_obj[1])
            filtered_events = mne.events_from_annotations(raw, regexp=regex)[0]
        
        if len(filtered_events)==0:
            print("No events found.")
            return
        filtered_positions = [event[idx] for event in filtered_events]
        filtered_events = np.array(filtered_events)
        triggers = filtered_positions
        num_triggers = len(filtered_positions)
        time_triggers_start = raw.times[triggers[0]]
        time_triggers_end = raw.times[triggers[-1]]
        self._eeg["triggers"] = triggers
        self._eeg["events"] = filtered_events
        self._eeg["num_triggers"] = num_triggers
        self._eeg["time_triggers_start"] = time_triggers_start - self._eeg["rel_trigger_pos"]
        self._eeg["time_triggers_end"] = time_triggers_end
        self._eeg["volume_gaps"] = False
        self._derive_art_length()
        self._eeg["tmin"] = self._eeg["rel_trigger_pos"]
        self._eeg["tmax"] = self._eeg["tmin"] + self._eeg["duration_art"]
        print("Channels after find trigger: "+str(self._eeg["raw"].ch_names))


    # TODO: Implement better Structure
    def get_mne_raw(self):
        return self._eeg["raw"]

    def get_mne_raw_orig(self):
        return self._eeg["raw_orig"]
    
    def get_eeg(self):
        return self._eeg

    def prepare(self):
        self._upsample_data()

    def plot_EEG(self, start = 0):
        self._plot_number += 1
        self._raw.plot(title=str(self._plot_number), start=start)


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
