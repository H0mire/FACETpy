import numpy as np
import mne, re
from scipy.stats import pearsonr

# import inst for mne python


class FacetController:
    def __init__(
        self,
        name,
        RelTrigPos,
        Upsample,
        AvgWindow,
        SliceTriggers,
        UpsampleCutoff,
        InterpolateVolumeGaps,
        OBSExcludeChannels,
    ):
        self._name = name
        self._rel_trigger_pos = RelTrigPos
        self._upsample = Upsample
        self._avg_window = AvgWindow
        self._slice_triggers = SliceTriggers
        self._upsample_cutoff = UpsampleCutoff
        self._interpolate_volume_gaps = InterpolateVolumeGaps
        self._obs_exclude_channels = OBSExcludeChannels
        self._plot_number = 0
        self._raw=0

    def import_EEG(self, filename):
        self._raw = mne.io.read_raw_edf(filename)
        self._raw.load_data()
        print(filename)

    def import_EEG_GDF(self, filename):
        self._raw = mne.io.read_raw_gdf(filename)
        self._raw.load_data()
        print(filename)

    def export_EEG(self, filename):
        self._raw.export(filename, fmt="edf", overwrite=True)

    def find_triggers(self, regex):
        # self._raw.add_events(mne.events_from_annotations(self._raw))
        # print(self._filterAnnotations(regex))

        annotations = self._filter_annotations(regex)
        positions = []
        for onset, duration, description in annotations:
            print(f"Onset: {onset}, Duration: {duration}, Description: {description}")
            positions.append(onset)

        self._triggers = positions
        self._num_triggers = len(positions)
        print(positions)

    def cut(self):
        self._raw.crop(
            tmin=self._raw.times[self._triggers[0]],
            tmax=min(
                self._raw.times[-1],
                self._raw.times[self._triggers[-1]]
                + (
                    self._raw.times[self._triggers[-1]]
                    - self._raw.times[self._triggers[len(self._triggers) - 1]]
                ),
            ),
        )
        return

    def find_triggers_with_events(self, regex, idx=0):
        print(self._raw.ch_names)
        events = mne.find_events(self._raw, stim_channel="Status", initial_event=True)
        pattern = re.compile(regex)

        filtered_events = [event for event in events if pattern.search(str(event[2]))]
        filtered_positions = [event[idx] for event in filtered_events]
        self._events = filtered_events
        self._triggers = filtered_positions
        self._num_triggers = len(filtered_events)

    def prepare(self):
        self._upsample_data()

    def plot_EEG(self):
        self._plot_number += 1
        self._raw.plot(title=str(self._plot_number), start=27)

    # Remove Artifacts from EEG
    def apply_MNE_AAS_old(self):
        raw = self._raw

        # get max of time difference between triggers while triggers have sample number and time is self._raw.times
        maxDiff = np.diff(self._raw.times[self._triggers]).max()
        print(maxDiff)
        # Schritt 1: Epochen erstellen
        tmin, tmax = 0, maxDiff
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False)
        epochs = mne.Epochs(
            raw,
            self._events,
            picks=picks,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            reject=None,
            preload=True,
        )

        # Schritt 2: Durchschnittlichen Artefakt berechnen
        evoked = epochs.average()

        # Schritt 4: Subtraktion
        raw_ssp = raw.copy().add_proj(mne.compute_proj_evoked(evoked))
        raw_ssp.apply_proj()
        self._raw = raw_ssp

    def compute_avg_correlation(self, epochs, channel_index):
        n_epochs = len(epochs)
        correlations = []

        # Berechne die Korrelation für jedes Paar von Epochen
        for i in range(n_epochs):
            for j in range(i + 1, n_epochs):
                corr, _ = pearsonr(epochs[i][channel_index], epochs[j][channel_index])
                correlations.append(corr)

        avg_corr = np.mean(correlations)
        return avg_corr

    def apply_MNE_AAS(self):
        events = self._events
        raw = (
            self._raw.copy()
        )  # Erstelle eine Kopie hier, um das Original unverändert zu lassen
        event_times = [event[0] for event in events]
        times = [raw.times[time] for time in event_times]
        diffs = np.diff(times)
        maxDiff = diffs.max()
        print(maxDiff)
        tmin = -0.01
        tmax = maxDiff -0.01

        print("tmin: ", tmin)
        print("tmax: ", tmax)

        # Schritt 1: Kanalauswahl
        raw.info["bads"] = ["Status", "EMG", "ECG"]
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
        raw.pick_channels(channels_to_keep)  # raw wird in-place modifiziert

        # Schritt 2: Epochen erstellen
        epochs = mne.Epochs(
            raw,
            events,
            picks=eeg_channels[:],
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            reject=None,
            preload=True,
        )

        print("Epochs shape: ", epochs.get_data().shape)

        good_epochs = self.highly_correlated_epochs(epochs)
        # Schritt 3: Durchschnittlichen Artefakt berechnen
        # evoked = epochs.average()
        evoked = epochs[good_epochs].average()
        #mne plot evoked
        evoked.plot()
        # Schritt 4: Subtraktion nur in den Bereichen um den Event
        corrected_data = raw._data.copy()
        for event in epochs.events:
            start, stop = raw.time_as_index([tmin, tmax], use_rounding=True)
            start += event[0]
            minColumn = evoked.data.shape[1]
            stop = start + minColumn
            corrected_data[:, start:stop] -= evoked.data

        raw._data = corrected_data

        self._raw = raw

    def highly_correlated_epochs(self, epochs, threshold=0.975):
        """Return list of epochs that are highly correlated to the average."""
        n_epochs = len(epochs)

        # Start by including the first five epochs
        chosen = list(range(5))
        sum_data = np.sum(epochs._data[chosen], axis=0)

        # Check subsequent epochs
        for idx in range(5, n_epochs):
            avg_data = sum_data / len(chosen)
            corr = np.corrcoef(
                avg_data.mean(axis=1).squeeze(),
                epochs._data[idx].mean(axis=1).squeeze(),
            )[0, 1]
            if corr > threshold:
                sum_data += epochs._data[idx]
                chosen.append(idx)

        return chosen

    def apply_MNE_AAS_slow(self, WINDOW_SIZE=30):
        events = self._events
        raw = (
            self._raw.copy()
        )  # Erstelle eine Kopie, um das Original unverändert zu lassen
        event_times = [event[0] for event in events]
        times = [raw.times[time] for time in event_times]
        diffs = np.diff(times)
        maxDiff = diffs.max()
        tmin = 0
        tmax = maxDiff+0.2

        # Kanalauswahl
        raw.info["bads"] = ["Status", "EMG", "ECG"]
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )

        # Epochen erstellen
        epochs = mne.Epochs(
            raw,
            events,
            picks=eeg_channels,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            reject=None,
            preload=True,
        )

        n_channels = len(eeg_channels)
        good_epochs_per_channel = []
        original_epochs = epochs.copy()
        for ch_name in epochs.ch_names:
            epochs_single_channel = original_epochs.copy().pick_channels([ch_name])
            good_epochs = self.highly_correlated_epochs_new(epochs_single_channel)
            good_epochs_per_channel.append(good_epochs)

        corrected_data = raw._data.copy()

        overall_mean = raw._data.mean(axis=1, keepdims=True)
        for ch_idx, good_epochs in enumerate(good_epochs_per_channel):
            print("Channel: ", (ch_idx + 1), "/", n_channels)
            epochs._data[good_epochs, ch_idx] -= overall_mean[ch_idx]

            evoked_data = epochs[good_epochs].average().data

            for event in epochs.events:
                start, stop = raw.time_as_index([tmin, tmax], use_rounding=True)
                start += event[0]
                stop = start + evoked_data.shape[1]

                # Using moving average on the evoked data
                evoked_data_smoothed = self.moving_average(
                    evoked_data[ch_idx], WINDOW_SIZE
                )
                corrected_data[
                    ch_idx, start:stop
                ] -= (evoked_data[ch_idx]*3)  # Using smoothed data for correction

        raw._data = corrected_data
        self._raw = raw


    def highly_correlated_epochs_new(self, epochs, threshold=0.975):
        """Return list of epochs that are highly correlated to the average."""
        n_epochs = len(epochs)

        # Start by including the first five epochs
        chosen = list(range(5))
        sum_data = np.sum(epochs._data[chosen], axis=0)

        # Check subsequent epochs
        for idx in range(5, n_epochs):
            avg_data = sum_data / len(chosen)
            corr = np.corrcoef(avg_data.squeeze(), epochs._data[idx].squeeze())[0, 1]
            if corr > threshold:
                sum_data += epochs._data[idx]
                chosen.append(idx)

        return chosen


    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    def pre_processing(self):
        # Apply highpassfilter
        self._raw.filter(l_freq=1, h_freq=None)
        print("Upsampling Data")
        self._upsample_data()

    def downsample(self):
        print("Downsampling Data")
        self._downsample_data()
        return

    def lowpass(self, h_freq=45):
        # Apply lowpassfilter
        print("Applying lowpassfilter")
        self._raw.filter(l_freq=1, h_freq=h_freq)
        return

    def printName(self):
        print(self._name)

    def _check_volume_gaps(self):
        if not hasattr(self, "_volumeGaps") or not self._volume_gaps:
            # np.diff berechnet die Differenz zwischen aufeinanderfolgenden Elementen in einem Array
            # np.ptp (peak to peak) gibt den Bereich (die Differenz zwischen dem Minimum und dem Maximum) eines Arrays zurück
            if np.ptp(np.diff(self._triggers)) > 3:
                self._volume_gaps = True
            else:
                self._volume_gaps = False

    def _derive_art_length(self):
        d = np.diff(
            [trigger * self._upsample for trigger in self._triggers]
        )  # trigger distances

        if self._volume_gaps:
            m = np.mean([np.min(d), np.max(d)])  # middle distance
            ds = d[d < m]  # trigger distances belonging to slice triggers
            # dv = d[d > m]  # trigger distances belonging to volume triggers

            # total length of an artifact
            self._art_length = np.max(ds)  # use max to avoid gaps between slices
        else:
            # total length of an artifact
            self._art_length = np.max(d)

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

    def _upsample_data(self):
        self._raw.resample(sfreq=self._raw.info["sfreq"] * self._upsample)

    def _downsample_data(self):
        # Resample (downsample) the data
        self._raw.resample(sfreq=self._raw.info["sfreq"] / self._upsample)
