import numpy as np
import mne, re
from scipy.stats import pearsonr

# import inst for mne python


class Correction_Framework:
    def __init__(
        self,
        eeg
    ):
        self._eeg=eeg


    def cut(self):
        self._eeg["raw"].crop(
            tmin=self._eeg["time_triggers_start"],
            tmax=min(self._eeg["time_end"],
                self._eeg["time_triggers_end"]
                + self._eeg["duration_art"],
            ),
        )
        return
    #TODO: Implement better Structure
    def get_mne_raw(self):
        return self._eeg["raw"]
    def get_mne_raw_orig(self):
        return self._eeg["raw_orig"]

    def prepare(self):
        self._upsample_data()

    def plot_EEG(self):
        self._plot_number += 1
        self._eeg["raw"].plot(title=str(self._plot_number), start=27)

    def remove_artifacts(self):        
        raw = self._eeg["raw"]
        corrected_data = raw._data.copy()
        evoked = self.avg_artifact
        for pos in self._eeg["triggers"]:
            start, stop = raw.time_as_index([self._tmin, self._tmax], use_rounding=True)
            start += pos
            minColumn = evoked.data.shape[1]
            stop = start + minColumn
            corrected_data[:evoked.data.shape[0], start:stop] -= evoked.data

        raw._data = corrected_data

        print(raw.ch_names)

        self._eeg["raw"] = raw

    # Remove Artifacts from EEG
    def apply_MNE_AAS_old(self):
        raw = self._eeg["raw"]
        # Schritt 1: Epochen erstellen
        
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False)
        epochs = mne.Epochs(
            raw,
            self._eeg["events"],
            picks=picks,
            tmin=self._eeg["tmin"],
            tmax=self._eeg["tmax"],
            baseline=None,
            reject=None,
            preload=True,
        )

        # Schritt 2: Durchschnittlichen Artefakt berechnen
        evoked = epochs.average()

        # Schritt 4: Subtraktion
        raw_ssp = raw.copy().add_proj(mne.compute_proj_evoked(evoked))
        raw_ssp.apply_proj()
        self._eeg["raw"] = raw_ssp

    # Calculating Average Artifact 
    def apply_MNE_AAS(self):
        raw = self._eeg["raw"].copy()  # Erstelle eine Kopie hier, um das Original unverändert zu lassen

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
            events=self._eeg["events"],
            picks=eeg_channels[:],
            tmin=self._eeg["tmin"],
            tmax=self._eeg["tmax"],
            baseline=None,
            reject=None,
            preload=True,
        )

        print("Epochs shape: ", epochs.get_data().shape)

        good_epochs = self.highly_correlated_epochs(epochs)
        # Schritt 3: Durchschnittliches Artefakt berechnen
        # evoked = epochs.average()
        evoked = epochs[good_epochs].average()
        #mne plot evoked
        evoked.plot()
        self.avg_artifact = evoked
        
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
        raw = self._eeg["raw"].copy() # Erstelle eine Kopie, um das Original unverändert zu lassen


        # Schritt 1: Kanalauswahl
        raw.info["bads"] = ["Status", "EMG", "ECG"]
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
        raw.pick_channels(channels_to_keep)  # raw wird in-place modifiziert

        # Epochen erstellen
        epochs = mne.Epochs(
            raw,
            events=self._eeg["events"],
            picks=eeg_channels[:],
            tmin=self._eeg["tmin"],
            tmax=self._eeg["tmax"],
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

        #overall_mean = raw._data.mean(axis=1, keepdims=True)
        for ch_idx, good_epochs in enumerate(good_epochs_per_channel):
            print("Channel: ", (ch_idx + 1), "/", n_channels)
            #epochs._data[good_epochs, ch_idx] -= overall_mean[ch_idx]

            evoked = epochs[good_epochs].average()

            for event in epochs.events:
                start, stop = raw.time_as_index([tmin, tmax], use_rounding=True)
                start += event[0]
                stop = start + evoked.data.shape[1]

            
                corrected_data[
                    ch_idx, start:stop
                ] -= evoked.data[ch_idx]  # Using smoothed data for correction

        raw._data = corrected_data
        self._eeg["raw"] = raw

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

    def downsample(self):
        print("Downsampling Data")
        self._downsample_data()
        return
    def upsample(self):
        print("Upsampling Data")
        self._upsample_data()
        return
    def lowpass(self, h_freq=45):
        # Apply lowpassfilter
        print("Applying lowpassfilter")
        self._raw.filter(l_freq=None, h_freq=h_freq)
        return
    def highpass(self, l_freq=1):
        # Apply highpassfilter
        print("Applying highpassfilter")
        self._raw.filter(l_freq=l_freq, h_freq=None)
    def _upsample_data(self):
        self._raw.resample(sfreq=self._raw.info["sfreq"] * self._upsample)
    def _downsample_data(self):
        # Resample (downsample) the data
        self._raw.resample(sfreq=self._raw.info["sfreq"] / self._upsample)
