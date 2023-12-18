import numpy as np
import mne, re
from scipy.stats import pearsonr

# import inst for mne python


class Correction_Framework:
    def __init__(
        self,
        eeg,
    ):
        self._eeg=eeg
        self._plot_number = 0
        self.avg_artifact = None
        self.avg_artifact_matrix = None


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
    def set_eeg(self, eeg):
        self._eeg = eeg
        return

    def prepare(self):
        self._upsample_data()

    def plot_EEG(self, start=0, title=None):
        if not title:
            self._plot_number += 1
            title = str(self._plot_number)
        self._eeg["raw"].plot(title=title, start=start)

    #TODO: Better indexing for channels as the first channels might be not eeg channels
    def remove_artifacts(self):        
        raw = self._eeg["raw"].copy()

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
        corrected_data = raw.pick(channels_to_keep)._data
        if self.avg_artifact_matrix is None:
            evoked = self.avg_artifact
            for pos in self._eeg["triggers"]:
                start, stop = raw.time_as_index([self._eeg["tmin"], self._eeg["tmax"]], use_rounding=True)
                start += pos 
                minColumn = evoked.data.shape[1]
                stop = min(start + minColumn, corrected_data.shape[1])
                corrected_data[:evoked.data.shape[0], start:stop] -= evoked.data[:,:stop-start]

            raw._data[:corrected_data.shape[0]] = corrected_data
            print(raw.ch_names)
            self._eeg["raw"]._data[:raw._data.shape[0]] = raw._data
        else:
            for key, pos in enumerate(self._eeg["triggers"]):
                start, stop = raw.time_as_index([self._eeg["tmin"], self._eeg["tmax"]], use_rounding=True)
                start += pos 
                minColumn = self.avg_artifact_matrix.shape[2]
                stop = min(start + minColumn, corrected_data.shape[1])
                corrected_data[:self.avg_artifact_matrix.shape[0], start:stop] -= self.avg_artifact_matrix[:,key,:stop-start]

            raw._data[:corrected_data.shape[0]] = corrected_data
            print(raw.ch_names)
            self._eeg["raw"]._data[:raw._data.shape[0]] = raw._data

    # Calculating Average Artifact 
    def apply_MNE_AAS_old(self):
        raw = self._eeg["raw"].copy()  # Erstelle eine Kopie hier, um das Original unverändert zu lassen
        # Schritt 1: Kanalauswahl

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
        raw.pick(channels_to_keep)  # raw wird in-place modifiziert

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

        good_epochs = self.highly_correlated_epochs_new(epochs)
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

    def apply_MNE_AAS(self):
        raw = self._eeg["raw"].copy()  # Erstelle eine Kopie, um das Original unverändert zu lassen

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
        raw.pick(channels_to_keep)  # raw wird in-place modifiziert

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

        evoked_data = []  # Liste für Daten der einzelnen Evoked-Objekte

        original_epochs = epochs.copy()
        for ch_name in epochs.ch_names:
            epochs_single_channel = original_epochs.copy().pick([ch_name])
            good_epochs = self.highly_correlated_epochs_new(epochs_single_channel)

            # Erstellen eines Evoked-Objekts für die guten Epochen dieses Kanals
            evoked = epochs_single_channel[good_epochs].average()
            evoked_data.append(evoked.data[0])  # Hinzufügen der Daten zum Array

        # Kombinieren der Daten in ein Array
        combined_data = np.stack(evoked_data, axis=0)

        # Erstellen eines neuen Evoked-Objekts mit den kombinierten Daten
        combined_evoked = mne.EvokedArray(
            combined_data,
            info=original_epochs.info,  # Nutze die Info des letzten Evoked-Objekts
            tmin=original_epochs.times[0]
        )

        combined_evoked.plot()

        self.avg_artifact = combined_evoked
    
    def apply_MNE_AAS_matrix(self): #TODO: Averaged artefact in window x Epochs around current epoch. Consider using Matrix.
        raw = self._eeg["raw"].copy()  # Erstelle eine Kopie, um das Original unverändert zu lassen

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
        raw.pick(channels_to_keep)  # raw wird in-place modifiziert

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

        evoked_data = []  # Liste für Daten der einzelnen Evoked-Objekte

        original_epochs = epochs.copy()
        for ch_name in epochs.ch_names:
            epochs_single_channel = original_epochs.copy().pick([ch_name])
            chosen_matrix = self.highly_correlated_epochs_matrix(epochs_single_channel)
            print(f"Averaging Channel {ch_name} Epochs:", end=" ")
            art_per_epoch = []
            for idx in range(len(chosen_matrix)):
                print(f"{idx}, ", end="")
                if np.sum(chosen_matrix[idx]) > 0:
                    indices = np.where(chosen_matrix[idx] == 1)[0]
                    evoked = epochs_single_channel[indices].average()
                    art_per_epoch.append(evoked.data[0])
                else:
                    art_per_epoch.append(np.zeros(epochs_single_channel[0].data.shape[1]))
            evoked_data.append(art_per_epoch)  # Hinzufügen der Daten zum Array
            print()
                

        self.avg_artifact_matrix = np.array(evoked_data)

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

        return np.array(chosen)
    
    def highly_correlated_epochs_matrix(self, epochs, threshold=0.975, window_size=25):
        """Return list of epochs that are highly correlated to the average."""
        n_epochs = len(epochs)

        chosen_matrix = np.zeros((n_epochs, n_epochs))

        for idx in range(0, n_epochs, window_size):
            candidates = np.arange(idx, min(idx + window_size, n_epochs))
            chosen = self.highly_correlated_epochs_new(epochs[candidates], threshold=threshold)
            chosen += idx 
            chosen_matrix[np.ix_(candidates, chosen)] = 1

        return chosen_matrix

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
        self._eeg["raw"].filter(l_freq=None, h_freq=h_freq)
        return
    def highpass(self, l_freq=1):
        # Apply highpassfilter
        print("Applying highpassfilter")
        self._eeg["raw"].filter(l_freq=l_freq, h_freq=None)
    def _upsample_data(self):
        self._eeg["raw"].resample(sfreq=self._eeg["raw"].info["sfreq"] * self._eeg["upsampling_factor"])

    def _downsample_data(self):
        # Resample (downsample) the data
        self._eeg["raw"].resample(sfreq=self._eeg["raw"].info["sfreq"] / self._eeg["upsampling_factor"])
