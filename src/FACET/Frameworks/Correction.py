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
        self.avg_artifact_matrix_numpy = None


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
        idx_start, idx_stop = raw.time_as_index([self._eeg["tmin"], self._eeg["tmax"]], use_rounding=True)

        if self.avg_artifact_matrix is not None:
            for key, pos in enumerate(self._eeg["triggers"]):
                start = idx_start +pos 
                minColumn = self.avg_artifact_matrix.shape[2]
                stop = min(start + minColumn, corrected_data.shape[1])
                corrected_data[:self.avg_artifact_matrix.shape[0], start:stop] -= self.avg_artifact_matrix[:,key,:stop-start]
        elif self.avg_artifact_matrix_numpy is not None:
            #iterate through first dimension of matrix
            trigger_offset = self._eeg["tmin"] * self._eeg["raw"].info["sfreq"]
            art_length = self._eeg["duration_art"] * self._eeg["raw"].info["sfreq"]
            info = mne.create_info(ch_names=["ch1"], sfreq=raw.info['sfreq'], ch_types='eeg')
            #raw_avg_artifact = mne.EvokedArray(np.empty((1, int(art_length))), info)
            for ch_id, ch_matrix in enumerate(self.avg_artifact_matrix_numpy):
                print(f"Removing Artifact from Channel {ch_id}", end=" ")
                eeg_data_zero_mean = np.array(corrected_data[ch_id]) - np.mean(corrected_data[ch_id])
                data_split_on_epochs = self.split_vector(eeg_data_zero_mean, np.array(self._eeg["triggers"])+trigger_offset, art_length)
                avg_artifact = ch_matrix @ data_split_on_epochs

                # raw_avg_artifact.data[0] = avg_artifact[0]
                # raw_avg_artifact.plot()

                for key, pos in enumerate(self._eeg["triggers"]):
                    start = idx_start+ pos 
                    minColumn = avg_artifact.shape[1]
                    stop = min(start + minColumn, corrected_data.shape[1])
                    corrected_data[ch_id, start:stop] -= avg_artifact[key,:stop-start]
                print()
        else:
            evoked = self.avg_artifact
            for pos in self._eeg["triggers"]:
                start = idx_start+pos 
                minColumn = evoked.data.shape[1]
                stop = min(start + minColumn, corrected_data.shape[1])
                corrected_data[:evoked.data.shape[0], start:stop] -= evoked.data[:,:stop-start]
            

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
    
    def apply_MNE_AAS_matrix(self): #DEPRECATED: Use Numpy Matrix instead
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
                #print(f"{idx}, ", end="")
                if np.sum(chosen_matrix[idx]) > 0:
                    indices = np.where(chosen_matrix[idx] == 1)[0]
                    evoked = epochs_single_channel[indices].average()
                    art_per_epoch.append(evoked.data[0])
                else:
                    art_per_epoch.append(np.zeros(epochs_single_channel[0].data.shape[1]))
            evoked_data.append(art_per_epoch)  # Hinzufügen der Daten zum Array
            print()
                

        self.avg_artifact_matrix = np.array(evoked_data)
    def apply_MNE_AAS_matrix_numpy(self, rel_window_offset=0):
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
        original_epochs = epochs.copy()
        avg_matrix_3d = np.zeros((len(epochs.ch_names), len(epochs), len(epochs)))
        for idx, ch_name in enumerate(epochs.ch_names):
            print(f"Averaging Channel {ch_name}", end=" ")
            epochs_single_channel = original_epochs.copy().pick([ch_name])
            chosen_matrix = self.highly_correlated_epochs_matrix_weighted(epochs_single_channel, rel_window_offset=rel_window_offset)
            print(chosen_matrix[:10,:10])
            avg_matrix_3d[idx] = chosen_matrix
            print()
        
        self.avg_artifact_matrix_numpy = avg_matrix_3d

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
    def highly_correlated_epochs_with_indices(self, full_epochs, epoch_indices, epochs_indices_reference, threshold=0.975):
        """Return list of epochs that are highly correlated to the average."""
        #check if vars are set
        #check if epochs_reference is not empty
        if len(epochs_indices_reference) == 0:
            return np.array([])
        #check if epochs is not empty
        if len(epoch_indices) == 0:
            return np.array([])
        
        sum_data = np.sum(full_epochs._data[epochs_indices_reference], axis=0)
        chosen = list(epochs_indices_reference)
        # Check subsequent epochs
        for idx in epoch_indices:
            #check if idx is already in chosen
            if idx in chosen:
                continue
            avg_data = sum_data / len(chosen)
            corr = np.corrcoef(avg_data.squeeze(), full_epochs._data[idx].squeeze())[0, 1]
            if corr > threshold:
                sum_data += full_epochs._data[idx]
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
    def highly_correlated_epochs_matrix_weighted(self, epochs, threshold=0.975, window_size=25, rel_window_offset=0):
        """Return list of epochs that are highly correlated to the average."""
        n_epochs = len(epochs)

        chosen_matrix = np.zeros((n_epochs, n_epochs))

        window_offset = int(window_size * rel_window_offset) # Get offset in amount of epochs

        for idx in range(0, n_epochs, window_size):
            offset_idx = idx + window_offset

            reference_indices = np.arange(idx, min(idx+5, n_epochs))
            candidates = np.arange(offset_idx, min(offset_idx + window_size, n_epochs))
            #remove all negative indices
            candidates = candidates[candidates >= 0]
            #calculate offset
            chosen = self.highly_correlated_epochs_with_indices(epochs, candidates,reference_indices, threshold=threshold)
            if len(chosen)== 0:
                continue
            indices = np.arange(idx, min(idx+window_size, n_epochs))
            chosen_matrix[np.ix_(indices, chosen)] = 1/len(chosen)

        return chosen_matrix

    def highly_correlated_epochs_matrix_weighted_every_epoch_separate(self, epochs, threshold=0.975, window_size=25, rel_window_offset=-0.5):
        """Return list of epochs that are highly correlated to the average."""
        n_epochs = len(epochs)

        chosen_matrix = np.zeros((n_epochs, n_epochs))
        if rel_window_offset > 0:
            raise ValueError("rel_window_offset must be negative")

        window_offset = int(window_size * rel_window_offset)

        for idx in range(0, n_epochs):
            offset_idx = idx + window_offset
            candidates = np.arange(offset_idx, min(offset_idx + window_size, n_epochs))
            #remove all negative indices
            candidates = candidates[candidates >= 0]
            include_window_length = 5
            #calculate offset
            offset = offset_idx-window_offset if offset_idx < 0 else window_offset*(-1)
            chosen = self.highly_correlated_epochs_with_reference_epochs(epochs[candidates], threshold=threshold, include_window_length=include_window_length, window_offset=offset)
            chosen += offset_idx 
            chosen_matrix[np.ix_([candidates[0]], chosen)] = 1/len(chosen)

        return chosen_matrix


    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    
    def apply_Moosmann(self):
        #TODO: Implement Moosmann Algorithm
        pass

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
    def split_vector(self, V, Marker, SecLength):
        SecLength = int(SecLength)
        M = np.zeros((len(Marker), SecLength))
        for i, marker in enumerate(Marker):
            marker = int(marker)
            M[i, :] = V[marker:(marker + SecLength)]
        return M
    def _upsample_data(self):
        self._eeg["raw"].resample(sfreq=self._eeg["raw"].info["sfreq"] * self._eeg["upsampling_factor"])

    def _downsample_data(self):
        # Resample (downsample) the data
        self._eeg["raw"].resample(sfreq=self._eeg["raw"].info["sfreq"] / self._eeg["upsampling_factor"])
