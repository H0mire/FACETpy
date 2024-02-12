import numpy as np
import mne, re
from scipy.stats import pearsonr
from FACET.helpers.moosmann import single_motion, moving_average, calc_weighted_matrix_by_realignment_parameters_file
from loguru import logger
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

    def remove_artifacts(self, avg_artifact_matrix_numpy=None):
        """
        Remove artifacts from the EEG data.

        Args:
            avg_artifact_matrix_numpy (numpy.ndarray, optional): The average artifact matrix. If not provided,
                it will be retrieved from the instance variable `avg_artifact_matrix_numpy`. If both are None,
                a ValueError will be raised.

        Raises:
            ValueError: If no artifact matrix is found.

        Returns:
            None
        """
        raw = self._eeg["raw"].copy()

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
        # create array of indices for each channel by channel name
        channel_indices = [raw.ch_names.index(ch) for ch in channels_to_keep]
        # create dict with index as key and channel data as value
        corrected_data_template = {i: raw._data[i] for i in channel_indices}
        idx_start, idx_stop = raw.time_as_index([self._eeg["tmin"], self._eeg["tmax"]], use_rounding=True)

        if avg_artifact_matrix_numpy is None:
            if self.avg_artifact_matrix_numpy is None:
                raise ValueError("No artifact matrix found")

        # iterate through first dimension of matrix
        trigger_offset = self._eeg["tmin"] * self._eeg["raw"].info["sfreq"]
        art_length = self._eeg["duration_art"] * self._eeg["raw"].info["sfreq"]
        info = mne.create_info(ch_names=channels_to_keep, sfreq=raw.info['sfreq'], ch_types='eeg')
        raw_avg_artifact = mne.EvokedArray(np.empty((len(channels_to_keep), int(art_length))), info)
        counter = 0
        for ch_id, ch_matrix in self.avg_artifact_matrix_numpy.items():

            logger.debug(f"Removing Artifact from Channel {ch_id}", end=" ")
            eeg_data_zero_mean = np.array(corrected_data_template[ch_id]) - np.mean(corrected_data_template[ch_id])
            data_split_on_epochs = self.split_vector(eeg_data_zero_mean, np.array(self._eeg["triggers"])+trigger_offset, art_length)
            avg_artifact = ch_matrix @ data_split_on_epochs

            raw_avg_artifact.data[counter] = avg_artifact[0]
            counter += 1

            for key, pos in enumerate(self._eeg["triggers"]):
                start = idx_start+ pos 
                minColumn = avg_artifact.shape[1]
                stop = min(start + minColumn, corrected_data_template[ch_id].shape[0])
                corrected_data_template[ch_id][start:stop] -= avg_artifact[key,:stop-start]
            # raw_avg_artifact.plot()
        for i in corrected_data_template.keys():
            self._eeg["raw"]._data[i] = corrected_data_template[i]


    def apply_AAS(self, rel_window_position=0, window_size=25):
        """
        Applies the AAS (Artifact Averaging Subtraction) matrix using numpy.

        Args:
            rel_window_offset (int): Relative window offset for artifact averaging.
            window_size (int): Size of the window for artifact averaging.

        Returns:
            dict: A dictionary containing the averaged artifact matrix for each channel.
        """
        raw = self._eeg["raw"].copy()  # Erstelle eine Kopie, um das Original unverändert zu lassen

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
        raw.pick(channels_to_keep)  # raw wird in-place modifiziert

        # Epochen erstellen
        epochs = mne.Epochs(
            raw,
            events=self._eeg["filtered_events"],
            tmin=self._eeg["tmin"],
            tmax=self._eeg["tmax"],
            baseline=None,
            reject=None,
            preload=True,
        )
        original_epochs = epochs.copy()
        avg_matrix_3d = {}
        for ch_name in epochs.ch_names:
            idx = self._eeg["raw"].ch_names.index(ch_name)
            logger.debug(f"Averaging Channel {ch_name}", end=" ")
            epochs_single_channel = original_epochs.copy().pick([ch_name])
            chosen_matrix = self.calc_chosen_matrix(epochs_single_channel, rel_window_offset=rel_window_position, window_size=window_size)
            avg_matrix_3d[idx] = chosen_matrix
        
        self.avg_artifact_matrix_numpy = avg_matrix_3d
        return avg_matrix_3d

    def highly_correlated_epochs_with_indices(self, full_epochs, epoch_indices, epochs_indices_reference, threshold=0.975):
        """Return list of epochs that are highly correlated to the average."""
        #check if vars are set
        #check if epochs_reference is not empty
        if len(epochs_indices_reference) == 0:
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
    

    def calc_chosen_matrix(self, epochs, threshold=0.975, window_size=25, rel_window_offset=0):
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


    def apply_Moosmann(self, file_path, window_size=25, threshold=5):
        motiondata_struct, weighting_matrix = calc_weighted_matrix_by_realignment_parameters_file(file_path, self._eeg["num_triggers"], window_size, threshold=threshold)
        logger.debug(weighting_matrix)
        #determine number of eeg data only channels
        eeg_channel_indices = mne.pick_types(self._eeg["raw"].info, meg=False, eeg=True, stim=False, exclude='bads')

        avg_artifact_matrix_every_channel = {}
        #ensure every row in weighting matrix sums up to 1 
        weighting_matrix = weighting_matrix / np.sum(weighting_matrix, axis=1)[:, np.newaxis]
        #add weighting matrix to every channel
        for idx in eeg_channel_indices:
            avg_artifact_matrix_every_channel[idx] = weighting_matrix

        self.avg_artifact_matrix_numpy=avg_artifact_matrix_every_channel
        return avg_artifact_matrix_every_channel

    def downsample(self):
        logger.info("Downsampling Data")
        self._downsample_data()
        return
    def upsample(self):
        logger.info("Upsampling Data")
        self._upsample_data()
        return
    def lowpass(self, h_freq=45):
        # Apply lowpassfilter
        logger.info("Applying lowpassfilter")
        self._eeg["raw"].filter(l_freq=None, h_freq=h_freq)
        return
    def highpass(self, l_freq=1):
        # Apply highpassfilter
        logger.info("Applying highpassfilter")
        self._eeg["raw"].filter(l_freq=l_freq, h_freq=None)
    def split_vector(self, V, Marker, SecLength):
        SecLength = int(SecLength)
        M = np.zeros((len(Marker), SecLength))
        for i, marker in enumerate(Marker):
            marker = int(marker)
            epoch = V[marker:(marker + SecLength)]
            M[i, :len(epoch)] = epoch
        return M
    def _upsample_data(self):
        self._eeg["raw"].resample(sfreq=self._eeg["raw"].info["sfreq"] * self._eeg["upsampling_factor"])

    def _downsample_data(self):
        # Resample (downsample) the data
        self._eeg["raw"].resample(sfreq=self._eeg["raw"].info["sfreq"] / self._eeg["upsampling_factor"])
