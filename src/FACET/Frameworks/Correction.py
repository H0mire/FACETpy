""" Correction Framework Module

This module contains the Correction_Framework class, which is used to correct EEG data.

Author: Janik Michael Müller
Date: 15.02.2024
Version: 1.0
"""

import numpy as np
import mne, re
from scipy.stats import pearsonr
from FACET.helpers.moosmann import calc_weighted_matrix_by_realignment_parameters_file
from FACET.helpers.fastranc import fastranc
from FACET.Frameworks.Analytics import Analytics_Framework
from loguru import logger
from scipy.signal import filtfilt
# import inst for mne python


class Correction_Framework:
    """
    The Correction_Framework class is used to correct EEG data.

    The class contains methods to correct EEG data, such as removing artifacts, applying AAS and Moosmann correction,
    and preprocessing and postprocessing the data.

    Attributes:
        _eeg (dict): A dictionary containing the EEG metadata and data
        _plot_number (int): A counter for the number of plots.
        avg_artifact (numpy.ndarray): The average artifact matrix.
        avg_artifact_matrix (dict): A dictionary containing the average artifact matrix for each EEG channel.
        avg_artifact_matrix_numpy (dict): A dictionary containing the average artifact matrix for each EEG channel as a numpy array.
    """

    def __init__(
        self,
        FACET,
        eeg
    ):
        self._eeg=eeg
        self._FACET = FACET
        self._plot_number = 0
        self.avg_artifact = None
        self.avg_artifact_matrix = None
        self.avg_artifact_matrix_numpy = None


    def cut(self):
        """
        Crops the raw EEG data based on the time triggers.

        The method crops the raw EEG data from the start of the time triggers
        until the minimum of the end time and the sum of the time triggers end
        and the duration of artifacts.

        Returns:
            None
        """
        self._eeg["raw"].crop(
            tmin=self._eeg["time_triggers_start"],
            tmax=min(self._eeg["time_end"],
                self._eeg["time_triggers_end"]
                + self._eeg["duration_art"]
            ),
        )
        return
    def get_mne_raw(self):
            """
            Returns the raw MNE object.

            Returns:
                mne.io.Raw: The raw MNE object.
            """
            return self._eeg["raw"]
    def get_mne_raw_orig(self):
            """
            Returns the original raw EEG data.

            Returns:
                mne.io.Raw: The original raw EEG data.
            """
            return self._eeg["raw_orig"]
    def set_eeg(self, eeg):
        """
        Sets the EEG data for the Correction object.

        Args:
            eeg: The EEG data to be set.

        Returns:
            None
        """
        self._eeg = eeg
        return

    def prepare(self):
        """
        Prepares the data for correction by upsampling it.
        """
        self._upsample_data()

    def plot_EEG(self, start=0, title=None):
            """
            Plots the raw EEG data.

            Parameters:
            - start (int): The starting index of the data to be plotted.
            - title (str): The title of the plot. If not provided, a default title will be used.

            Returns:
                None
            """
            if not title:
                self._plot_number += 1
                title = str(self._plot_number)
            self._eeg["raw"].plot(title=title, start=start)

    def remove_artifacts(self, avg_artifact_matrix_numpy=None,plot_artifacts=False):
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
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels]
        # create dict with index as key and channel data as value
        corrected_data_template = {i: raw._data[i] for i in eeg_channels}
        idx_start, idx_stop = raw.time_as_index([self._eeg["tmin"], self._eeg["tmax"]], use_rounding=True)

        if avg_artifact_matrix_numpy is None:
            if self.avg_artifact_matrix_numpy is None:
                raise ValueError("No artifact matrix found")

        # iterate through first dimension of matrix
        trigger_offset = self._eeg["tmin"] * self._eeg["raw"].info["sfreq"]
        art_length = self._eeg["duration_art"] * self._eeg["raw"].info["sfreq"]
        info = mne.create_info(ch_names=channels_to_keep, sfreq=raw.info['sfreq'], ch_types='eeg')
        raw_avg_artifact = mne.EvokedArray(np.empty((len(channels_to_keep), int(art_length))), info) # Only for optional plotting
        noise = self._eeg["noise"]
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
                noise[ch_id,start:stop] += avg_artifact[key,:stop-start] 
                corrected_data_template[ch_id][start:stop] -= avg_artifact[key,:stop-start]
        if plot_artifacts: raw_avg_artifact.plot()
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
        """
        Returns the indices of highly correlated epochs based on a reference set of epochs.

        Parameters:
            full_epochs (Epochs): The full set of epochs.
            epoch_indices (list): The indices of epochs to be checked for correlation.
            epochs_indices_reference (list): The indices of reference epochs.
            threshold (float, optional): The correlation threshold. Defaults to 0.975.

        Returns:
            numpy.ndarray: The indices of highly correlated epochs.
        """
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
        """
        Calculate the chosen matrix based on the given epochs.

        Parameters:
            epochs (list): List of epochs.
            threshold (float): Threshold value for correlation.
            window_size (int): Size of the window for calculating correlations.
            rel_window_offset (float): Relative offset of the window.

        Returns:
            numpy.ndarray: The chosen matrix.
        """
        n_epochs = len(epochs)

        chosen_matrix = np.zeros((n_epochs, n_epochs))

        window_offset = int(window_size * rel_window_offset) # Get offset in amount of epochs

        for idx in range(0, n_epochs, window_size):
            offset_idx = idx + window_offset

            reference_indices = np.arange(idx, min(idx+5, n_epochs))
            candidates = np.arange(offset_idx, min(offset_idx + window_size, n_epochs))
            #remove all negative indices
            candidates = candidates[candidates >= 0]
            chosen = self.highly_correlated_epochs_with_indices(epochs, candidates,reference_indices, threshold=threshold)
            if len(chosen)== 0:
                continue
            indices = np.arange(idx, min(idx+window_size, n_epochs))
            chosen_matrix[np.ix_(indices, chosen)] = 1/len(chosen)

        return chosen_matrix


    def apply_Moosmann(self, file_path, window_size=25, threshold=5):
        """
        Apply Moosmann correction to the given file.

        Args:
            file_path (str): The path to the file.
            window_size (int, optional): The size of the window for calculating the weighted matrix. Defaults to 25.
            threshold (int, optional): The threshold for determining artifact segments. Defaults to 5.

        Returns:
            dict: A dictionary containing the average artifact matrix for each EEG channel.
        """
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
    def apply_ANC(self):
        """
        This method utilizes the _anc method to clean the eeg data from each channel
        """
    	
        logger.debug("applying ANC")
        try:
            raw = self._eeg["raw"].copy()
            eeg_channels = mne.pick_types(
                raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
            )
            channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
            raw.pick(channels_to_keep)  # raw wird in-place modifiziert
            noise = self._eeg["noise"][eeg_channels]

            _corrected_data = raw._data.copy()

            for key, val in enumerate(raw._data):
                _corrected_channel_data = self._anc(val, noise[key])
                _corrected_data[key] = _corrected_channel_data
            
            for ch_id, ch_d in enumerate(_corrected_data):
                ch_name = raw.ch_names[ch_id]
                ch_id_real = self._eeg["raw"].ch_names.index(ch_name)
                self._eeg["raw"]._data[ch_id_real] = ch_d
        except Exception as ex:
            logger.exception("An exception occured while applying ANC", ex)  
    def align_subsample():
        return "TODO: Implement"
             


    def _anc(self, EEG, Noise):
        acq_start = int(self._eeg["triggers"][0])
        acq_end = int(acq_start + self._eeg["art_length"])

        Reference = Noise[acq_start:acq_end].T
        tmpd = filtfilt(self._eeg["ANC_HP_Filter_Weights"], 1, EEG).T
        Data = tmpd[acq_start:acq_end].astype(float)
        Alpha = np.sum(Data * Reference) / np.sum(Reference * Reference)
        Reference = (Alpha * Reference).astype(float)
        mu = float(0.05 / (self._eeg["ANC_Filter_order"] * np.var(Reference)))

        # Use the fastranc function for adaptive noise cancellation
        _, FilteredNoise = fastranc(Reference, Data, self._eeg["ANC_Filter_order"], mu)

        if np.isinf(np.max(FilteredNoise)):
            logger.error('Warning: ANC failed, skipping ANC.')
        else:
            EEG[acq_start:acq_end] = EEG[acq_start:acq_end] - FilteredNoise.T

        return EEG

    def downsample(self):
        """
        Downsamples the data.

        This method downsamples the data by performing a downsampling operation on the data.
        It logs a message indicating that the downsampling is being performed.
        """
        logger.info("Downsampling Data")
        self._downsample_data()
        return
    def upsample(self):
        """
        Upsamples the data.
        This method performs upsampling on the data.

        Returns:
            None
        """
        logger.info("Upsampling Data")
        self._upsample_data()
        return
    
    
    def filter(self, l_freq=None, h_freq=None):
        """
        Apply a highpass filter to the raw EEG data.

        Args:
            l_freq (float): The lower cutoff frequency for the highpass filter.
        Returns:
            None
        """
        # Apply highpass filter
        logger.debug(f"Applying filter with l_freq={l_freq} and h_freq={h_freq}")
        self._eeg["raw"].filter(l_freq=l_freq, h_freq=h_freq)
    def split_vector(self, V, Marker, SecLength):
        """
        Splits a vector into multiple sections based on marker positions.

        Parameters:
        V (numpy.ndarray): The input vector.
        Marker (list): List of marker positions.
        SecLength (int): Length of each section.

        Returns:
        numpy.ndarray: A 2D array containing the split sections of the vector.
        """
        SecLength = int(SecLength)
        M = np.zeros((len(Marker), SecLength))
        for i, marker in enumerate(Marker):
            marker = int(marker)
            epoch = V[marker:(marker + SecLength)]
            M[i, :len(epoch)] = epoch
        return M
    def _upsample_data(self):
        """
        Upsamples the raw EEG data.

        This method resamples the raw EEG data by multiplying the sampling frequency
        with the upsampling factor.

        Parameters:
            None

        Returns:
            None
        """
        self.resample_data(self._eeg["raw"].info["sfreq"] * self._eeg["upsampling_factor"])

    def _downsample_data(self):
        """
        Resamples (downsamples) the data by reducing the sampling frequency.

        This method resamples the raw EEG data by dividing the sampling frequency
        by the upsampling factor. The resulting downsampled data is stored in the
        `_eeg["raw"]` attribute.

        Note:
            The `_eeg` attribute should be initialized before calling this method.

        """
        self.resample_data(self._eeg["raw"].info["sfreq"] / self._eeg["upsampling_factor"])

    def resample_data(self, sfreq):
        """
        Resamples (downsamples) the data by a given sampling frequency.

        Note:
            The `_eeg` attribute should be initialized before calling this method.
        """
        noise_raw=self._eeg["raw"].copy()
        noise_raw._data = self._eeg["noise"]
        self._eeg["noise"] = noise_raw.resample(sfreq=sfreq)._data.copy()
        #unload noise_raw
        noise_raw = None
        self._eeg["raw"].resample(sfreq=sfreq)
        regex = self._eeg["trigger_regex"]
        if regex:
            self._FACET._analytics.find_triggers(regex)


