"""
Correction framework Module

This module contains the CorrectionFramework class, which is used to correct EEG data.

Author: Janik Michael Müller
Date: 15.02.2024
Version: 1.0
"""

import numpy as np
import mne
from facet.helpers.moosmann import calc_weighted_matrix_by_realignment_parameters_file
from facet.helpers.fastranc import fastr_anc
from facet.helpers.utils import split_vector
from facet.helpers.crosscorr import crosscorrelation
from loguru import logger
from scipy.signal import firls, filtfilt, fftconvolve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from facet.eeg_obj import EEG


# import inst for mne python


class CorrectionFramework:
    """
    The CorrectionFramework class is used to correct EEG data.

    The class contains methods to correct EEG data, such as removing artifacts, applying Adaptive Autoregressive Models (AAR) and Moosmann correction,
    and preprocessing and postprocessing the data.

    Attributes:
        _eeg (facet.eeg_obj): A dictionary containing the EEG metadata and data.
        _plot_number (int): A counter for the number of plots generated.
        avg_artifact (numpy.ndarray): The average artifact matrix.
        avg_artifact_matrix (dict): A dictionary containing the average artifact matrix for each EEG channel.
        avg_artifact_matrix_numpy (dict): A dictionary containing the average artifact matrix for each EEG channel as a numpy array.
    """

    def __init__(self, facet, eeg):
        """
        Initializes the CorrectionFramework class with necessary components for EEG correction.

        Parameters:
            facet: A reference to a facet class instance, providing access to facet's functionalities.
            eeg: An EEG data structure, including metadata and raw EEG data.
        """
        self._eeg = eeg
        self._facet = facet
        self._plot_number = 0
        self.avg_artifact = None
        self.avg_artifact_matrix = None
        self.avg_artifact_matrix_numpy = None
        self.sub_sample_alignment = None

    def cut(self):
        """
        Crops the raw EEG data based on the time triggers.

        The method crops the raw EEG data from the start of the time triggers
        until the minimum of the end time and the sum of the time triggers end
        and the duration of artifacts.

        Returns:
            None
        """
        self._eeg.mne_raw.crop(
            tmin=self._eeg.time_acq_start, tmax=self._eeg.time_acq_end
        )
        return

    def get_mne_raw(self):
        """
        Returns the raw MNE object.

        Returns:
            mne.io.Raw: The raw MNE object.
        """
        return self._eeg.mne_raw

    def get_mne_raw_orig(self):
        """
        Returns the original raw EEG data.

        Returns:
            mne.io.Raw: The original raw EEG data.
        """
        return self._eeg.mne_raw_orig

    def set_eeg(self, eeg):
        """
        Sets the EEG data for the Correction object.

        Parameters:
            eeg: The EEG data to be set.

        Returns:
            None
        """
        self._eeg = eeg
        return

    def calc_avg_artifact(self, avg_artifact_matrix_numpy=None, plot_artifacts=False):
        """
        Calculates the average artifact for each channel.

        Parameters:
            avg_artifact_matrix_numpy (numpy.ndarray, optional): The average artifact matrix. If not provided,
            it will be retrieved from the instance variable `avg_artifact_matrix_numpy`. If both are None,
            a ValueError will be raised.
            plot_artifacts (bool, optional): Whether to plot the artifacts. Defaults to False.

        Raises:
            ValueError: If no artifact matrix is found.
        """
        logger.debug("Calculating Average Artifacts")
        raw = self._eeg.mne_raw

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels]
        # create dict with index as key and channel data as value
        corrected_data_template = {i: raw._data[i] for i in eeg_channels}

        # iterate through first dimension of matrix
        trigger_offset_in_samples = (
            self._eeg.artifact_to_trigger_offset * self._eeg.mne_raw.info["sfreq"]
        )
        art_length = self._eeg.artifact_length
        if plot_artifacts:
            info = mne.create_info(
                ch_names=channels_to_keep, sfreq=raw.info["sfreq"], ch_types="eeg"
            )
        if plot_artifacts:
            raw_avg_artifact = mne.EvokedArray(
                np.empty((len(channels_to_keep), int(art_length))), info
            )  # Only for optional plotting
        counter = 0
        artifacts = []
        for ch_id, ch_matrix in avg_artifact_matrix_numpy.items():

            logger.debug(
                f"Calculating Artifact for Channel {ch_id}:{raw.ch_names[ch_id]}",
                end=" ",
            )
            eeg_data_zero_mean = corrected_data_template[ch_id] - np.mean(
                corrected_data_template[ch_id]
            )
            data_split_on_epochs = split_vector(
                eeg_data_zero_mean,
                np.array(self._eeg.loaded_triggers) + trigger_offset_in_samples,
                art_length,
            )
            # check if the number of epochs in matrix is equal to the number of triggers
            while len(data_split_on_epochs) != len(ch_matrix):
                # remove the last epoch from data_split_on_epochs
                data_split_on_epochs = data_split_on_epochs[:-1]
            avg_artifact = ch_matrix @ data_split_on_epochs
            if len(ch_matrix) != len(self._eeg.loaded_triggers):
                # copy last artifact to the end of the avg_artifact
                avg_artifact = np.append(
                    avg_artifact, avg_artifact[-1].reshape(1, -1), axis=0
                )
            artifacts.append(avg_artifact)

            if plot_artifacts:
                raw_avg_artifact.data[counter] = avg_artifact[0]
            counter += 1
        if plot_artifacts:
            raw_avg_artifact.plot()
        return artifacts

    def remove_artifacts(self, avg_artifact_matrix_numpy=None, plot_artifacts=False):
        """
        Removes artifacts from the EEG data.

        Parameters:
            avg_artifact_matrix_numpy (numpy.ndarray, optional): The average artifact matrix. If not provided,
            it will be retrieved from the instance variable `avg_artifact_matrix_numpy`. If both are None,
            a ValueError will be raised.
            plot_artifacts (bool, optional): Whether to plot the artifacts. Defaults to False.

        Raises:
            ValueError: If no artifact matrix is found.
        """
        if avg_artifact_matrix_numpy is None:
            if self.avg_artifact_matrix_numpy is None:
                raise ValueError(
                    "No artifact matrix found. Please provide an artifact matrix by passing it as an argument or by calling the calc_matrix_aas method before calling remove_artifacts."
                )
            avg_artifact_matrix_numpy = self.avg_artifact_matrix_numpy
        raw = self._eeg.mne_raw

        artifacts = self.calc_avg_artifact(avg_artifact_matrix_numpy, plot_artifacts)
        smin, smax = self._eeg.smin, self._eeg.smax
        aligned_triggers = self._align_triggers_averaged_artifacts(
            raw._data[list(avg_artifact_matrix_numpy.keys())[0]],
            artifacts[0],
            search_window=3 * self._eeg.upsampling_factor,
        )
        noise = self._eeg.estimated_noise
        for i, ch_id in enumerate(avg_artifact_matrix_numpy.keys()):
            logger.debug(
                f"Removing Artifact from Channel {ch_id}:{raw.ch_names[ch_id]}"
            )
            for key, pos in enumerate(aligned_triggers):
                start = pos + smin
                stop = min(pos + smax, raw._data[ch_id].shape[0])
                avg_artifact = artifacts[i][key, : stop - start]
                noise[ch_id, start:stop] += avg_artifact
                raw._data[ch_id][start:stop] -= avg_artifact

    def calc_matrix_aas(self, rel_window_position=0, window_size=30, channels=None):
        """
        Applies the Adaptive Artifact Subtraction (AAS) matrix using numpy.

        Parameters:
            rel_window_position (int, optional): Relative window position for artifact averaging.
            window_size (int, optional): Size of the window for artifact averaging. Defaults to 30.
            channels (list, optional): Channels to average. If None, all EEG channels are used.

        Returns:
            dict: A dictionary containing the averaged artifact matrix for each channel.
        """
        raw = (
            self._eeg.mne_raw
        )  # Erstelle eine Kopie, um das Original unverändert zu lassen

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        if channels is None:
            channels_to_average = [raw.ch_names[i] for i in eeg_channels[:]]
        else:
            channels_to_average = [raw.ch_names[i] for i in channels[:]]
        # Epochen erstellen
        epochs = mne.Epochs(
            raw,
            events=self._eeg.triggers_as_events,
            tmin=self._eeg.tmin,
            tmax=self._eeg.tmax,
            baseline=None,
            reject=None,
            preload=True,
            picks=eeg_channels,
            event_repeated='drop',  # Handle duplicate event times
        )
        if len(epochs) != len(self._eeg.loaded_triggers):
            # Because of possible bad triggers, we need to check if the number of epochs is equal to the number of triggers
            logger.warning(
                "Number of epochs is not equal to the number of triggers. Please check your data. Imcomplete data?"
            )
        avg_matrix_3d = {}
        for key, ch_name in enumerate(channels_to_average):
            idx = self._eeg.mne_raw.ch_names.index(ch_name)
            logger.debug(f"Averaging Channel {idx}:{ch_name}", end=" ")
            epochs_single_channel = np.squeeze(epochs._data[:, key, :])
            chosen_matrix = self.calc_chosen_matrix(
                epochs_single_channel,
                rel_window_offset=rel_window_position,
                window_size=window_size,
            )
            avg_matrix_3d[idx] = chosen_matrix

        self.avg_artifact_matrix_numpy = avg_matrix_3d
        return avg_matrix_3d

    def highly_correlated_epochs_with_indices(
        self, full_epochs, epoch_indices, epochs_indices_reference, threshold=0.975
    ):
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
        # check if epochs_reference is not empty
        if len(epochs_indices_reference) == 0:
            return np.array([])

        sum_data = np.sum(full_epochs[epochs_indices_reference], axis=0)
        chosen = list(epochs_indices_reference)
        # Check subsequent epochs
        for idx in epoch_indices:
            # check if idx is already in chosen
            if idx in chosen:
                continue
            avg_data = sum_data / len(chosen)
            corr = np.corrcoef(avg_data.squeeze(), full_epochs[idx].squeeze())[0, 1]
            if corr > threshold:
                sum_data += full_epochs[idx]
                chosen.append(idx)

        return np.array(chosen)

    def calc_chosen_matrix(
        self, epochs, threshold=0.975, window_size=30, rel_window_offset=0
    ):
        """
        Calculates the chosen matrix based on the given epochs.

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

        window_offset = int(
            window_size * rel_window_offset
        )  # Get offset in amount of epochs

        for idx in range(0, n_epochs, window_size):
            offset_idx = idx + window_offset

            reference_indices = np.arange(idx, min(idx + 5, n_epochs))
            candidates = np.arange(offset_idx, min(offset_idx + window_size, n_epochs))
            # remove all negative indices
            candidates = candidates[candidates >= 0]
            chosen = self.highly_correlated_epochs_with_indices(
                epochs, candidates, reference_indices, threshold=threshold
            )
            if len(chosen) == 0:
                continue
            indices = np.arange(idx, min(idx + window_size, n_epochs))
            chosen_matrix[np.ix_(indices, chosen)] = 1 / len(chosen)

        return chosen_matrix

    def calc_matrix_motion(self, file_path, window_size=30, threshold=5):
        """
        Applies Moosmann correction to the given file.

        Parameters:
            file_path (str): The path to the file.
            window_size (int, optional): The size of the window for calculating the weighted matrix. Defaults to 25.
            threshold (int, optional): The threshold for determining artifact segments. Defaults to 5.

        Returns:
            dict: A dictionary containing the average artifact matrix for each EEG channel.
        """
        motiondata_struct, weighting_matrix = (
            calc_weighted_matrix_by_realignment_parameters_file(
                file_path, self._eeg.count_triggers, window_size, threshold=threshold
            )
        )
        logger.debug(weighting_matrix)
        # determine number of eeg data only channels
        eeg_channel_indices = mne.pick_types(
            self._eeg.mne_raw.info, meg=False, eeg=True, stim=False, exclude="bads"
        )

        avg_artifact_matrix_every_channel = {}
        # ensure every row in weighting matrix sums up to 1
        weighting_matrix = (
            weighting_matrix / np.sum(weighting_matrix, axis=1)[:, np.newaxis]
        )
        # add weighting matrix to every channel
        for idx in eeg_channel_indices:
            avg_artifact_matrix_every_channel[idx] = weighting_matrix

        self.avg_artifact_matrix_numpy = avg_artifact_matrix_every_channel
        return avg_artifact_matrix_every_channel

    def apply_ANC(self):
        """
        Applies Adaptive Noise Cancellation (ANC) to clean the EEG data from each channel.
        """

        logger.debug("applying ANC")
        try:
            raw = self._eeg.mne_raw
            eeg_channels = mne.pick_types(
                raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
            )
            channel_names_to_modify = [raw.ch_names[i] for i in eeg_channels[:]]

            for key, ch_id in enumerate(eeg_channels):
                logger.debug(
                    f"Applying ANC to Channel {ch_id}:{channel_names_to_modify[key]}"
                )
                raw._data[ch_id] = self._anc(
                    raw._data[ch_id], self._eeg.estimated_noise[key], ch_id
                )

        except Exception as ex:
            logger.exception("An exception occured while applying ANC", ex)

    def align_triggers(
        self,
        ref_trigger,
        ref_channel=None,
        save=False,
        search_window=None,
        upsample=True,
    ):
        """
        Aligns slices based on a reference trigger.

        Parameters:
            ref_trigger (int): The reference trigger.

        Returns:
            None
        """
        f = self._facet
        logger.debug("Aligning triggers")
        if search_window is None:
            search_window = 3 * self._eeg.upsampling_factor
        try:
            needed_to_upsample = False
            eeg_channels = mne.pick_types(
                self._eeg.mne_raw.info,
                meg=False,
                eeg=True,
                stim=False,
                eog=False,
                exclude="bads",
            )
            if ref_channel is None:
                ref_channel = eeg_channels[0]
            if not self._eeg.mne_raw.preload:
                f = self._facet.create_facet_with_channel_picks([ref_channel])
                raw = f._eeg.mne_raw
                ref_channel = 0
            else:
                raw = self._eeg.mne_raw
            if (
                self._eeg.mne_raw.info["sfreq"] == self._eeg.mne_raw_orig.info["sfreq"]
                and upsample
            ):
                logger.debug("Data is not upsampled. Upsampling data")
                if self._eeg.mne_raw.preload:
                    f = f.create_facet_with_channel_picks(
                        [ref_channel], raw=self._eeg.mne_raw_orig
                    )
                    raw = f._eeg.mne_raw
                f.upsample()
                needed_to_upsample = True
            trigger_positions = f._eeg.loaded_triggers
            smin, smax = f._eeg.smin, f._eeg.smax
            chosen_artifact = raw._data[ref_channel][
                trigger_positions[ref_trigger]
                + smin : trigger_positions[ref_trigger]
                + smax
            ]
            # Iterate through all triggers and shift the trigger positions
            for key, val in enumerate(trigger_positions):
                if key == ref_trigger:
                    continue
                # Shift the trigger position
                trigger_positions[key] = f._correction._align_trigger(
                    val, chosen_artifact, search_window, ref_channel, raw
                )
            if needed_to_upsample:
                self._eeg._loaded_triggers_upsampled = trigger_positions
                trigger_positions = [
                    int(val / self._eeg.upsampling_factor) for val in trigger_positions
                ]
            # Update the trigger positions
            self._eeg.loaded_triggers = trigger_positions[:]
            if needed_to_upsample:
                del trigger_positions
                del raw
                del f._eeg.mne_raw
                del f

            # Update related attributes
            self._facet._analysis._derive_art_length()
            self._eeg._tmax = self._eeg._tmin + self._eeg.artifact_duration
            if save:
                # replace the triggers as events in the raw object
                self._eeg.mne_raw.set_annotations(
                    mne.Annotations(
                        onset=np.array(self._eeg.loaded_triggers)
                        / self._eeg.mne_raw.info["sfreq"],
                        duration=np.zeros(len(self._eeg.loaded_triggers)),
                        description=["Aligned_Trigger"]
                        * len(self._eeg.loaded_triggers),
                    )
                )
                self._eeg.all_events = self._eeg.triggers_as_events
                # get stimchannel index
                stim_channel = mne.pick_types(
                    self._eeg.mne_raw.info, meg=False, eeg=False, stim=True, eog=False
                )[0]
                if self._eeg.mne_raw.preload and len(stim_channel) > 0:
                    # update the stimulus channel with the new triggers
                    self._eeg.mne_raw._data[stim_channel] = np.zeros(
                        self._eeg.mne_raw._data[stim_channel].shape
                    )
                    for idx, val in enumerate(self._eeg.loaded_triggers):
                        self._eeg.mne_raw._data[stim_channel][val] = 1

        except Exception as ex:
            logger.exception("An exception occured while aligning triggers", ex)

    def _align_trigger(
        self, trigger_pos, reference, search_window, ref_channel=0, raw=None
    ):
        """
        Aligns the trigger based on a reference artifact.

        Parameters:
            template (numpy.ndarray): The template artifact.
            reference (numpy.ndarray): The reference artifact.

        Returns:
            int: new trigger position
        """
        if raw is None:
            raw = self._eeg.mne_raw
        smin, smax = self._eeg.smin, self._eeg.smax
        current_artifact = raw._data[ref_channel][
            trigger_pos + smin : trigger_pos + smax + search_window
        ]
        # Calculate the cross correlation
        max_corr = self._find_max_cross_correlation(
            current_artifact, reference, search_window
        )
        shift = max_corr - search_window
        # Shift the trigger position
        return trigger_pos + shift

    def apply_PCA(self, n_components=0.95):  # TODO: Use MNE's PCA implementation
        # apply pca to each channel
        logger.debug("applying PCA")
        try:
            raw = self._eeg.mne_raw
            eeg_channels = mne.pick_types(
                raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
            )
            channel_names_to_modify = [raw.ch_names[i] for i in eeg_channels[:]]

            s_acq_start, s_acq_end = self._eeg.s_acq_start, self._eeg.s_acq_end

            for key, ch_id in enumerate(eeg_channels):
                logger.debug(
                    f"Applying PCA to Channel {ch_id}:{channel_names_to_modify[key]}"
                )
                res = self._calc_pca_residuals(ch_id, n_components=n_components)
                raw._data[ch_id][s_acq_start:s_acq_end] -= res
                self._eeg.estimated_noise[ch_id][s_acq_start:s_acq_end] += res

        except Exception as ex:
            logger.exception("An exception occured while applying PCA", ex)

    def _calc_pca_residuals(
        self, ch_id, n_components=0.95
    ):  # TODO: Use MNE's PCA implementation

        s_acq_start, s_acq_end = self._eeg.s_acq_start, self._eeg.s_acq_end
        ch_d = self._eeg.mne_raw._data[ch_id]
        ch_d_acq = ch_d[s_acq_start:s_acq_end]

        # Calculate the PCA residuals
        if np.intersect1d(self._eeg.obs_exclude_channels, ch_id).size == 0 and (
            n_components != 0
        ):
            Ipca = filtfilt(self._eeg.obs_hp_filter_weights, 1, ch_d_acq)
            epochs = self._calc_pca(Ipca)
            # paste the epochs back to the original data
            fitted_res = np.zeros(len(ch_d_acq))
            for i in range(0, len(epochs)):
                start_pos = self._eeg.loaded_triggers[i] - s_acq_start + self._eeg.smin
                end_pos = self._eeg.loaded_triggers[i] - s_acq_start + self._eeg.smax
                
                # Ensure start position is valid
                if start_pos < 0:
                    continue  # Skip this epoch if it starts before the data
                
                # Adjust end position if it exceeds data length
                if end_pos > len(ch_d_acq):
                    epoch_length = len(ch_d_acq) - start_pos
                    if epoch_length <= 0:
                        continue  # Skip if no valid data points
                    fitted_res[start_pos:len(ch_d_acq)] = epochs[i][:epoch_length]
                else:
                    # Full epoch fits
                    fitted_res[start_pos:end_pos] = epochs[i]

        else:
            fitted_res = np.zeros(len(ch_d_acq))

        # fitted_res now holds a column vector with residuals
        return fitted_res.flatten()

    def _calc_pca(self, ch_d, n_components=0.95):  # TODO: Use MNE's PCA implementation

        epochs = split_vector(
            ch_d,
            np.array(self._eeg.loaded_triggers)
            + self._eeg.smin
            - self._eeg.s_acq_start,
            self._eeg.artifact_length,
        )
        # Now transpose the epochs matrix
        epochs = epochs.T
        # Now standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(epochs)  # Standardize the data
        # Now apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Now reconstruct the data
        X_cleaned = pca.inverse_transform(X_pca)
        X_cleaned = scaler.inverse_transform(X_cleaned)
        residuals = epochs - X_cleaned

        return residuals.T

    def align_subsample(self, ref_trigger):  # WIP
        """
        Aligns subsamples based on a reference trigger.

        Parameters:
            ref_trigger (int): The reference trigger.

        Returns:
            None
        """
        logger.info("Aligning subsamples")
        eeg_channels = mne.pick_types(
            self._eeg.mne_raw.info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            exclude="bads",
        )
        # Call _align_subsample for each channel
        for ch_id in eeg_channels:
            logger.debug(
                f"Aligning subsamples for Channel {ch_id}:{self._eeg.mne_raw.ch_names[ch_id]}"
            )
            self._align_subsample(ch_id, ref_trigger)
        return

    def _align_subsample(self, ch_id, ref_trigger):  # WARNING: Not working yet
        """
        Aligns subsamples based on a reference trigger for a single channel.
        """
        # Maximum distance between triggers
        max_trig_dist = np.max(np.diff(self._eeg.loaded_triggers))
        num_samples = max_trig_dist + 20
        acq_start, acq_end = self._eeg.s_acq_start, self._eeg.s_acq_end
        raeeg_acq = self._eeg.mne_raw._data[ch_id][acq_start:acq_end]
        smin = self._eeg.smin

        # Phase shift
        shift_angles = (
            np.arange(1, num_samples + 1) - np.floor(num_samples / 2) + 1
        ) / num_samples

        # Initial filter setup if conditions are met
        if self.sub_sample_alignment is None:
            logger.debug("Initial filter setup")
            if self._eeg.ssa_hp_frequency and self._eeg.ssa_hp_frequency > 0:
                nyq = 0.5 * self._eeg.mne_raw.info["sfreq"]
                f = [
                    0,
                    (self._eeg.ssa_hp_frequency * 0.9)
                    / (nyq * self._eeg.upsampling_factor),
                    (self._eeg.ssa_hp_frequency * 1.1)
                    / (nyq * self._eeg.upsampling_factor),
                    1,
                ]
                a = [0, 0, 1, 1]
                fw = firls(101, f, a)

                # HPEEG signal filtering
                # Original MATLAB code used fftfilt which is equivalent to scipy's fftconvolve in "same" mode
                hpeeg = fftconvolve(raeeg_acq, fw, mode="same")
                hpeeg = np.concatenate(
                    [hpeeg[100:], np.zeros(100)]
                )  # Adjusting the shift
            else:
                hpeeg = raeeg_acq

            eeg_matrix = split_vector(
                hpeeg,
                np.array(self._eeg.loaded_triggers) + smin - 10 - acq_start,
                num_samples,
            )
            eeg_ref = eeg_matrix[ref_trigger, :]

            self.sub_sample_alignment = np.zeros(self._eeg.count_triggers)
            corrs = np.zeros((self._eeg.count_triggers, 20))
            shifts = np.zeros((self._eeg.count_triggers, 20))

            # Loop over every epoch, skipping the reference epoch
            for epoch in set(range(1, self._eeg.count_triggers + 1)) - {ref_trigger}:
                eeg_m = eeg_matrix[
                    epoch - 1, :
                ]  # Adjusting index for Python's 0-based indexing
                fft_m = np.fft.fftshift(np.fft.fft(eeg_m))
                shift_l, shift_m, shift_r = -1, 0, 1
                fft_l = fft_m * np.exp(-1j * 2 * np.pi * shift_angles * shift_l)
                fft_r = fft_m * np.exp(-1j * 2 * np.pi * shift_angles * shift_r)
                eeg_l = np.real(np.fft.ifft(np.fft.ifftshift(fft_l)))
                eeg_r = np.real(np.fft.ifft(np.fft.ifftshift(fft_r)))
                corr_l = self._compare(eeg_ref, eeg_l)
                corr_m = self._compare(eeg_ref, eeg_m)
                corr_r = self._compare(eeg_ref, eeg_r)

                fft_ori = fft_m  # Save original FFT for later IFFT

                # Iterative optimization
                for iteration in range(15):
                    corrs[epoch - 1, iteration] = corr_m
                    shifts[epoch - 1, iteration] = shift_m

                    if corr_l > corr_r:
                        corr_r, eeg_r, fft_r, shift_r = corr_m, eeg_m, fft_m, shift_m
                    else:
                        corr_l, eeg_l, fft_l, shift_l = corr_m, eeg_m, fft_m, shift_m

                    shift_m = (shift_l + shift_r) / 2
                    fft_m = fft_ori * np.exp(-1j * 2 * np.pi * shift_angles * shift_m)
                    eeg_m = np.real(np.fft.ifft(np.fft.ifftshift(fft_m)))
                    corr_m = self._compare(eeg_ref, eeg_m)

                self.sub_sample_alignment[epoch - 1] = shift_m
                eeg_matrix[epoch - 1, :] = eeg_m

        # logger.debug("Applying subsample alignment")
        # Assuming split_vector and other utility methods are defined similarly to the MATLAB version
        eeg_matrix = split_vector(
            self._eeg.mne_raw._data[ch_id],
            np.array(self._eeg.loaded_triggers) + smin - 10,
            num_samples,
        )

        # Apply calculated alignments to all epochs
        for epoch in set(range(1, self._eeg.count_triggers + 1)) - {ref_trigger}:
            eeg = eeg_matrix[epoch - 1, :]
            fft = np.fft.fftshift(np.fft.fft(eeg))
            fft *= np.exp(
                -1j * 2 * np.pi * shift_angles * self.sub_sample_alignment[epoch - 1]
            )
            eeg = np.real(np.fft.ifft(np.fft.ifftshift(fft)))
            eeg_matrix[epoch - 1, :] = eeg

        # Joining epochs back into the main signal
        for tr_id, tr_pos in enumerate(self._eeg.loaded_triggers):
            start_index = tr_pos + smin
            end_index = start_index + self._eeg.artifact_length
            self._eeg.mne_raw._data[ch_id][start_index:end_index] = eeg_matrix[
                tr_id, 10 : 10 + self._eeg.artifact_length
            ]

    def _compare(self, ref, arg):
        """
        Compare the reference data to the shifted data.
        The larger the better. This uses a simple sum of squared differences for comparison.
        """
        # result = correlate(ref, arg, mode='valid')[0] # If you prefer cross-correlation
        result = -np.sum((ref - arg) ** 2)
        return result

    def _anc(self, EEG, Noise, ch_id):
        """
        Internal method for Adaptive Noise Cancellation.

        Parameters:
            EEG (numpy.ndarray): The EEG data to be cleaned.
            Noise (numpy.ndarray): The noise data used as reference for ANC.

        Returns:
            numpy.ndarray: The cleaned EEG data.
        """
        s_acq_start, s_acq_end = self._eeg.s_acq_start, self._eeg.s_acq_end
        Reference = Noise[s_acq_start:s_acq_end]
        # plt.plot(Reference[0:self._eeg.artifact_length])
        tmpd = filtfilt(self._eeg.anc_hp_filter_weights, 1, EEG, axis=0, padtype="odd")
        Data = tmpd[s_acq_start:s_acq_end].astype(float)
        Alpha = np.sum(Data * Reference) / np.sum(Reference * Reference)
        Reference = (Alpha * Reference).astype(float)
        mu = float(0.05 / (self._eeg.anc_filter_order * np.var(Reference)))

        # Use the fastranc function for adaptive noise cancellation
        _, FilteredNoise = fastr_anc(Reference, Data, self._eeg.anc_filter_order, mu)
        if np.isinf(np.max(FilteredNoise)):
            logger.error("Warning: ANC failed, skipping ANC.")
        else:
            EEG[s_acq_start:s_acq_end] -= FilteredNoise
            self._eeg.estimated_noise[ch_id][s_acq_start:s_acq_end] += FilteredNoise

        return EEG

    def _align_triggers_averaged_artifacts(
        self, ch_d, avg_artifact, search_window=None
    ):
        """
        Aligns the triggers based on the averaged artifacts.

        Parameters:
            ch_d (numpy.ndarray): The EEG data.
            avg_artifact_matrix_numpy (numpy.ndarray): The average artifact matrix.
            search_window (int, optional): The search window. Defaults to None.

        Returns:
            list: The new triggers.
        """
        if search_window is None:
            search_window = 3 * self._eeg.upsampling_factor
        new_triggers = []
        smin, smax = self._eeg.smin, self._eeg.smax
        for i in range(self._eeg.count_triggers):
            base = ch_d[
                self._eeg.loaded_triggers[i]
                + smin : self._eeg.loaded_triggers[i]
                + smax
                + search_window
            ]
            Beta = self._find_max_cross_correlation(
                base, avg_artifact[i, :], search_window
            )
            shift = Beta - search_window
            new_triggers.append(self._eeg.loaded_triggers[i] + shift)
        return new_triggers

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
        if self._eeg._loaded_triggers_upsampled is not None:
            self._eeg.loaded_triggers = self._eeg._loaded_triggers_upsampled
            self._eeg._loaded_triggers_upsampled = None
        return

    def filter(self, l_freq=None, h_freq=None):
        """
        Applies a bandpass filter to the raw EEG data.

        Parameters:
            l_freq (float, optional): The lower cutoff frequency for the bandpass filter. If None, no lower cutoff is applied.
            h_freq (float, optional): The higher cutoff frequency for the bandpass filter. If None, no higher cutoff is applied.
        """

        logger.debug(f"Applying filter with l_freq={l_freq} and h_freq={h_freq}")
        # performant check if the estimated noise is all zeros with any
        if np.any(self._eeg.estimated_noise):
            # Apply highpass filter

            # TODO: Consider changing estimated_noise to a mne object, to avoid copying
            noise_raw = self._eeg.mne_raw.copy()
            noise_raw._data = self._eeg.estimated_noise
            self._eeg.estimated_noise = noise_raw.filter(
                l_freq=l_freq, h_freq=h_freq
            )._data.copy()
            # unload noise_raw
            noise_raw = None
        self._eeg.mne_raw.filter(l_freq=l_freq, h_freq=h_freq)

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
        self.resample_data(
            self._eeg.mne_raw.info["sfreq"] * self._eeg.upsampling_factor
        )

    def _downsample_data(self):
        """
        Resamples (downsamples) the data by reducing the sampling frequency.

        This method resamples the raw EEG data by dividing the sampling frequency
        by the upsampling factor. The resulting downsampled data is stored in the
        `_eeg.mne_raw` attribute.

        Note:
            The `_eeg` attribute should be initialized before calling this method.
        """
        self.resample_data(
            self._eeg.mne_raw.info["sfreq"] / self._eeg.upsampling_factor
        )

    def remove_nans(self):
        """
        Removes NaN values from the EEG data.

        This method removes NaN values from the EEG data by replacing them with the mean value
        of the data.

        Returns:
            None
        """
        self._eeg.mne_raw._data = np.nan_to_num(
            self._eeg.mne_raw._data, nan=np.nanmean(self._eeg.mne_raw._data)
        )

    def resample_data(self, sfreq):
        """
        Resamples the EEG data to a specified sampling frequency.

        Parameters:
            sfreq (float): The target sampling frequency.
        """
        sfreq_old = self._eeg.mne_raw.info["sfreq"]
        # TODO: Consider changing estimated_noise to a mne object, to avoid copying
        noise_raw = self._eeg.mne_raw.copy()
        self._eeg.mne_raw.resample(sfreq=sfreq)
        # performant check if the estimated noise is all zeros with any
        if not np.any(self._eeg.estimated_noise):
            self._eeg.estimated_noise = np.zeros(self._eeg.mne_raw._data.shape)
        else:
            noise_raw._data = self._eeg.estimated_noise
            self._eeg.estimated_noise = noise_raw.resample(sfreq=sfreq)._data.copy()
            # unload noise_raw
            noise_raw = None
        if self._eeg.loaded_triggers is None:
            return
        # update the trigger positions
        self._eeg.loaded_triggers = [
            int(trigger * (sfreq / sfreq_old)) for trigger in self._eeg.loaded_triggers
        ]

        self._facet._analysis.derive_parameters()

    def _find_max_cross_correlation(self, base, compare, search_window):
        """
        Finds the maximum cross correlation between two signals.

        Parameters:
            base (numpy.ndarray): The base signal.
            compare (numpy.ndarray): The signal to compare.
            pos (int): The position of the signal.
            pre_pos (int): The previous position of the signal.
            search_window (int): The search window.

        Returns:
            int: The maximum cross correlation.
        """
        # Calculate the cross correlation
        # Reduce positions to a window of 3 * _eeg.mne_raw.upsampling_factor

        corr = crosscorrelation(base, compare, search_window)
        # Find the maximum of the cross correlation
        max_corr = np.argmax(corr)
        return max_corr

    def apply_per_channel(self, function):
        """
        Applies a function to each channel. It loads the data for the current channel. The functions is called. Finally the data is saved and unloaded and the next channel is loaded.

        Parameters:
            function (function): The function to apply to each channel.

        Returns:
            None
        """
        from facet.facet import facet

        # check if raw is preloaded
        if hasattr(self._eeg.mne_raw, "_data"):
            logger.warning(
                "Raw data is already loaded. Therefore no memory benefits can be achieved by applying the function per channel."
            )

        raw = self._eeg.mne_raw
        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        # create a list of channels
        data_list = []
        subsample_alignment_conf = None

        for ch_id in eeg_channels:
            # create raw with only one channel
            # load data
            logger.info(f"Applying function to Channel {ch_id}:{raw.ch_names[ch_id]}")
            one_channel_facet_obj = self._facet.create_facet_with_channel_picks(ch_id)
            one_channel_facet_obj._correction.sub_sample_alignment = (
                subsample_alignment_conf
            )
            function(one_channel_facet_obj)
            if one_channel_facet_obj._correction.sub_sample_alignment is not None:
                subsample_alignment_conf = (
                    one_channel_facet_obj._correction.sub_sample_alignment.copy()
                )
            # append data to list
            one_channel_data = one_channel_facet_obj._eeg.mne_raw._data[0]
            data_list.append(one_channel_data)
            # unload data
            del one_channel_facet_obj
        # Now load the data and replace the data
        raw.load_data()
        self._eeg.estimated_noise = np.zeros(raw._data.shape)
        for i, ch_id in enumerate(eeg_channels):
            raw._data[ch_id, : data_list[0].shape[0]] = data_list[i]
        # delete data_list
        del data_list
        return
