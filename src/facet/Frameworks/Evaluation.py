"""
Evaulation Framework Module

This module contains the Evaluation_Framework class, which is used to evaluate EEG data.

Author: Janik Michael Müller
Date: 15.02.2024
Version: 1.0
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
from loguru import logger


class Evaluation_Framework:
    def __init__(self, facet):
        """
        Initializes the Evaluation_Framework class.

        Parameters:
            facet (facet class instance): An instance of the facet class.
        """
        self._eeg_eval_dict_list = []
        self._facet = facet
        return

    def add_to_evaluate(self, eeg, start_time=None, end_time=None, name=None):
        """
        Add EEG data to the evaluation list.

        Parameters:
            eeg (facet.EEG_obj): The EEG data to be evaluated.
            start_time (float, optional): Start time of the data to be evaluated.
            end_time (float, optional): End time of the data to be evaluated.
            name (str, optional): Name of the evaluation dataset.

        Returns:
            None
        """
        if not end_time:
            end_time = (
                eeg.time_last_artifact_end
                if eeg.time_last_artifact_end
                else eeg.data_time_end
            )
        if not start_time:
            start_time = (
                eeg.time_first_artifact_start
                if eeg.time_first_artifact_start
                else eeg.data_time_start
            )
        raw = eeg.mne_raw
        logger.debug("Channels that will be evaluated: " + str(raw.ch_names))

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
        cropped_mne_raw = self._crop(
            raw=eeg.mne_raw, tmin=start_time, tmax=end_time
        ).pick(channels_to_keep)
        ref_mne_raw = self._cutout(
            raw=eeg.mne_raw, tmin=start_time, tmax=end_time
        ).pick(channels_to_keep)
        artifact_raw_reference_raw_dict = {
            "eeg": eeg,
            "raw": cropped_mne_raw,
            "ref": ref_mne_raw,
            "raw_orig": eeg.mne_raw_orig,
            "name": name,
        }

        self._eeg_eval_dict_list.append(artifact_raw_reference_raw_dict)

        return

    def _crop(self, raw, tmin, tmax):
        """
        Crop the raw EEG data to a specified time window.

        Parameters:
            raw (mne.io.Raw): The raw EEG data.
            tmin (float): The start time of the crop window.
            tmax (float): The end time of the crop window.

        Returns:
            mne.io.Raw: The cropped raw EEG data.
        """
        # check if tmax is in the data
        if tmax > raw.times[-1]:
            tmax = raw.times[-1]
        # ensure that tmin is smaller than tmax
        if tmin >= tmax:
            return raw.copy().crop(tmin=0, tmax=0)
        return raw.copy().crop(tmin=tmin, tmax=tmax)

    def _cutout(self, raw, tmin, tmax):
        """
        Cut out a specified time window from the raw EEG data.

        Parameters:
            raw (mne.io.Raw): The raw EEG data.
            tmin (float): The start time of the window to cut out.
            tmax (float): The end time of the window to cut out.

        Returns:
            mne.io.Raw: The raw EEG data with the specified window removed.
        """
        # check if tmax is in the data
        if tmax > raw.times[-1]:
            tmax = raw.times[-1]

        first_part = raw.copy().crop(tmax=tmin)

        second_part = raw.copy().crop(tmin=tmax)

        first_part.append(second_part)
        return first_part

    def evaluate(self, plot=True, measures=[]):
        """
        Evaluate the EEG datasets based on specified measures.

        Parameters:
            plot (bool, optional): Whether to plot the results. Defaults to True.
            measures (list, optional): A list of measures to calculate. Defaults to an empty list.

        Returns:
            list: A list of dictionaries containing the results of the evaluation for each measure.
        """
        results = []
        if "SNR" in measures:
            results.append(
                {"Measure": "SNR", "Values": self.evaluate_SNR(), "Unit": "dB"}
            )
        if "RMS" in measures:
            results.append(
                {
                    "Measure": "RMS Uncorrected to Corrected",
                    "Values": self.evaluate_RMS_corrected_ratio(),
                    "Unit": "Ratio",
                }
            )
        if "RMS2" in measures:
            results.append(
                {
                    "Measure": "RMS Corrected to Unimpaired",
                    "Values": self.evaluate_RMS_residual_ratio(),
                    "Unit": "Ratio",
                }
            )
        if "MEDIAN" in measures:
            results.append(
                {
                    "Measure": "MEDIAN",
                    "Values": self.calculate_median_imaging_artifact(),
                    "Unit": "V",
                }
            )
        if plot:
            self.plot(results)
        return results

    # Plot all results with matplotlib

    def plot(self, results):
        """
        Plot the evaluation results.

        Parameters:
            results (list): A list of dictionaries containing the evaluation results.

        Returns:
            int: 0 if successful.
        """

        # Determine the number of subplots based on the number of measures
        num_subplots = len(results)

        # Create subplots with 1 row and as many columns as there are measures
        fig, axs = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))

        # If there is only one measure, axs is not returned as a list
        if num_subplots == 1:
            axs = [axs]

        # Fill each subplot
        for ax, result in zip(axs, results):
            bars = ax.bar(range(len(result["Values"])), result["Values"])
            ax.set_title(result["Measure"])
            ax.set_ylabel(
                result["Measure"] + " in " + (result["Unit"] if result["Unit"] else "")
            )
            # Replace with your labels
            x_labels = [
                eval_eeg_ref_dict["name"]
                for eval_eeg_ref_dict in self._eeg_eval_dict_list
            ]
            ax.set_xticks(range(len(result["Values"])))
            ax.set_xticklabels(x_labels, rotation=45)

        # Display the entire window with all subplots
        plt.tight_layout()  # Used to ensure that the subplots do not overlap
        plt.show()

        return 0

    def evaluate_RMS_corrected_ratio(self):
        """
        Calculate the ratio of the Root Mean Square (RMS) values before and after correction.

        Returns:
            list: A list containing the RMS ratios for each evaluated dataset.
        """
        if not self._eeg_eval_dict_list:
            logger.error(
                "Please set at least one EEG dataset and crop the EEG to evaluate before calculating RMS."
            )
            return
        results = []
        for mnedict in self._eeg_eval_dict_list:
            # Extracting the data
            data_corrected = mnedict["raw"].get_data()
            data_uncorrected = mnedict["raw_orig"].get_data()

            # TODO: Bugfix for different number of channels
            if data_corrected.shape[0] != data_uncorrected.shape[0]:
                data_uncorrected = data_uncorrected[: data_corrected.shape[0], :]

            # Calculate RMS
            rms_corrected = np.sqrt(np.mean(data_corrected**2, axis=1))
            rms_uncorrected = np.sqrt(np.mean(data_uncorrected**2, axis=1))

            # Calculate Ratio
            rms = rms_uncorrected / rms_corrected
            np.median(rms)
            results.append(np.median(rms))

        return results

    def evaluate_RMS_residual_ratio(self):
        """
        Calculate the ratio of the Root Mean Square (RMS) values of the corrected data to the unimpaired reference.

        Returns:
            list: A list containing the RMS ratios for each evaluated dataset.
        """
        if not self._eeg_eval_dict_list:
            logger.error(
                "Please set at least one EEG dataset and crop the EEG to evaluate before calculating RMS."
            )
            return
        results = []
        for mnedict in self._eeg_eval_dict_list:
            # Extracting the data
            data_corrected = mnedict["raw"].get_data()
            data_ref = mnedict["ref"].get_data()

            # Calculate RMS
            rms_corrected = np.sqrt(np.mean(data_corrected**2, axis=1))
            rms_ref = np.sqrt(np.mean(data_ref**2, axis=1))

            # Calculate Ratio
            rms = rms_corrected / rms_ref
            np.median(rms)
            results.append(np.median(rms))

        return results

    def calculate_median_imaging_artifact(self):
        """
        Calculate the median imaging artifact value for each evaluated EEG dataset.

        Returns:
            list: A list containing the median imaging artifact values for each dataset.
        """
        if not hasattr(self, "_eeg_eval_dict_list") or not self._eeg_eval_dict_list:
            logger.error("eeg_list is not set or empty.")
            return

        results = []

        for mne_dict in self._eeg_eval_dict_list:
            _eeg = mne_dict["eeg"]
            if _eeg.mne_raw is None:
                logger.error("EEG dataset is not set for this mne_dict.")
                continue

            # Create epochs around the artifact triggers
            events = np.column_stack(
                (
                    _eeg.loaded_triggers,
                    np.zeros_like(_eeg.loaded_triggers),
                    np.ones_like(_eeg.loaded_triggers),
                )
            )
            tmin = _eeg.get_tmin()  # Start time before the event
            tmax = _eeg.get_tmax()  # End time after the event
            baseline = None  # No baseline correction
            picks = mne.pick_types(
                _eeg.mne_raw.info,
                meg=False,
                eeg=True,
                stim=False,
                eog=False,
                exclude="bads",
            )

            epochs = mne.Epochs(
                _eeg.mne_raw,
                events=events,
                tmin=tmin,
                tmax=tmax,
                proj=True,
                reject=None,
                picks=picks,
                baseline=baseline,
                preload=True,
            )
            # Calculate the peak-to-peak value for each epoch and channel
            p2p_values_per_epoch = [
                np.ptp(epoch, axis=1) for epoch in epochs.get_data()
            ]

            # Calculate the mean peak-to-peak value per epoch across all channels
            mean_p2p_per_epoch = [
                np.mean(epoch_p2p) for epoch_p2p in p2p_values_per_epoch
            ]

            # Calculate the median of these mean values
            vmed = np.median(mean_p2p_per_epoch)

            results.append(vmed)

        return results

    def evaluate_SNR(self):
        """
        Calculate the Signal-to-Noise Ratio (SNR) for each evaluated EEG dataset.

        Returns:
            list: A list containing the SNR values for each dataset.
        """
        if not self._eeg_eval_dict_list:
            logger.error(
                "Please set both EEG datasets and crop the EEG to evaluate before calculating SNR."
            )
            return
        results = []
        for mnedict in self._eeg_eval_dict_list:
            # Extracting the data
            data_to_evaluate = mnedict["raw"].get_data()
            data_reference = mnedict["ref"].get_data()

            # Calculate power of the signal
            power_corrected = np.var(data_to_evaluate, axis=1)
            power_without = np.var(data_reference, axis=1)

            # Calculate power of the residual (noise)
            power_residual = power_corrected - power_without

            # Calculate SNR
            snr = np.abs(power_without / power_residual)

            results.append(np.mean(snr))

        return results
