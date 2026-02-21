"""
Evaulation framework Module

This module contains the EvaluationFramework class, which is used to evaluate EEG data.

Author: Janik Michael Müller
Date: 15.02.2024
Version: 1.0
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
from loguru import logger
from ..utils.i18n import get_translation


class EvaluationFramework:
    def __init__(self, facet):
        """
        Initializes the EvaluationFramework class.

        Parameters:
            facet (facet class instance): An instance of the facet class.
        """
        self._facet = facet
        return

    def evaluate(
        self, eeg, start_time=None, end_time=None, ref_start_time=None, ref_end_time=None, name=None, plot=True, measures=[]
    ):
        """
        Evaluate the EEG data based on specified measures.

        Parameters:
            eeg (facet.eeg_obj): The EEG data to be evaluated.
            start_time (float, optional): Start time of the data to be evaluated.
            end_time (float, optional): End time of the data to be evaluated.
            ref_start_time (float, optional): Start time of the clean reference interval.
                If provided along with ref_end_time, this interval will be used as the reference
                instead of the default behavior (everything except the artifact interval).
            ref_end_time (float, optional): End time of the clean reference interval.
                If provided along with ref_start_time, this interval will be used as the reference
                instead of the default behavior (everything except the artifact interval).
            name (str, optional): Name of the evaluation dataset.
            plot (bool, optional): Whether to plot the results. Defaults to True.
            measures (list, optional): A list of measures to calculate. Defaults to an empty list.

        Returns:
            dict: A dictionary containing the results of the evaluation for each measure.
        """
        if not start_time:
            start_time = eeg.time_acq_start
        if not end_time:
            end_time = eeg.time_acq_end
        raw = eeg.mne_raw
        logger.debug("Channels that will be evaluated: " + str(raw.ch_names))
        logger.debug("Signal interval: " + str(start_time) + " - " + str(end_time))
        logger.debug("Reference interval: " + str(ref_start_time) + " - " + str(ref_end_time))

        eeg_channels = mne.pick_types(
            raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )
        channels_to_keep = [raw.ch_names[i] for i in eeg_channels[:]]
        cropped_mne_raw = self._crop(
            raw=eeg.mne_raw, tmin=start_time, tmax=end_time
        ).pick(channels_to_keep)
        
        # Use specified reference interval if provided, otherwise use cutout (default behavior)
        if ref_start_time is not None and ref_end_time is not None:
            ref_mne_raw = self._crop(
                raw=eeg.mne_raw, tmin=ref_start_time, tmax=ref_end_time
            ).pick(channels_to_keep)
        else:
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

        results = {"name": name}
        if "SNR" in measures:
            results["SNR"] = self.evaluate_SNR(artifact_raw_reference_raw_dict)
        if "RMS" in measures:
            results["RMS"] = self.evaluate_RMS_corrected_ratio(
                artifact_raw_reference_raw_dict
            )
        if "RMS2" in measures:
            results["RMS2"] = self.evaluate_RMS_residual_ratio(
                artifact_raw_reference_raw_dict
            )
        if "MEDIAN" in measures:
            results["MEDIAN"] = self.calculate_median_imaging_artifact(
                artifact_raw_reference_raw_dict
            )

        if plot:
            self.plot([results], plot_measures=measures)
        return results

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

    def evaluate_RMS_corrected_ratio(self, mnedict):
        """
        Calculate the ratio of the Root Mean Square (RMS) values before and after correction.

        Parameters:
            mnedict (dict): A dictionary containing the EEG data to be evaluated.

        Returns:
            float: The RMS ratio for the evaluated dataset.
        """
        # Extracting the data
        data_corrected = mnedict["raw"].get_data()
        data_uncorrected = mnedict["raw_orig"].get_data()

        # TODO: Bugfix for different number of channels
        if data_corrected.shape[0] != data_uncorrected.shape[0]:
            data_uncorrected = data_uncorrected[: data_corrected.shape[0], :]

        # Calculate RMS
        rms_corrected = np.sqrt(np.mean(data_corrected**2, axis=1))
        rms_uncorrected = np.sqrt(np.mean(data_uncorrected**2, axis=1))

        # Calculate Ratio (add epsilon to prevent division by zero)
        rms = rms_uncorrected / (rms_corrected + 1e-10)
        return np.median(rms)

    def evaluate_RMS_residual_ratio(self, mnedict):
        """
        Calculate the ratio of the Root Mean Square (RMS) values of the corrected data to the unimpaired reference.

        Parameters:
            mnedict (dict): A dictionary containing the EEG data to be evaluated.

        Returns:
            float: The RMS ratio for the evaluated dataset.
        """
        # Extracting the data
        data_corrected = mnedict["raw"].get_data()
        data_ref = mnedict["ref"].get_data()

        # Calculate RMS
        rms_corrected = np.sqrt(np.mean(data_corrected**2, axis=1))
        rms_ref = np.sqrt(np.mean(data_ref**2, axis=1))

        # Calculate Ratio (add epsilon to prevent division by zero)
        rms = rms_corrected / (rms_ref + 1e-10)
        return np.median(rms)

    def calculate_median_imaging_artifact(self, mnedict):
        """
        Calculate the median imaging artifact value for the evaluated EEG dataset.

        Parameters:
            mnedict (dict): A dictionary containing the EEG data to be evaluated.

        Returns:
            float: The median imaging artifact value for the dataset.
        """
        _eeg = mnedict["eeg"]
        if _eeg.mne_raw is None:
            logger.error("EEG dataset is not set for this mne_dict.")
            return

        # Create epochs around the artifact triggers
        events = np.column_stack(
            (
                _eeg.loaded_triggers,
                np.zeros_like(_eeg.loaded_triggers),
                np.ones_like(_eeg.loaded_triggers),
            )
        )
        tmin = _eeg.tmin  # Start time before the event
        tmax = _eeg.tmax  # End time after the event
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
        p2p_values_per_epoch = [np.ptp(epoch, axis=1) for epoch in epochs.get_data()]

        # Calculate the mean peak-to-peak value per epoch across all channels
        mean_p2p_per_epoch = [np.mean(epoch_p2p) for epoch_p2p in p2p_values_per_epoch]

        # Calculate the median of these mean values
        vmed = np.median(mean_p2p_per_epoch)

        return vmed

    def evaluate_SNR(self, mnedict):
        """
        Calculate the Signal-to-Noise Ratio (SNR) for the evaluated EEG dataset.

        Parameters:
            mnedict (dict): A dictionary containing the EEG data to be evaluated.

        Returns:
            float: The SNR value for the dataset.
        """
        # Extracting the data
        data_to_evaluate = mnedict["raw"].get_data()
        data_reference = mnedict["ref"].get_data()

        # Calculate power of the signal
        power_corrected = np.var(data_to_evaluate, axis=1)
        power_without = np.var(data_reference, axis=1)

        # Calculate power of the residual (noise)
        power_residual = power_corrected - power_without

        # Calculate SNR (add epsilon to prevent division by zero)
        snr = np.abs(power_without / (power_residual + 1e-10))

        return np.mean(snr)

    # Plot all results with matplotlib
    def plot(self, results, plot_measures=["SNR"]):
        """
        Plot the evaluation results.

        Parameters:
            results (list): A list of dictionaries containing the evaluation results.

        Returns:
            int: 0 if successful.
        """
        # Create subplots with 1 row and as many columns as there are measures
        num_subplots = len(plot_measures)
        fig, axs = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 5))

        # If there is only one measure, axs is not returned as a list
        if num_subplots == 1:
            axs = [axs]

        # Group results by measure
        measures = plot_measures
        grouped_results = {measure: [] for measure in measures}
        names = []

        for result in results:
            result_name = result["name"]
            for measure in measures:
                if measure in result:
                    grouped_results[measure].append(result[measure])
                else:
                    grouped_results[measure].append(None)
            names.append(result_name)

        # Fill each subplot
        for ax, measure in zip(axs, measures):
            valid_indices = [
                i
                for i, value in enumerate(grouped_results[measure])
                if value is not None
            ]
            valid_values = [grouped_results[measure][i] for i in valid_indices]
            valid_names = [names[i] for i in valid_indices]

            bars = ax.bar(
                range(len(valid_values)),
                valid_values,
                color=plt.cm.viridis(np.linspace(0, 1, len(valid_values))),
            )
            ax.set_title(get_translation(f"evaluation.{measure}"))
            ax.set_ylabel(get_translation(f"evaluation.{measure}"))
            ax.set_xticks(range(len(valid_names)))
            ax.set_xticklabels(valid_names, rotation=45, ha="right")

            # Remove borders
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Add value on top of each bar
            for bar, value in zip(bars, valid_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                )

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

        return 0
