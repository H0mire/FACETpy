""" Analysis framework Module

This module contains the AnalysisFramework class, which provides methods for importing, exporting, and analyzing EEG data.

Author: Janik Michael Müller
Date: 15.02.2024
Version: 1.0
"""

import numpy as np
import mne
import re
from mne_bids import BIDSPath, write_raw_bids, read_raw_bids
from scipy.stats import pearsonr
from scipy.signal import firls
from facet.eeg_obj import EEG
import numpy as np
from loguru import logger
from scipy.signal import find_peaks
import neurokit2 as nk
import matplotlib.pyplot as plt


# import inst for mne python


class AnalysisFramework:
    def __init__(self, facet, eeg=None):
        """
        Initializes an instance of the AnalysisFramework class.

        Parameters:
            facet (facet.facet): A reference to an instance of a facet class (or similar) that provides additional functionality for EEG data processing.
            eeg (facet.eeg_obj, optional): An instance of an EEG object. If not provided, a new EEG object is created.
        """
        self._loaded_triggers = None
        self._plot_number = 0
        self._facet = facet

        if eeg:
            self._eeg = eeg
        else:
            self._eeg = EEG()

    def import_eeg(
        self,
        path,
        artifact_to_trigger_offset=0,
        upsampling_factor=10,
        fmt="edf",
        bads=[],
        subject="subjectid",
        session="sessionid",
        task="corrected",
        preload=True,
    ):
        """
        Imports EEG data from a file, supporting various formats, and loads it into the EEG object.

        Parameters:
            filename (str): The path to the EEG file.
            artifact_to_trigger_offset (float): The relative position of the trigger in the data.
            upsampling_factor (int): The factor by which to upsample the data.
            fmt (str): The format of the EEG file (either "edf" or "gdf").
            bads (list): A list of bad channels to exclude from the data.

        Returns:
            EEG: The EEG object containing the imported data and metadata.
        """
        if fmt == "edf":
            raw = mne.io.read_raw_edf(path)
        elif fmt == "gdf":
            raw = mne.io.read_raw_gdf(path)
        elif fmt == "bids":
            bids_path_i = BIDSPath(
                subject=subject, session=session, task=task, root=path
            )
            raw = read_raw_bids(bids_path_i)
        elif fmt == "eeglab":
            raw = mne.io.read_raw_eeglab(path)
        else:
            raise ValueError("Format not supported")

        mne_raw_orig = raw.copy()
        if preload:
            raw.load_data()

        all_channels = raw.ch_names
        exclude = [item for i, item in enumerate(all_channels) if item in bads]
        raw.info["bads"] = exclude
        data_time_start = raw.times[0]
        data_time_end = raw.times[-1]

        self._eeg = EEG(
            mne_raw=raw,
            mne_raw_orig=mne_raw_orig,
            estimated_noise=(
                np.zeros(len(raw.ch_names), raw.n_times) if preload else None
            ),
            artifact_to_trigger_offset=artifact_to_trigger_offset,
            upsampling_factor=upsampling_factor,
            data_time_start=data_time_start,
            data_time_end=data_time_end,
        )
        events = self._try_to_get_events()
        if events is not None:
            self._eeg.all_events = events
        if fmt == "bids":
            self._eeg.BIDSPath = bids_path_i
        logger.debug("Importing EEG with:")
        logger.debug("Channels " + str(raw.ch_names))
        logger.debug(f"Time Start: {data_time_start}s")
        logger.debug(f"Time End: {data_time_end}s")
        logger.debug(f"Number of Samples: {raw.n_times}")
        logger.debug(f"Sampling Frequency: {raw.info['sfreq']}Hz")
        logger.debug(path)
        return self._eeg

    def export_eeg(
        self,
        path,
        fmt="edf",
        subject="subjectid",
        session="sessionid",
        task="corrected",
        event_id=None,
    ):
        """
        Exports the EEG data to a file.

        Parameters:
            path (str): The destination path for the exported file.
            fmt (str, optional): The format of the exported EEG file. (e.g., "edf", "bdf", "fif", "bids")
            Other parameters are similar to import_eeg, relevant for BIDS format.
        """
        if fmt == "bids":
            _BIDSPath = BIDSPath(subject=subject, session=session, task=task, root=path)
            logger.info("Exporting Channels: " + str(self._eeg.mne_raw.ch_names))

            raw = self._eeg.mne_raw.copy()
            # drop stim channels
            stim_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
            raw.drop_channels([raw.ch_names[ch] for ch in stim_channels])

            if self._eeg.mne_raw is not None:
                write_raw_bids(
                    raw=raw,
                    bids_path=_BIDSPath,
                    overwrite=True,
                    format="EDF",
                    allow_preload=True,
                    events=self._eeg.triggers_as_events,
                    event_id=event_id,
                )
        else:
            raw = self._eeg.mne_raw
            raw.export(path, fmt=fmt, overwrite=True)

    def find_triggers(self, regex, save=False):
        """
        Finds triggers in the EEG data based on a regular expression matching trigger values.

        It automatically detects whether the trigger values are stored in the EEG data like a Stim Channel or in annotations.
        It also derives some parameters like the time of the first artifact start, the time of the last trigger, the artifact length, and the ANC high-pass filter parameters.

        Parameters:
            regex (str): The regular expression pattern to match against trigger values.

        Returns:
            None
        """
        raw = self._eeg.mne_raw
        stim_channels = mne.pick_types(raw.info, meg=False, eeg=False, stim=True)
        events = []
        filtered_events = []

        if len(stim_channels) > 0:
            logger.debug(
                "Stim-Kanäle gefunden:", [raw.ch_names[ch] for ch in stim_channels]
            )
            events = mne.find_events(
                raw, stim_channel=raw.ch_names[stim_channels[0]], initial_event=True
            )
            pattern = re.compile(regex)
            filtered_events = [
                event for event in events if pattern.search(str(event[2]))
            ]

        else:
            logger.debug("No Stim-Channels found.")
            events_obj = mne.events_from_annotations(raw)
            logger.debug(events_obj[1])
            filtered_events = mne.events_from_annotations(raw, regexp=regex)[0]

        if len(filtered_events) == 0:
            logger.error("No events found!")
            return
        filtered_positions = [event[0] for event in filtered_events]
        triggers = filtered_positions
        logger.debug(f"Found {len(triggers)} triggers")
        self._eeg.last_trigger_search_regex = regex
        self._eeg.loaded_triggers = triggers
        if save:
            # replace the triggers as events in the raw object
            self._eeg.mne_raw.set_annotations(
                mne.Annotations(
                    onset=np.array(self._eeg.loaded_triggers)
                    / self._eeg.mne_raw.info["sfreq"],
                    duration=np.zeros(len(self._eeg.loaded_triggers)),
                    description=["Trigger"] * len(self._eeg.loaded_triggers),
                )
            )

        self.derive_parameters()
    
    def find_triggers_qrs(self, save=False):
        """
        Find triggers in the EEG data based on the QRS complex of an ECG channel.
        """

        raw = self._eeg.mne_raw
        ecg_channels = mne.pick_types(raw.info, meg=False, eeg=False, ecg=True)
        

        if len(ecg_channels) == 0:
            # Now check if there are Channels called ECG and warn that there are not marked as ECG
            ecg_channels = [
                i
                for i, ch in enumerate(raw.ch_names)
                if "ECG" in ch.upper() or "EKG" in ch.upper()
            ]
            if len(ecg_channels) == 0:
                logger.error("No ECG channels found!")
                return
            else:
                logger.warning(
                    "No ECG channels found! Found channels with ECG in name. Using these channels."
                )

        ecg_channel = raw.get_data(picks=ecg_channels)
        # Plot the ECG signal
        # plt.plot(ecg_channel[0])
        ecg_channel = ecg_channel[0]

        # Find the peaks of the ECG signal
        _, rpeaks = nk.ecg_peaks(ecg_channel, sampling_rate=raw.info['sfreq'])
        peaks = np.abs(rpeaks['ECG_R_Peaks']).astype(int)

        # Create events based on the peak positions
        events = np.zeros((len(peaks), 3))
        events[:, 0] = peaks
        events[:, 2] = 1

        # Store the events in the EEG object
        self._eeg.last_trigger_search_regex = "QRS"
        self._eeg.loaded_triggers = list(map(int, events[:, 0]))

        if save:
            # replace the triggers as events in the raw object
            self._eeg.mne_raw.set_annotations(
                mne.Annotations(
                    onset=events[:, 0] / self._eeg.mne_raw.info["sfreq"],
                    duration=np.zeros(len(events)),
                    description=["Trigger"] * len(events),
                )
            )

        self.derive_parameters()

    def derive_parameters(self):
        """
        Calculates various parameters based on the loaded triggers, including artifact start times and durations.

        Derives:
            time_first_artifact_start
            time_last_artifact_end
            artifact_length
            artifact_duration
            ANC high-pass filter parameters
        """
        self._check_volume_gaps()
        self._derive_art_length()
        self._derive_times()
        self._derive_anc_hp_params()
        self._derive_pca_params()
        self._derive_tmin_tmax()

    def get_mne_raw(self):
        """
        Returns the raw EEG data.

        Returns:
            mne.io.Raw: The raw EEG data.
        """
        return self._eeg.mne_raw

    def get_mne_raw_orig(self):
        """
        Returns the original raw MNE object, prior to any processing.

        Returns:
            mne.io.Raw: The original raw EEG data.
        """
        return self._eeg.mne_raw_orig

    def get_eeg(self):
        """
        Returns the EEG data object associated with this instance.

        Returns:
            The EEG data object.
        """
        return self._eeg

    def plot_eeg(self, start=0, title=None, eeg=None):
        """
        Plots the raw EEG data.

        Parameters:
            start (int): The starting index of the data to be plotted.
            title (str, optional): The title of the plot. If not provided, a default title will be used.
            eeg (facet.eeg_obj, optional): The EEG data to plot. If not provided, the instance's EEG data is used.
        """
        eeg = eeg if eeg is not None else self._eeg
        if not title:
            self._plot_number += 1
            title = str(self._plot_number)
        eeg.mne_raw.plot(title=title, start=start)

    def _try_to_get_events(self):
        """
        Tries to extract events from the EEG data, either from annotations or explicit event objects.

        Returns:
            An array of events or None if no events are found.
        """
        # Check if there are annotations and convert
        if self._eeg.mne_raw.annotations:
            return mne.events_from_annotations(self._eeg.mne_raw)[0]

        # Check if there are events
        if hasattr(self._eeg.mne_raw, "events"):
            return self._eeg.mne_raw.events

        return None

    def _derive_art_length(self):
        """
        Calculate the length of an artifact based on trigger distances.

        This method calculates the length of an artifact by analyzing the trigger distances
        between consecutive triggers in the EEG data. If there are volume gaps in the data,
        the middle distance is used to determine the trigger distances belonging to slice triggers.
        Otherwise, all trigger distances are considered.

        The calculated artifact length is stored in the `_eeg.artifact_length` attribute.

        If there are no volume gaps, the duration of the artifact is also calculated and stored
        in the `_eeg.artifact_duration` attribute.

        Returns:
            None
        """
        d = np.diff(self._eeg.loaded_triggers)  # trigger distances

        if self._eeg.volume_gaps:
            m = np.mean([np.min(d), np.max(d)])  # middle distance
            ds = d[d < m]  # trigger distances belonging to slice triggers
            # dv = d[d > m]  # trigger distances belonging to volume triggers

            # total length of an artifact
            # use max to avoid gaps between slices
            self._eeg.artifact_length = np.max(ds)

        else:
            # total length of an artifact
            self._eeg.artifact_length = np.max(d)
        self._eeg.artifact_duration = (
            self._eeg.artifact_length / self._eeg.mne_raw.info["sfreq"]
        )

    def add_triggers(self, triggers):
        """
        Add triggers to the EEG data.

        This method adds triggers to the EEG data based on the provided trigger positions.

        Parameters:
            triggers (list): List of trigger positions.

        Returns:
            None
        """
        if len(triggers) == 0:
            logger.error("No triggers provided!")
            return
        triggers = np.sort(triggers)
        # check if triggers are within the data
        if triggers[0] < 0 or triggers[-1] > self._eeg.mne_raw.n_times:
            logger.warning("Some Triggers are not within the data! Removing them...")
            triggers = triggers[triggers >= 0]
            triggers = triggers[triggers <= self._eeg.mne_raw.n_times]

        # check if triggers are not already in the data
        intersection = np.intersect1d(triggers, self._eeg.loaded_triggers)
        if len(intersection) > 0:
            logger.warning(
                f"There are {len(intersection)} triggers already in the data at positions {intersection}. Removing them..."
            )
            triggers = np.setdiff1d(triggers, intersection)

        # add triggers and ensure they are sorted
        self._eeg.loaded_triggers = np.sort(
            np.concatenate([self._eeg.loaded_triggers, triggers])
        ).tolist()

        self.derive_parameters()

    def find_missing_triggers(
        self,
        mode="auto",
        ref_channel=0,
        upsample=True,
        add_sub_periodic_artifacts=None,
    ):
        """
        Attempts to identify and add missing triggers in the EEG data.

        Parameters:
            mode (str, optional): The mode for finding missing triggers, default is "auto".
            ref_channel (int, optional): The reference channel to use.

        Returns:
            list: A list of the missing trigger positions.
        """
        missing_triggers = []
        f = None
        if self._eeg.mne_raw.preload:
            f = self._facet
        else:
            f = self._facet.create_facet_with_channel_picks([ref_channel])
            ref_channel = 0

        if mode == "auto":
            needed_to_upsample = False
            if (
                f._eeg.mne_raw.info["sfreq"]
                == self._facet._eeg.mne_raw_orig.info["sfreq"]
                and upsample
            ):
                if self._eeg.mne_raw.preload:
                    f = self._facet.create_facet_with_channel_picks(
                        [ref_channel], raw=self._eeg.mne_raw_orig
                    )
                # upsample data here because this achieves better results
                f.upsample()
                needed_to_upsample = True

            smin, smax = f._eeg.smin, f._eeg.smax

            search_window = int(0.1 * f._eeg.artifact_length)
            logger.info("Finding missing triggers using auto mode...")
            if f._eeg.volume_gaps:
                logger.warning(
                    "Volume gaps are detected or flag is manually set to True. Results may be inaccurate"
                )
            logger.debug("Generating template from reference channel...")
            _3d_matrix = f._correction.calc_matrix_aas(channels=[ref_channel])
            template = f._correction.calc_avg_artifact(_3d_matrix)[0][0]
            # iterate through the trigger positions check for each trigger if the next trigger is within the artifact length with a threshold of 1.9*artifactlength
            logger.debug("Checking holes in the trigger positions...")
            for i in range(len(f._eeg.loaded_triggers) - 1):
                if (
                    f._eeg.loaded_triggers[i + 1] - f._eeg.loaded_triggers[i]
                    > f._eeg.artifact_length * 1.9
                ):
                    new_trigger = f._correction._align_trigger(
                        f._eeg.loaded_triggers[i] + f._eeg.artifact_length,
                        template,
                        search_window,
                        ref_channel,
                    )
                    missing_triggers.append(new_trigger)

            logger.debug(f"Found {len(missing_triggers)} missing triggers")
            logger.debug("Now removing triggers that are not artifacts...")
            # now check if each missing trigger is an artifact and remove if it is not
            for trigger in missing_triggers:
                if not f._analysis._is_artifact(trigger, template):
                    missing_triggers.remove(trigger)
            logger.debug(
                f"Found {len(missing_triggers)} missing triggers that are artifacts"
            )
            logger.debug(
                "Now adding missing triggers at the beginning and end of the data..."
            )
            # now check iteratively if triggers are missing at the beginning and end of the data by checking if first trigger - artifact length is an artifact and if last trigger + artifact length is an artifact
            # adding at the beginning and the end as long as the triggers are artifacts
            temp_pos = f._eeg.loaded_triggers[0] - f._eeg.artifact_length
            new_pos = f._correction._align_trigger(
                temp_pos, template, search_window, ref_channel
            )
            count = 0
            while f._analysis._is_artifact(new_pos, template):
                missing_triggers.insert(0, new_pos)
                count += 1
                temp_pos = new_pos - f._eeg.artifact_length
                new_pos = f._correction._align_trigger(
                    temp_pos, template, search_window, ref_channel
                )
            logger.debug(f"Found {count} missing triggers at the beginning of the data")
            count = 0
            temp_pos = f._eeg.loaded_triggers[-1] + f._eeg.artifact_length
            new_pos = f._correction._align_trigger(
                temp_pos, template, search_window, ref_channel
            )
            while f._analysis._is_artifact(new_pos, template):
                missing_triggers.append(new_pos)
                count += 1
                temp_pos = new_pos + f._eeg.artifact_length
                new_pos = f._correction._align_trigger(
                    temp_pos, template, search_window, ref_channel
                )
            logger.debug(f"Found {count} missing triggers at the end of the data")
            # now check if there are sub periodic artifacts in the data
            logger.debug("Now checking for sub periodic artifacts...")
            sub_periodic_artifacts = self._detect_sub_periodic_artifacts(
                f._eeg.mne_raw.get_data()[ref_channel][
                    f._eeg.loaded_triggers[0] + smin : f._eeg.loaded_triggers[0] + smax
                ]
            )
            if len(sub_periodic_artifacts) != 0:
                logger.warning(
                    f"{len(sub_periodic_artifacts)} sub periodic artifacts between each triggerpair detected! (This often happens if you use volume triggers only and not slice triggers.)"
                )
                if add_sub_periodic_artifacts is None:
                    logger.info("Do you want to add them as triggers? (y/n)")
                    logger.debug("Prompting user for response...")
                    response = input("y/n: ")
                    add_sub_periodic_artifacts = response == "y"
                if add_sub_periodic_artifacts:
                    logger.debug("Generating sub triggers...")
                    all_triggers = f._eeg.loaded_triggers + missing_triggers
                    template = template[
                        : len(template) // (len(sub_periodic_artifacts) + 1)
                    ]
                    missing_triggers += f._analysis._generate_sub_triggers(
                        all_triggers, len(sub_periodic_artifacts)
                    )
                    search_window = int(
                        0.1
                        * (f._eeg.artifact_length / (len(sub_periodic_artifacts) + 1))
                    )
            logger.info(f"Found {len(missing_triggers)} missing triggers in total")
            if len(missing_triggers) == 0:
                logger.info("No missing triggers found. Finishing...")
                return []
            logger.debug("Now aligning the missing triggers...")
            # align all missing triggers with self._facet._correction._align_trigger
            for i in range(len(missing_triggers)):
                missing_triggers[i] = f._correction._align_trigger(
                    missing_triggers[i], template, search_window, ref_channel
                )
            # add the missing triggers as annotations with description "missing_trigger"
            on_sets = np.array(missing_triggers) / f._eeg.mne_raw.info["sfreq"]
            # zero duration
            durations = np.zeros(len(missing_triggers))
            descriptions = ["missing_trigger"] * len(missing_triggers)
            annotations = mne.Annotations(
                onset=on_sets, duration=durations, description=descriptions
            )
            f._analysis._add_annotations(annotations)
            # add the missing triggers to the EEG data
            f._analysis.add_triggers(missing_triggers)
            if needed_to_upsample:
                f.downsample()
            self._eeg.loaded_triggers = f._eeg.loaded_triggers
            self._eeg.mne_raw.set_annotations(
                f._eeg.mne_raw.annotations
            )  # set the annotations to the raw object
        else:
            logger.error("Mode not supported!")
        if not self._eeg.mne_raw.preload:
            # unload data
            del f

        return missing_triggers

    def _detect_sub_periodic_artifacts(self, ch_d):
        """
        detect if there are not registered triggers in the data. Often the triggers represent only volume triggers and not slice triggers.
        This method tries the determine if there are sub triggers in the data and returns them.

        Returns:
            list: A list of the sub trigger positions.
        """
        # first check if there are peaks in the data that around the same amplitude to each other and do not contain a higher peak in between
        # if there are peaks that are around the same amplitude to each other and do not contain a higher peak in between, they are considered as sub triggers
        # the peaks are considered as sub triggers if they are within the current determined artifact length
        may_be_sub_sub_periodic_artifacts = []

        data_zero_mean = ch_d - np.mean(ch_d)
        # now determine a threshold for the peaks
        threshold = np.max(data_zero_mean) * 0.9
        threshold_diff = np.max(data_zero_mean) * 0.1
        # now determine the peaks positions and values
        peaks, _ = find_peaks(data_zero_mean, height=threshold)

        values = data_zero_mean[peaks]
        # now determine if there more than 1 peak and if the peaks are around the same amplitude to each other
        if len(peaks) > 1:
            diffs = np.diff(values)
            if np.ptp(diffs) < threshold_diff:
                may_be_sub_sub_periodic_artifacts = peaks[1:]
        return may_be_sub_sub_periodic_artifacts

    def generate_sub_triggers(self, count):
        """
        Generate subtriggers for each trigger in the EEG data.

        Parameters:
            count (int): The number of subtriggers to generate for each trigger.

        Returns:
            None
        """
        triggers = self._eeg.loaded_triggers
        sub_triggers = self._generate_sub_triggers(triggers, count)
        self.add_triggers(sub_triggers)

    def _generate_sub_triggers(self, triggers, count):
        """
        Generate {count} subtriggers, for each trigger in triggers.

        Parameters:
            triggers (list): List of trigger positions.
            count (int): Number of subtriggers to generate.

        Returns:
            list: A list of the sub trigger positions.
        """
        sub_triggers = []
        for trigger in triggers:
            # determine the distance between the new sub triggers
            distance = self._eeg.artifact_length / (count + 1)
            for i in range(1, count + 1, 1):
                sub_triggers.append(int(trigger + i * distance))
        return sub_triggers

    def _add_annotations(self, annotations):
        """
        Add annotations to the EEG data.

        Parameters:
            annotations (list): List of annotations to add.

        Returns:
            None
        """
        raw = self._eeg.mne_raw
        # Hole die bestehenden Annotations aus dem Raw-Objekt
        existing_annotations = raw.annotations

        # Kombiniere die bestehenden mit den neuen Annotationen
        # Dazu fügst du die Listen von Onsets, Durations und Descriptions zusammen
        combined_onset = list(existing_annotations.onset) + list(annotations.onset)
        combined_duration = list(existing_annotations.duration) + list(
            annotations.duration
        )
        combined_description = list(existing_annotations.description) + list(
            annotations.description
        )

        # Erstelle ein neues Annotations-Objekt mit den kombinierten Daten
        combined_annotations = mne.Annotations(
            onset=combined_onset,
            duration=combined_duration,
            description=combined_description,
        )

        # Setze die kombinierten Annotations zurück in das Raw-Objekt
        raw.set_annotations(combined_annotations)

    def _is_artifact(self, position, template, threshold=0.9):
        """
        Check if a given position mark an artifact.

        This method checks if a given position based on a template with a correlation threshold of 0.9

        Parameters:
            position (int): The position to check.
            template: The artifact template for comparison.
            threshold (float, optional): The correlation threshold to determine if a position is an artifact.

        Returns:
            bool: True if the position is an artifact, False otherwise.
        """
        new_position = self._facet._correction._align_trigger(
            position, template, 3 * self._eeg.upsampling_factor, 0
        )
        smin = self._eeg.smin
        data = self._eeg.mne_raw.get_data(
            start=new_position + smin,
            stop=new_position + smin + self._eeg.artifact_length,
        )
        template = template[: len(data[0])]
        if len(template) < 3:
            return False
        corr = np.abs(pearsonr(data[0], template)[0])
        return corr > threshold

    def _derive_anc_hp_params(self):
        """
        Derive ANC high-pass filter parameters.

        This method derives the parameters for the ANC high-pass filter based on the trigger frequency
        and the sampling frequency of the EEG data. The filter order is calculated based on the
        artifact length and the upsampling factor.

        The calculated filter weights are stored in the `_eeg.anc_hp_filter_weights` attribute.

        Returns:
            None
        """

        sfreq = self._eeg.mne_raw.info["sfreq"]
        artifact_length = self._eeg.artifact_length
        trans = 0.15
        nyq = 0.5 * sfreq

        if self._eeg.count_triggers >= 1:
            # Schätzung der Frequenz der Trigger
            Tr = 1
            while Tr <= self._eeg.count_triggers:
                # Python-Indexierung beginnt bei 0
                tr_samp_diff = (
                    self._eeg.loaded_triggers[Tr] - self._eeg.loaded_triggers[0]
                )
                if tr_samp_diff >= sfreq:
                    break
                Tr += 1
            # ANC HP cut-off Frequenz ist 25% niedriger als die geschätzte Triggerfrequenz
            self._eeg.anc_hp_frequency = 0.75 * Tr
        else:
            self._eeg.anc_hp_frequency = 2

        filtorder = round(1.2 * sfreq / (self._eeg.anc_hp_frequency * (1 - trans)))
        if filtorder % 2 == 0:
            filtorder += 1

        f = [
            0,
            (self._eeg.anc_hp_frequency * (1 - trans)) / nyq,
            self._eeg.anc_hp_frequency / nyq,
            1,
        ]
        a = [0, 0, 1, 1]
        self._eeg.anc_hp_filter_weights = firls(filtorder, f, a)
        self._eeg.anc_filter_order = artifact_length

    def _derive_pca_params(self):
        """
        Derives the parameters for performing Principal Component Analysis (PCA) on the EEG data.

        This method calculates the filter order and filter weights for high-pass filtering the EEG data
        before applying PCA.

        Returns:
            None
        """
        sfreq = self._eeg.mne_raw.info["sfreq"]
        nyq = 0.5 * sfreq

        filtorder = round(1.2 * sfreq / (self._eeg.obs_hp_frequency - 10))
        if filtorder % 2 == 0:
            filtorder += 1

        f = [
            0,
            (self._eeg.obs_hp_frequency - 10) / nyq,
            (self._eeg.obs_hp_frequency + 10) / nyq,
            1,
        ]
        a = [0, 0, 1, 1]
        self._eeg.obs_hp_filter_weights = firls(filtorder, f, a)

    def _derive_tmin_tmax(self):
        """
        Derive the tmin and tmax values for the EEG data.

        This method calculates the tmin and tmax values for the EEG data based on the artifact length
        and the artifact to trigger offset.

        The calculated values are stored in the `_eeg._tmin` and `_eeg._tmax` attributes.

        Returns:
            None
        """
        self._eeg._tmin = self._eeg.artifact_to_trigger_offset
        self._eeg._tmax = (
            self._eeg.artifact_to_trigger_offset + self._eeg.artifact_duration
        )

    def _derive_times(self):
        """
        Derive the time of the first artifact start and the time of the last trigger.

        This method calculates the time of the first artifact start and the time of the last trigger
        based on the EEG data and the loaded triggers.

        The calculated values are stored in the `_eeg.time_first_artifact_start` and `_eeg.time_last_artifact_end` attributes.

        Returns:
            None
        """
        triggers = self._eeg.loaded_triggers
        time_first_artifact_start = self._eeg.mne_raw.times[triggers[0]]
        time_last_trigger = self._eeg.mne_raw.times[triggers[-1]]
        self._eeg.time_first_artifact_start = np.max(
            [time_first_artifact_start + self._eeg.artifact_to_trigger_offset, 0]
        )
        self._eeg.time_last_artifact_end = np.min(
            [
                time_last_trigger
                + self._eeg.artifact_to_trigger_offset
                + self._eeg.artifact_duration,
                self._eeg.mne_raw.times[-1],
            ]
        )
        self._eeg.time_acq_padding_left = self._eeg.artifact_duration
        self._eeg.time_acq_padding_right = self._eeg.artifact_duration

    def _check_volume_gaps(self):
        """
        Check for volume gaps in the EEG data.

        This method checks for volume gaps in the EEG data by analyzing the trigger distances
        between consecutive triggers. If the difference between the minimum and maximum
        trigger distance is greater than 3, volume gaps are assumed.

        The result is stored in the `_eeg.volume_gaps` attribute.

        Returns:
            None
        """
        # Due to asynchronous sampling the distances might vary a bit. We
        # accept one mean value, plus and minus one (gives a range of 2),
        # plus one more to be a bit more robust.
        if self._eeg.volume_gaps is None:
            if np.ptp(np.diff(self._eeg.loaded_triggers)) > 3:
                self._eeg.volume_gaps = True
            else:
                self._eeg.volume_gaps = False

    def _filter_annotations(self, regex):
        """
        Extract specific annotations from an MNE Raw object.

        Parameters:
            regex (str): Regular expression pattern to match the annotation description.

        Returns:
            list: List of tuples containing the matched annotations (time, duration, description).
        """
        eeg = self._eeg
        # initialize list to store results
        specific_annotations = []

        # compile the regular regex pattern
        pattern = re.compile(regex)

        # loop through each annotation in the raw object
        for annot in eeg.mne_raw.annotations:
            # check if the annotation description matches the pattern
            if pattern.search(annot["description"]):
                # if it does, append the annotation (time, duration, description) to our results list
                specific_annotations.append(
                    (annot["onset"], annot["duration"], annot["description"])
                )

        return specific_annotations

    def print_analytics(self):
        """
        Prints analysis information.

        This method logs various analysis information, including the number of triggers found,
        art length, duration of art in seconds, number of channels, and channel names.

        """
        logger.info("Analysis:")
        logger.info(f"Number of Triggers found: {self._eeg.count_triggers}")
        logger.info(f"Art Length: {self._eeg.artifact_length}")
        logger.info(f"Duration of Art in seconds: {self._eeg.artifact_duration}")

        # EEG information
        # print EEG Channels
        raw = self._eeg.mne_raw
        ch_names = raw.ch_names
        count_ch = len(ch_names)
        logger.info("Time Start: " + str(raw.times[0]) + " s")
        logger.info("Time End: " + str(raw.times[-1]) + " s")
        logger.info("Sampling Frequency: " + str(raw.info["sfreq"]) + " Hz")
        logger.info("Number of Samples: " + str(raw.n_times))
        logger.info("Number of Channels: " + str(count_ch))
        logger.info("Channel Names: " + str(ch_names))
