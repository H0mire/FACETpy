from .frameworks.correction import CorrectionFramework
from .frameworks.evaluation import EvaluationFramework
from .frameworks.analysis import AnalysisFramework
import mne
from loguru import logger
import numpy as np


class facet:

    def __init__(self):
        self._analysis = AnalysisFramework(self)
        self._correction = None
        self._evaluation = EvaluationFramework(self)
        self._eeg = None
        mne.set_log_level("ERROR")

    def get_eeg(self):
        return self._eeg

    def import_eeg(
        self,
        path,
        fmt="edf",
        artifact_to_trigger_offset=0,
        upsampling_factor=10,
        bads=[],
        subject="subject1",
        session="session1",
        task="task1",
        preload=True,
    ):
        logger.info(f"Importing EEG from {path}")
        self._eeg = self._analysis.import_eeg(
            path,
            fmt=fmt,
            artifact_to_trigger_offset=artifact_to_trigger_offset,
            upsampling_factor=upsampling_factor,
            bads=bads,
            subject=subject,
            session=session,
            task=task,
            preload=preload,
        )
        self._correction = CorrectionFramework(self, self._eeg)
        return self._eeg

    def import_by_eeg_obj(self, eeg):
        self._eeg = eeg
        self._analysis._eeg = eeg
        self._evaluation._eeg = eeg
        self._correction = CorrectionFramework(self, self._eeg)
        return self._eeg

    def export_eeg(
        self,
        path,
        fmt="edf",
        subject="subject1",
        session="session1",
        task="task1",
        event_id=None,
    ):
        self._analysis.export_eeg(
            path,
            fmt=fmt,
            subject=subject,
            session=session,
            task=task,
            event_id=event_id,
        )

    def find_triggers(self, regex, save=False):
        logger.info("finding triggers")
        self._analysis.find_triggers(regex, save=save)
        num_triggers = self._eeg.count_triggers
        logger.info(f"Found {num_triggers} triggers")

    def find_missing_triggers(
        self, upsample=True, mode="auto", ref_channel=0, add_sub_periodic_artifacts=None
    ):
        logger.info("Finding missing triggers...")
        self._analysis.find_missing_triggers(
            upsample=upsample,
            mode=mode,
            ref_channel=ref_channel,
            add_sub_periodic_artifacts=add_sub_periodic_artifacts,
        )

    def prepare(self):
        self._correction.prepare()

    def calc_matrix_aas(self, method="numpy", rel_window_position=0, window_size=30):
        logger.info(f"Calculating matrix with allen et al. averaging method {method}")
        if method == "numpy":
            return self._correction.calc_matrix_aas(
                rel_window_position, window_size=window_size
            )
        else:
            raise ValueError("Invalid method parameter")

    def calc_matrix_motion(self, file_path, threshold=5, window_size=30):
        logger.info(f"Calculating Matrix with motiondata in {file_path}")
        return self._correction.calc_matrix_motion(
            file_path=file_path, threshold=threshold, window_size=window_size
        )

    def prepare_ANC(self):
        logger.warning("This method is not necessary anymore. Skipping...")

    def apply_ANC(self):
        self._correction.apply_ANC()

    def remove_artifacts(self, avg_artifact_matrix_numpy=None, plot_artifacts=False):
        self._correction.remove_artifacts(
            avg_artifact_matrix_numpy=avg_artifact_matrix_numpy,
            plot_artifacts=plot_artifacts,
        )

    def pre_processing(self):  # Change to your liking
        # change to your liking
        self._correction.filter(l_freq=0.5)
        self._correction.upsample()

    def post_processing(self):  # Change to your liking
        # change to your liking
        self._correction.downsample()
        self._correction.filter(h_freq=70)
        self._correction.apply_ANC()

    def cut(self):
        self._correction.cut()

    def plot_eeg(self, start=0, title=None, eeg=None):
        eeg = eeg if eeg is not None else self._eeg
        self._analysis.plot_eeg(start=start, title=title, eeg=eeg)

    def downsample(self):
        self._correction.downsample()

    def lowpass(self, freq=45):
        self._correction.filter(h_freq=freq, l_freq=None)

    def highpass(self, freq=1):
        self._correction.filter(l_freq=freq, h_freq=None)

    def upsample(self):
        self._correction.upsample()

    def evaluate(self, eeg=None, name=None, plot=False, measures=["SNR"]):
        logger.info("Evaluating...")
        return self._evaluation.evaluate(eeg=eeg, name=name, plot=plot, measures=measures)
    
    def plot(self, results, plot_measures=["SNR"]):
        self._evaluation.plot(results, plot_measures=plot_measures)

    def apply_per_channel(self, function):
        self._correction.apply_per_channel(function)

    def align_triggers(
        self,
        ref_trigger_index,
        ref_channel=None,
        save=False,
        search_window=None,
        upsample=True,
    ):
        self._correction.align_triggers(
            ref_trigger_index,
            ref_channel=ref_channel,
            save=save,
            search_window=search_window,
            upsample=upsample,
        )

    def align_subsample(self, ref_trigger):
        self._correction.align_subsample(ref_trigger)

    def get_correction(self):
        return self._correction

    def get_evaluation(self):
        return self._evaluation

    def get_analysis(self):
        return self._analysis

    def set_correction(self, correction):
        self._correction = correction

    def set_evaluation(self, evaluation):
        self._evaluation = evaluation

    def set_analysis(self, analysis):
        self._analysis = analysis

    def create_facet_with_channel_picks(self, ch_picks, load_data=True, raw=None):
        if raw is None:
            one_channel_raw = self._eeg.mne_raw.copy().pick(ch_picks)
        else:
            one_channel_raw = raw.copy().pick(ch_picks)
        one_channel_facet_obj = facet()
        one_channel_eeg_obj = self._eeg.copy()
        one_channel_eeg_obj.mne_raw = one_channel_raw
        one_channel_eeg_obj.mne_raw_orig = one_channel_raw.copy()
        # load data
        if load_data:
            one_channel_raw.load_data()
            one_channel_eeg_obj.estimated_noise = np.zeros(one_channel_raw._data.shape)
        one_channel_facet_obj.import_by_eeg_obj(one_channel_eeg_obj)
        return one_channel_facet_obj
