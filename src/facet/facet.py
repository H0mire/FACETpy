from .frameworks.correction import correction_framework
from .frameworks.evaluation import evaluation_framework
from .frameworks.analysis import analysis_framework
import mne
from loguru import logger


class facet:

    def __init__(self):
        self._analysis = analysis_framework(self)
        self._correction = None
        self._evaluation = evaluation_framework(self)
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
        )
        self._correction = correction_framework(self, self._eeg)
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

    def find_triggers(self, regex):
        logger.info("finding triggers")
        self._analysis.find_triggers(regex)
        num_triggers = self._eeg.count_triggers
        logger.info(f"Found {num_triggers} triggers")

    def find_missing_triggers(self):
        logger.info("Finding missing triggers...")
        self._analysis.find_missing_triggers()

    def prepare(self):
        self._correction.prepare()

    def calc_matrix_aas(self, method="numpy", rel_window_position=0, window_size=30):
        logger.info(f"Applying AAS with method {method}")
        if method == "numpy":
            self._correction.calc_matrix_aas(
                rel_window_position, window_size=window_size
            )
        else:
            raise ValueError("Invalid method parameter")

    def prepare_ANC(self):
        logger.warning("This method is not necessary anymore. Skipping...")

    def apply_ANC(self):
        self._correction.apply_ANC()

    def align_triggers(self, ref_trigger_index):
        self._correction.align_triggers(ref_trigger_index)

    def calc_matrix_motion(self, file_path, threshold=5, window_size=30):
        logger.info(f"Applying Moosmann with {file_path}")
        self._correction.calc_matrix_motion(
            file_path=file_path, threshold=threshold, window_size=window_size
        )

    def remove_artifacts(self, avg_artifact_matrix_numpy=None, plot_artifacts=False):
        self._correction.remove_artifacts(
            avg_artifact_matrix_numpy=avg_artifact_matrix_numpy,
            plot_artifacts=plot_artifacts,
        )

    def pre_processing(self):  # Change to your liking
        # change to your liking
        self._correction.filter(l_freq=1)
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
        self._correction.plot_eeg(start=start, title=title, eeg=eeg)

    def downsample(self):
        self._correction.downsample()

    def lowpass(self, freq=45):
        self._correction.filter(h_freq=freq, l_freq=None)

    def highpass(self, freq=1):
        self._correction.filter(l_freq=freq, h_freq=None)

    def upsample(self):
        self._correction.upsample()

    def add_to_evaluate(self, eeg, start_time=None, end_time=None, name=None):
        logger.info("Adding to evaluation...")
        self._evaluation.add_to_evaluate(
            eeg, start_time=start_time, end_time=end_time, name=name
        )

    def evaluate(self, plot=True, measures=["SNR"]):
        logger.info("Evaluating...")
        return self._evaluation.evaluate(plot=plot, measures=measures)

    def align_triggers(self, ref_trigger_index):
        self._correction.align_triggers(ref_trigger_index)

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
