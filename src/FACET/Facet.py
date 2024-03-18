from .Frameworks.Correction import Correction_Framework
from .Frameworks.Evaluation import Evaluation_Framework
from .Frameworks.Analytics import Analytics_Framework
import mne
from loguru import logger

class Facet:

    def __init__(self):
        self._analytics = Analytics_Framework(self)
        self._correction = None
        self._evaluation = Evaluation_Framework(self)
        self._eeg = None
        mne.set_log_level('ERROR')
    def get_EEG(self):
        return self._eeg
    def import_EEG(self, filename, artifact_to_trigger_offset=0, upsampling_factor=10, bads=[]):
        logger.info(f"Importing EEG from {filename}")
        self._eeg = self._analytics.import_EEG(filename, artifact_to_trigger_offset=artifact_to_trigger_offset, upsampling_factor=upsampling_factor, bads=bads)
        self._correction = Correction_Framework(self,self._eeg)
        return self._eeg

    def export_EEG(self, filename):
        self._analytics.export_EEG(filename)
    def get_eeg(self):
        return self._eeg
    def find_triggers(self, regex):
        logger.info("finding triggers")
        self._analytics.find_triggers(regex)
        num_triggers=self._eeg.count_triggers
        logger.info(f"Found {num_triggers} triggers")
    def prepare(self):
        self._correction.prepare()
    def apply_AAS(self, method="numpy", rel_window_position=0, window_size=25):
        logger.info(f"Applying AAS with method {method}")
        if method == "numpy":
            self._correction.apply_AAS(rel_window_position, window_size=window_size)
        else:
            raise ValueError("Invalid method parameter")
        
    def apply_Moosmann(self, file_path, threshold=5, window_size=25):
        logger.info(f"Applying Moosmann with {file_path}")
        self._correction.apply_Moosmann(file_path=file_path, threshold=threshold, window_size=window_size)
    def remove_artifacts(self, avg_artifact_matrix_numpy=None, plot_artifacts=False):  
        self._correction.remove_artifacts(avg_artifact_matrix_numpy=avg_artifact_matrix_numpy, plot_artifacts=plot_artifacts, )
    def pre_processing(self): # Change to your liking
        #change to your liking
        self._correction.filter(l_freq=1)
        self._correction.upsample()
        #self._correction.prepare_ANC()
    def post_processing(self): # Change to your liking
        #change to your liking
        self._correction.downsample()
        self._correction.filter(h_freq=45)
        #self._correction.apply_ANC()
    def cut(self):
        self._correction.cut()
    def plot_EEG(self, start=0, title=None):
        self._correction.plot_EEG(start=start, title=title)
    def downsample(self):
        self._correction.downsample()
    def lowpass(self, freq=45):
        self._correction.filter(h_freq=freq, l_freq=None)
    def highpass(self, freq=1):
        self._correction.filter(l_freq=freq, h_freq=None)
    def upsample(self):
        self._correction.upsample()
    def add_to_evaluate(self, eeg,start_time=None, end_time=None, name=None):
        logger.info("Adding to evaluation...")
        self._evaluation.add_to_evaluate(eeg,start_time=start_time, end_time=end_time, name=name)
    def evaluate(self, plot=True, measures=["SNR"]):
        logger.info("Evaluating...")
        return self._evaluation.evaluate(plot=plot, measures=measures)
    def export_as_bids(self, event_id=None, bids_path="./bids_dir", subject="subjectid", session="sessionid", task="corrected"):
        self._analytics.export_as_bids(event_id=event_id, bids_path=bids_path, subject=subject, session=session, task=task)
    def import_from_bids(self, bids_path="./bids_dir", artifact_to_trigger_offset=0, upsampling_factor=10, bads=[], subject="subjectid", session="sessionid", task="corrected"):
        self._eeg = self._analytics.import_from_bids(bids_path, artifact_to_trigger_offset=artifact_to_trigger_offset, upsampling_factor=upsampling_factor, bads=bads, subject=subject, session=session, task=task)
        self._correction = Correction_Framework(self,self._eeg)
    
    def get_correction(self):
        return self._correction
    def get_evaluation(self):
        return self._evaluation
    def get_analytics(self):
        return self._analytics
    def set_correction(self, correction):
        self._correction = correction
    def set_evaluation(self, evaluation):
        self._evaluation = evaluation
    def set_analytics(self, analytics):
        self._analytics = analytics