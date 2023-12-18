from .Frameworks.Correction import Correction_Framework
from .Frameworks.Evaluation import Evaluation_Framework
from .Frameworks.Analytics import Analytics_Framework
import mne

class Facet:

    def __init__(self):
        self._analytics = Analytics_Framework()
        self._correction = None
        self._evaluation = Evaluation_Framework()
        self._eeg = None
        mne.set_log_level('ERROR')
    def get_EEG(self):
        return self._eeg
    def import_EEG(self, filename, rel_trig_pos=0, upsampling_factor=10, bads=[]):
        self._eeg = self._analytics.import_EEG(filename, rel_trig_pos=rel_trig_pos, upsampling_factor=upsampling_factor, bads=bads)
        self._correction = Correction_Framework(self._eeg)
        return self._eeg

    def export_EEG(self, filename):
        self._analytics.export_EEG(filename)
    def get_eeg(self):
        return self._eeg
    def find_triggers(self, regex, idx = 0):
        self._analytics.find_triggers(regex, idx=idx)
        print("finding triggers")
    def find_triggers_with_events(self, regex, idx=0):
        self._analytics.find_triggers_with_events(regex, idx=idx)
        print("finding triggers")
    def prepare(self):
        self._correction.prepare()
    def apply_MNE_AAS(self, method="normal"):
        if method == "old":
            self._correction.apply_MNE_AAS_old()
        elif method == "matrix":
            self._correction.apply_MNE_AAS_matrix()
        elif method == "normal":
            self._correction.apply_MNE_AAS()
        else:
            raise ValueError("Invalid method parameter")
    def remove_artifacts(self):  
        self._correction.remove_artifacts()
    def pre_processing(self):
        self._correction.highpass(1)
        self._correction.upsample()
    def post_processing(self):
        self._correction.downsample()
        self._correction.lowpass(50)
    def cut(self):
        self._correction.cut()
    def plot_EEG(self, start=0, title=None):
        self._correction.plot_EEG(start=start, title=title)
    def downsample(self):
        self._correction.downsample()
    def lowpass(self, h_freq=45):
        self._correction.lowpass(h_freq=h_freq)
    def highpass(self, l_freq=1):
        self._correction.highpass(l_freq=l_freq)
    def upsample(self):
        self._correction.upsample()
    def add_to_evaluate(self, eeg,start_time=None, end_time=None, name=None):
        self._evaluation.add_to_evaluate(eeg,start_time=start_time, end_time=end_time, name=name)
    def evaluate(self, plot=True, measures=["SNR"]):
        return self._evaluation.evaluate(plot=plot, measures=measures)
    def export_as_bids(self, event_id=None):
        self._analytics.export_as_bids(event_id=event_id)
    def import_from_bids(self, bids_path="./bids_dir", rel_trig_pos=0, upsampling_factor=10, bads=[]):
        self._eeg = self._analytics.import_from_bids(bids_path, rel_trig_pos=rel_trig_pos, upsampling_factor=upsampling_factor, bads=bads)
        self._correction = Correction_Framework(self._eeg)