from .Frameworks.Correction import Correction_Framework
from .Frameworks.Evaluation import Evaluation_Framework
from .Frameworks.Analytics import Analytics_Framework

class Facet:

    def __init__(self):
        self._analytics = Analytics_Framework()
        self._correction = None
        self._evaluation = Evaluation_Framework()
        self._eeg = None

    def import_EEG(self, filename, rel_trig_pos=0, upsampling_factor=10):
        self._eeg = self._analytics.import_EEG(filename, rel_trig_pos=rel_trig_pos, upsampling_factor=upsampling_factor)
        self._correction = Correction_Framework(self._eeg)
        return self._eeg

    def import_EEG_GDF(self, filename):
        self._eeg = self._analytics.import_EEG_GDF(filename)
    def export_EEG(self, filename):
        self._analytics.export_EEG(filename)
    def get_eeg(self):
        return self._eeg
    def find_triggers(self, regex):
        self._analytics.find_triggers(regex)
        print("finding triggers")
    def find_triggers_with_events(self, regex, idx=0):
        self._analytics.find_triggers_with_events(regex, idx=idx)
        print("finding triggers")
    def prepare(self):
        self._correction.prepare()
    def apply_MNE_AAS(self):
        #self._correction.apply_MNE_AAS_old()
        self._correction.apply_MNE_AAS()
        #self._correction.apply_MNE_AAS_slow()
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
    def plot_EEG(self):
        self._correction.plot_EEG()
    def downsample(self):
        self._correction.downsample()
    def lowpass(self, h_freq=45):
        self._correction.lowpass(h_freq=h_freq)
    def highpass(self, l_freq=1):
        self._correction.highpass(l_freq=l_freq)
    def upsample(self):
        self._correction.upsample()
    def get_mne_raw(self):
        return self._correction.get_mne_raw()
    def add_to_evaluate(self, eeg,start_time=None, end_time=None):
        self._evaluation.add_to_evaluate(eeg,start_time=start_time, end_time=end_time)
    def evaluate(self, plot=True, measures=["SNR"]):
        self._evaluation.evaluate(plot=plot, measures=measures)
