from .Frameworks.Correction import Correction_Framework
from .Frameworks.Evaluation import Evaluation_Framework

class Facet:

    def __init__(self, relative_trigger_position=0.03, upsample = 10):
        self._correction = Correction_Framework(relative_trigger_position, upsample)
        self._rel_trig_pos = relative_trigger_position
        self._upsample = upsample

    def import_EEG(self, filename):
        self._correction.import_EEG(filename)

    def import_EEG_GDF(self, filename):
        self._correction.import_EEG_GDF(filename)
    def export_EEG(self, filename):
        self._correction.export_EEG(filename)

    def find_triggers(self, regex):
        self._correction.find_triggers(regex)
        print("finding triggers")
    def find_triggers_with_events(self, regex, idx=0):
        self._correction.find_triggers_with_events(regex, idx=idx)
        print("finding triggers")
    def prepare(self):
        self._correction.prepare()
    def apply_MNE_AAS(self):
        #self._facetController.apply_MNE_AAS_old()
        self._correction.apply_MNE_AAS()
        #self._facetController.apply_MNE_AAS_slow()
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

    #TODO: implement better structure
    def init_evaluation_framework(self):
        temp = self._correction.get_raw_eeg()
        self._evaluation = Evaluation_Framework()
        self._evaluation.init_with_correction(self._correction)

    def evaluate_SNR(self):
        SNR = self._evaluation.evaluate_SNR()
        print(SNR)
        return
    def evaluate_MRA(self):
        #TODO: Implement
        return
    def evaluate_RMS(self):
        #TODO: Implement
        return
    
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
