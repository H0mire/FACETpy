import numpy as np

from .FacetController import FacetController
class Facet:

    def __init__(self, RelTrigPos=0.03, Upsample = 10, AvgWindow=30, SliceTriggers=True, UpsampleCutoff=1, InterpolateVolumeGaps=True, OBSExcludeChannels=[]):
        self._facetController = FacetController(RelTrigPos, Upsample, AvgWindow, SliceTriggers, UpsampleCutoff, InterpolateVolumeGaps, OBSExcludeChannels)
        self._relTrigPos = RelTrigPos
        self._upsample = Upsample
        self._avg_window = AvgWindow

    def import_EEG(self, filename):
        self._facetController.import_EEG(filename)

    def import_EEG_GDF(self, filename):
        self._facetController.import_EEG_GDF(filename)
    def export_EEG(self, filename):
        self._facetController.export_EEG(filename)

    def find_triggers(self, regex):
        self._facetController.find_triggers(regex)
        print("finding triggers")
    def find_triggers_with_events(self, regex, idx=0):
        self._facetController.find_triggers_with_events(regex, idx=idx)
        print("finding triggers")
    def prepare(self):
        self._facetController.prepare()
    def apply_AAS(self):
        self._facetController.apply_AAS()
    def apply_MNE_AAS(self):
        #self._facetController.apply_MNE_AAS_old()
        self._facetController.apply_MNE_AAS()
        #self._facetController.apply_MNE_AAS_slow()
    def remove_artifacts(self):  
        self._facetController.remove_artifacts()
    def pre_processing(self):
        self._facetController.highpass(1)
        self._facetController.upsample()
    def post_processing(self):
        self._facetController.downsample()
        self._facetController.lowpass(50)
    def cut(self):
        self._facetController.cut()
    def plot_EEG(self):
        self._facetController.plot_EEG()

    def downsample(self):
        self._facetController.downsample()
    def lowpass(self, h_freq=45):
        self._facetController.lowpass(h_freq=h_freq)
    def highpass(self, l_freq=1):
        self._facetController.highpass(l_freq=l_freq)
    def upsample(self):
        self._facetController.upsample()

    def printName(self):
        self._facetController.printName()
