
FILENAME_EEG = "NiazyFMRI.edf"
# Unit Test Class
import pytest, os, edflib
from src.FACET.Facet import Facet
from loguru import logger

class TestAnalyticsFramework:
    def setup_method(self):
        self.af = Facet()
        self.af.import_EEG(FILENAME_EEG, artifact_to_trigger_offset=-0.01, bads=['EMG', 'ECG'])

    def test_import_EEG(self):
        assert self.af._eeg.mne_raw is not None
        assert self.af._eeg.mne_raw_orig is not None
        assert self.af._eeg.data_time_start == 0

    def test_find_triggers(self):
        self.af.pre_processing()
        self.af.find_triggers(r'\b1\b')
        assert self.af._eeg.loaded_triggers is not None
        assert self.af._eeg.triggers_as_events is not None
        assert self.af._eeg.count_triggers is not None
        assert self.af._eeg.time_first_trigger_start is not None
        #trigger count should be 840
        assert self.af._eeg.count_triggers == 840
        #check if num_triggers is correct
        assert len(self.af._eeg.loaded_triggers) == self.af._eeg.count_triggers
        assert len(self.af._eeg.triggers_as_events) == self.af._eeg.count_triggers

    def test_plot_EEG(self):
        try:
            self.af.plot_EEG()
        except:
            pytest.fail("plot_EEG raised ExceptionType unexpectedly!")
        assert True
        # Add an assertion here if possible

    def test_export_EEG(self):
        self.af.export_EEG("exported_filename.edf")
        # assert file exists
        assert os.path.isfile("exported_filename.edf")

    def test_prepare(self):
        self.af.prepare()
        assert self.af._eeg is not None

    def test_artifakt_removal(self):
        #assert number triggers is 840
        self.test_find_triggers()
        self.af.apply_AAS(method="numpy")
        self.af.remove_artifacts()
        self.af.downsample()
        self.af.lowpass(h_freq=40)
        self.af.find_triggers(r'\b1\b')
        #evaluate if the artifact removal was successful
        self.af.add_to_evaluate(self.af.get_eeg(), name="MNE_new")
        results = self.af.evaluate(plot=False,measures=["SNR", "RMS", "RMS2", "MEDIAN"])
        logger.info(results)
        for result in results:
            if result["Measure"] == "SNR":
                assert result["Values"][0] > 3
            if result["Measure"] == "RMS":
                assert result["Values"][0] > 60
            if result["Measure"] == "RMS2":
                assert result["Values"][0] < 1.1 and result["Values"][0] > 0.9 
            if result["Measure"] == "MEDIAN":
                assert result["Values"][0] < 1
        assert results is not None
        
        assert self.af._eeg.mne_raw is not None
    

#to run this test, run "pytest" in the root directory of the project    
