FILENAME_eeg = "./examples/datasets/NiazyFMRI.edf"
# Unit Test Class
import pytest, os
from facet.facet import facet
from loguru import logger
import random


class TestAnalysisframework:
    def setup_method(self):
        self.f = facet()
        self.f.import_eeg(
            FILENAME_eeg,
            artifact_to_trigger_offset=-0.005,
            bads=["EMG", "ECG"],
            upsampling_factor=10,
        )
        self.f._eeg.mne_raw.crop(0, 162)
        self.af = self.f.get_analysis()
        self.cf = self.f.get_correction()
        self.ef = self.f.get_evaluation()

    def test_import_eeg(self):
        assert self.f._eeg.mne_raw is not None
        assert self.f._eeg.mne_raw_orig is not None
        assert self.f._eeg.data_time_start == 0

    def test_find_triggers(self):
        self.cf.filter(l_freq=1)
        self.cf.upsample()
        self.af.find_triggers(r"\b1\b")
        assert self.f._eeg.loaded_triggers is not None
        assert self.f._eeg.triggers_as_events is not None
        assert self.f._eeg.count_triggers is not None
        assert self.f._eeg.time_first_artifact_start is not None
        # trigger count should be 840
        assert self.f._eeg.count_triggers == 840
        # check if num_triggers is correct
        assert len(self.f._eeg.loaded_triggers) == self.f._eeg.count_triggers
        assert len(self.f._eeg.triggers_as_events) == self.f._eeg.count_triggers

    def test_missing_triggers(self):
        self.cf.filter(l_freq=1)
        self.cf.upsample()
        self.af.find_triggers(r"\b1\b")

        # remove some triggers
        self.af._eeg.loaded_triggers = self.af._eeg.loaded_triggers[20:820]
        # generate 10 random numbers between 0 and 839
        missing_triggers = random.sample(range(0, 800), 10)
        # remove triggers from loaded_triggers to simulate missing triggers
        for i in missing_triggers:
            self.af._eeg.loaded_triggers.pop(i)
        self.af.find_missing_triggers()

        assert self.f._eeg.loaded_triggers is not None
        assert self.f._eeg.triggers_as_events is not None
        assert self.f._eeg.count_triggers is not None
        assert self.f._eeg.time_first_artifact_start is not None
        # trigger count should be 840
        assert self.f._eeg.count_triggers == 840
        # check if num_triggers is correct
        assert len(self.f._eeg.loaded_triggers) == self.f._eeg.count_triggers
        assert len(self.f._eeg.triggers_as_events) == self.f._eeg.count_triggers

    def test_missing_triggers_sub_periodic_artifacts(self):
        self.cf.filter(l_freq=1)
        self.cf.upsample()
        self.af.find_triggers(r"\b1\b")

        missing_triggers = random.sample(range(0, 800), 10)
        # remove triggers from loaded_triggers to simulate missing triggers
        for i in missing_triggers:
            self.af._eeg.loaded_triggers.pop(i)

        self.af._eeg.loaded_triggers = self.af._eeg.loaded_triggers[15:830]
        self.af.find_missing_triggers()
        # remove every second trigger to simulate sub periodic artifacts (calculating Slice triggers)
        self.af._eeg.loaded_triggers = self.af._eeg.loaded_triggers[::2]  #
        self.af.derive_parameters()
        self.af.find_missing_triggers(add_sub_periodic_artifacts=True)

        assert self.f._eeg.loaded_triggers is not None
        assert self.f._eeg.triggers_as_events is not None
        assert self.f._eeg.count_triggers is not None
        assert self.f._eeg.time_first_artifact_start is not None
        # trigger count should be 840
        assert self.f._eeg.count_triggers == 840
        # check if num_triggers is correct
        assert len(self.f._eeg.loaded_triggers) == self.f._eeg.count_triggers
        assert len(self.f._eeg.triggers_as_events) == self.f._eeg.count_triggers

    def test_plot_eeg(self):
        try:
            self.af.plot_eeg()
        except:
            pytest.fail("plot_eeg raised ExceptionType unexpectedly!")
        assert True
        # Add an assertion here if possible

    def test_export_eeg(self):
        self.af.export_eeg("exported_filename.edf")
        # assert file exists
        assert os.path.isfile("exported_filename.edf")

    def test_artifakt_removal(self):
        # assert number triggers is 840
        self.test_find_triggers()
        self.cf.calc_matrix_aas()
        self.cf.remove_artifacts()
        self.cf.downsample()
        self.cf.filter(h_freq=70)
        self.af.find_triggers(r"\b1\b")
        # evaluate if the artifact removal was successful
        self.ef.add_to_evaluate(self.f._eeg, name="MNE_new")
        results = self.ef.evaluate(
            plot=False, measures=["SNR", "RMS", "RMS2", "MEDIAN"]
        )
        assert results is not None
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

        assert self.f._eeg.mne_raw is not None


# to run this test, run "pytest" in the root directory of the project
