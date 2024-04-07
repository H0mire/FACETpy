from FACET.Facet import Facet
from loguru import logger
import numpy as np
import scipy.io as sio

# It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
# Begin Configuration Block
# Path to your EEG file
file_path_edf_without_anc = (
    "C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_without_lowpass.edf"
)
file_path_edf_with_anc = (
    "C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_with_anc.edf"
)
file_path_edf_with_alignment = (
    "C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_with_alignment.edf"
)
file_path_edf_with_alignment_subsamplealignment_anc = "C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_with_alignment_subsamplealignment_anc.edf"
file_path_edf_full = (
    "C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_full.edf"
)
file_path_edf_full_fastr = (
    "C:\\Users\\janik\\Projekte\\pyFACET\\Datasets\\Matlab_cleaned_full_fastr.edf"
)

event_regex = r"\b1\b"


# End Configuration Block

# Loading the EEG data by creating a FACET object and importing the EEG data
f = Facet()
f.import_EEG(file_path_edf_without_anc)
f.get_EEG().mne_raw.crop(0, 162)
f.find_triggers(event_regex)
f.lowpass(70)
f.add_to_evaluate(f.get_EEG(), name="Without ANC")
f.plot_EEG(start=29, title="Without ANC")

f.import_EEG(file_path_edf_with_anc)
f.get_EEG().mne_raw.crop(0, 162)
f.find_triggers(event_regex)
f.add_to_evaluate(f.get_EEG(), name="With ANC")
f.plot_EEG(start=29, title="With ANC")

# Own implementation
file_path = "src/NiazyFMRI.edf"
# Event Regex assuming using stim channel
# Upsampling factor
upsample_factor = 10

relative_window_offset = -0.5
# unwanted channels
unwanted_bad_channels = ["EKG", "EMG", "EOG", "ECG"]
# Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.005
# End Configuration Block
# Loading the EEG data by creating a FACET object and importing the EEG data
f.import_EEG(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)
f.get_EEG().mne_raw.crop(0, 162)
# f.pre_processing()
f.highpass(1.5)
f.upsample()
f.find_triggers(event_regex)
f.align_triggers(0)
# f.calc_matrix_motion("./src/headmotiondata.tsv")
f.calc_matrix_AAS(rel_window_position=relative_window_offset)
f.remove_artifacts(plot_artifacts=False)
# f.post_processing()
# loaded_trigger_positions = f.get_EEG().loaded_triggers
f.downsample()
f.lowpass(70)
eeg_without_anc = f.get_EEG().copy()
f.plot_EEG(start=29, title="Own Implementation without ANC", eeg=eeg_without_anc)
f.add_to_evaluate(eeg_without_anc, name="Own Implementation without ANC")
# f.get_EEG().loaded_triggers = (np.asarray(loaded_trigger_positions) / upsample_factor).astype(int).tolist()

f.get_correction().apply_ANC()
f.add_to_evaluate(f.get_EEG(), name="Own Implementation with anc")
f.plot_EEG(start=29, title="Own Implementation with ANC")

logger.info(f.evaluate(plot=True, measures=["SNR", "RMS", "RMS2", "MEDIAN"]))

input("Press Enter to end the script...")
