from facet.facet import facet
from matplotlib import pyplot as plt

# It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
# Begin Configuration Block
# Path to your EEG file
file_path = "./examples/datasets/NiazyFMRI.edf"
# Event Regex assuming using stim channel
event_regex = r"\b1\b"
# Upsampling factor
upsample_factor = 10
# unwanted channels
unwanted_bad_channels = ["EKG", "EMG", "EOG", "ECG"]
# Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.005
# End Configuration Block

# Loading the EEG data by creating a facet object and importing the EEG data
f = facet()
f.import_eeg(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
)
f.get_eeg().mne_raw.crop(0, 162)
f.find_triggers(event_regex)


# f.plot_eeg(start=29)
f.pre_processing()
# f.find_missing_triggers()
f.align_triggers(0)
# f.align_subsample(0)
# d3_matrix = f.calc_matrix_aas()
d3_matrix = f.calc_matrix_motion(
    file_path="./examples/datasets/headmotiondata_829s.tsv", threshold=0.2
)
# plot 1. matrix
matrix = d3_matrix[0][:150, :150]
plt.imshow(matrix, cmap="gray_r", interpolation="None")
plt.title("Moosmann Averaging Matrix")
plt.xlabel("Epochs targeted")
plt.ylabel("Epochs chosen")
plt.show()
f.remove_artifacts()
# f.get_correction().apply_PCA()
# f.post_processing()

# f.plot_eeg(start=29)
# f.plot_eeg(start=29)
# f.export_eeg("processed_eeg_file.edf")
