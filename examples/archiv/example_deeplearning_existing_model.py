from facet.facet import facet
from facet.utils.facet_result import FACETResult
from facet.frameworks.deeplearning import ArtifactEstimator
import matplotlib.pyplot as plt
import numpy as np
import torch

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
evaluation_measures = ["SNR", "RMS", "RMS2", "MEDIAN"]
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

f.plot_eeg(start=29)
f.pre_processing()
f.find_missing_triggers()
f.align_triggers(0)
#f.align_subsample(0)
f.calc_matrix_aas()
f.remove_artifacts(plot_artifacts=False)
#f.get_correction().apply_PCA()
f.post_processing()

f.plot_eeg(start=29)

results = f.evaluate(f.get_eeg(), measures=evaluation_measures)
# f.plot_eeg(start=29)
# f.export_eeg('processed_eeg_file.edf')




facet_result = FACETResult.from_facet_object(f)
print(facet_result.get_metadata('_tmin'))

eeg_obj = f.get_eeg()
min_length = min(eeg_obj.mne_raw_orig.times[-1], eeg_obj.mne_raw.times[-1])
eeg_obj.mne_raw_orig.crop(tmin=0, tmax=min_length)
eeg_obj.mne_raw.crop(tmin=0, tmax=min_length)

estimator = ArtifactEstimator.from_file(eeg_obj, "my_artifact_model")

# 1. Prepare epochs
print("Preparing epochs...")
clean_epochs, noisy_epochs = estimator.prepare_epochs()

cleaned_data = estimator.clean_data(noisy_epochs)

eeg_obj.mne_raw_orig.load_data()
cleaned_raw = eeg_obj.mne_raw_orig.copy()
# Only update the first 30 channels, keep the rest unchanged
cleaned_data_reshaped = cleaned_data.transpose(1, 0, 2).reshape(cleaned_data.shape[1], -1)
n_channels_to_update = cleaned_data_reshaped.shape[0]
cleaned_raw._data[:n_channels_to_update,eeg_obj.s_first_artifact_start-1000:eeg_obj.s_first_artifact_start-1000+cleaned_data_reshaped.shape[1]] = cleaned_data_reshaped[:n_channels_to_update]

# Plot comparison of original vs cleaned data for the first channel


eeg_obj.mne_raw = cleaned_raw;
f._eeg = eeg_obj;
f.lowpass();
f.highpass();
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))
times = f.get_eeg().mne_raw.times
plt.plot(times, eeg_obj.mne_raw_orig._data[0, :], label='Original', alpha=0.7)
plt.plot(times, f.get_eeg().mne_raw._data[0, :], label='Cleaned', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Channel 1: Original vs Cleaned EEG Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

f.plot_eeg(start=29)
results_dl = f.evaluate(eeg_obj, measures=evaluation_measures)

f.plot([results, results_dl], evaluation_measures)
