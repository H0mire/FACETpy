from facet.eeg_obj import EEG
from facet.facet import facet
from facet.utils.facet_result import FACETResult
import numpy as np
import mne

file_path = '/Volumes/JanikProSSD/DataSets/EEG Datasets/EEGfMRI20250310_20180101_014134.mff'

artifact_to_trigger_offset_in_seconds = -0.030
evaluation_measures = ["SNR", "RMS", "RMS2", "MEDIAN"]

unwanted_bad_channels = []


f = facet()

raw = mne.io.read_raw_egi(file_path, preload=True, verbose=False)
raw.info['bads'] = unwanted_bad_channels

print(raw.n_times)
eeg = EEG(
            mne_raw=raw,
            mne_raw_orig=raw.copy(),
            estimated_noise=(
                np.zeros((len(raw.ch_names), raw.n_times))
            ),
            artifact_to_trigger_offset=artifact_to_trigger_offset_in_seconds,
            upsampling_factor=1,
            data_time_start=raw.times[0],
            data_time_end=raw.times[-1],
        )
print(eeg.estimated_noise.shape)
f.import_by_eeg_obj(eeg)
f.highpass(1)
f.upsample()
f.find_triggers(r"\b1\b", save=True)
f.pad_missing_triggers(count_before=10, count_after=0)
f.align_triggers(0)
# f.align_subsample(0)
# Calculate the averaging matrix
f.calc_matrix_aas()

print(f.get_eeg().estimated_noise.shape)
# Calculate the artifacts and remove them
f.remove_artifacts(plot_artifacts=True)
f.get_analysis().find_triggers_qrs()
f.calc_matrix_aas()
f.remove_artifacts(plot_artifacts=True)

# Define a clean reference interval (without artifacts) for evaluation
# Adjust these times based on your data - this should be an interval that doesn't contain artifacts
ref_start_time = 600.0  # Start of clean reference interval in seconds
ref_end_time = 1000.0    # End of clean reference interval in seconds

# Signal interval: the corrected signal to be evaluated (typically contains artifacts)
signal_start_time = 2800.0   # Start of corrected signal interval in seconds
signal_end_time = 3200.0   # End of corrected signal interval in seconds

results_after_aas = f.evaluate(
    f.get_eeg(), 
    measures=evaluation_measures, 
    name="After AAS",
    start_time=signal_start_time,
    end_time=signal_end_time,
    ref_start_time=ref_start_time,
    ref_end_time=ref_end_time
)
# Now Postprocess the data
#f.get_correction().apply_PCA()
f.downsample()
f.lowpass(70)
#f.apply_ANC()
results_after_anc = f.evaluate(
    f.get_eeg(), 
    measures=evaluation_measures, 
    name="After ANC",
    start_time=signal_start_time,
    end_time=signal_end_time,
    ref_start_time=ref_start_time,
    ref_end_time=ref_end_time
)
f.plot_eeg(title="after postprocessing")

print(results_after_aas)
print(results_after_anc)
f.plot([results_after_aas, results_after_anc], plot_measures=evaluation_measures)
f.export_eeg(file_path.replace(".mff", "_processed.edf"), fmt="edf")
input("Press Enter to end the script...")