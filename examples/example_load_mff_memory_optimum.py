from facet.facet import facet
import time

# Begin Configuration Block
# Path to your MFF EEG file
file_path = '/Volumes/JanikProSSD/DataSets/EEG Datasets/EEGfMRI20250310_20180101_014134.mff'
# Event Regex assuming using stim channel
event_regex = r"\b1\b"
# Upsampling factor
upsample_factor = 10
# unwanted channels
unwanted_bad_channels = []
# Add Artifact to Trigger Offset in seconds
artifact_to_trigger_offset = -0.030
# Evaluation measures
evaluation_measures = ["SNR", "RMS", "RMS2", "MEDIAN"]
# Define a clean reference interval (without artifacts) for evaluation
ref_start_time = 600.0
ref_end_time = 1000.0
# Signal interval: the corrected signal to be evaluated
signal_start_time = 2800.0
signal_end_time = 3200.0
# End Configuration Block

# Start measuring time
start_time = time.time()

f = facet()

f.import_eeg(
    file_path,
    fmt="mff",  # MFF format for EGI files
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
    preload=False,  # Important for memory optimization
)

# Step 1: Find gradient artifact triggers
f.find_triggers(event_regex, save=True)
f.align_triggers(0)
gradient_triggers = f.get_eeg().loaded_triggers.copy()  # Store gradient triggers

# Step 2: Detect QRS peaks from ECG channel BEFORE apply_per_channel
# This needs to be done with ECG channel loaded
f_ecg = f.create_facet_with_channel_picks(['ECG'])
f_ecg.get_analysis().find_triggers_qrs(ref_channel='ECG')
qrs_triggers = f_ecg.get_eeg().loaded_triggers  # Store QRS triggers for later use


def apply_per_channel(f):
    # Step 3a: Remove gradient artifacts
    f.highpass(1)
    f.upsample()
    f.get_eeg().loaded_triggers = gradient_triggers
    f.get_analysis().derive_parameters()
    f.calc_matrix_aas()
    f.remove_artifacts(plot_artifacts=False)
    
    # Step 3b: Remove BCG artifacts using pre-detected QRS triggers
    f.get_eeg().loaded_triggers = qrs_triggers
    f.get_analysis().derive_parameters()
    f.calc_matrix_aas()
    f.remove_artifacts(plot_artifacts=True)
    
    f.downsample()
    f.lowpass(70)


f.apply_per_channel(apply_per_channel)

# End measuring time
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# Evaluation
results_after_processing = f.evaluate(
    f.get_eeg(),
    measures=evaluation_measures,
    name="After Processing",
    start_time=signal_start_time,
    end_time=signal_end_time,
    ref_start_time=ref_start_time,
    ref_end_time=ref_end_time
)

f.plot_eeg(title="after postprocessing")

print(results_after_processing)
f.plot([results_after_processing], plot_measures=evaluation_measures)

f.export_eeg(file_path.replace(".mff", "_processed.edf"), fmt="edf")

input("Press Enter to end the script...")
