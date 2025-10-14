from facet.facet import facet
from facet.utils.facet_result import FACETResult
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# It is adviced to add a configuration block here, to keep an overview of the settings used for the analysis.
# Begin Configuration Block
# Path to your EEG file
file_path = "./examples/datasets/output_with_qrs_events.edf"
# Path to your event file
event_file_path = "./examples/datasets/events.txt"
# Event Regex assuming using stim channel
event_regex = r"\b1\b"
# Upsampling factor
upsample_factor = 10
# unwanted channels
unwanted_bad_channels = ["EKG", "EMG", "EOG", "ECG"]
# Add Artifact to Trigger Offset in seconds. Adjust this if the trigger events are not aligned with the artifact occurence
artifact_to_trigger_offset = -0.005
# Sample range to plot
start_sample = 10000
end_sample = 30000
# End Configuration Block

# Loading the EEG data by creating a facet object and importing the EEG data
f = facet()
f.import_eeg(
    file_path,
    upsampling_factor=upsample_factor,
    bads=unwanted_bad_channels,
    artifact_to_trigger_offset=artifact_to_trigger_offset,
    fmt="edf",
)

# Read the event file
event_data = pd.read_csv(event_file_path, sep="\t")

# Get the length of the EEG data
data_length = f.get_eeg().mne_raw._data.shape[1]

# Create an array of zeros with the same length as the EEG data
event_array = np.zeros(data_length)

# Place 1's at the event latencies (converting latencies to integers)
latencies = event_data['latency'].astype(int).values
event_array[latencies] = 1

# Get the ECG channel data (channel 32)
ecg_data = f.get_eeg().mne_raw._data[31]

# Create time array (assuming sampling rate from the EEG data)
sampling_rate = f.get_eeg().mne_raw.info['sfreq']
time = np.arange(len(ecg_data)) / sampling_rate

# Limit to specified sample range
time = time[start_sample:end_sample]
ecg_data = ecg_data[start_sample:end_sample]

# Filter events that fall within the specified sample range
mask = (latencies >= start_sample) & (latencies < end_sample)
filtered_latencies = latencies[mask]

# Create the plot
plt.figure(figsize=(15, 6))
plt.plot(time, ecg_data, 'b-', label='ECG')
# Plot events as points using ECG amplitude at those positions
if len(filtered_latencies) > 0:  # Only plot events if there are any in this range
    plt.plot(filtered_latencies/sampling_rate, ecg_data[filtered_latencies - start_sample], 'ro', label='Events', markersize=10)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f'ECG Data with Events (Samples {start_sample} to {end_sample})')
plt.legend()
plt.grid(True)
plt.show()

input("Press Enter to end the script...")
