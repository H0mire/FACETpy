from facet.facet import facet

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
f.find_triggers(event_regex)

f.export_eeg(
    "example_simple_bids",
    fmt="bids",
    subject="muellerj",
    session="01",
    task="restingstate",
    event_id={"trigger": 1},
)

f.import_eeg(
    "export/example_simple_bids",
    fmt="bids",
    subject="muellerj",
    session="01",
    task="restingstate",
)

print(f._eeg.mne_raw.info)

input("Press Enter to end the script...")
