import mne
import numpy as np
input_file = '/Volumes/JanikProSSD/DataSets/EEG Datasets/EEGfMRI20250310_20180101_014134.mff'

# First load the data using mne
raw = mne.io.read_raw_egi(input_file, preload=True, verbose=False)


# Check if there are any annotations present in the raw object
if hasattr(raw, 'annotations') and raw.annotations is not None and len(raw.annotations) > 0:
    print("Annotations present in raw.annotations:")
    for ann in raw.annotations:
        print(f"  Onset: {ann['onset']}, Duration: {ann['duration']}, Description: {ann['description']}")
else:
    print("No annotations found in raw.annotations.")


# Check specifically for stim channels
stim_channel_names = mne.pick_types(raw.info, stim=True)
if stim_channel_names.size > 0:
    print("\nStim channels found:")
    for idx in stim_channel_names:
        print(f"  {idx}: {raw.ch_names[idx]}")
else:
    print("\nNo stim channels found in the data.")


# Load annotations from stim channel "TREV"
if "TREV" in raw.ch_names:
    print("\nLoading annotations from stim channel 'TREV'...")
    try:
        # Extract events from the TREV stim channel
        events = mne.find_events(
            raw,
            stim_channel="TREV",
            initial_event=True,
            verbose=False
        )
        
        if len(events) > 0:
            print(f"Found {len(events)} events in TREV channel")
            print(f"Event values: {sorted(set(events[:, 2]))}")
            
            # Convert events to annotations
            # Events format: [sample, 0, event_id]
            # Annotations need: onset (in seconds), duration, description
            onsets = events[:, 0] / raw.info['sfreq']  # Convert samples to seconds
            durations = np.zeros(len(events))  # Zero duration for events
            descriptions = [f"TREV_{event_id}" for event_id in events[:, 2]]
            
            # Get orig_time from existing annotations to ensure compatibility
            orig_time = None
            if raw.annotations is not None and len(raw.annotations) > 0:
                orig_time = raw.annotations.orig_time
            
            trev_annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions,
                orig_time=orig_time
            )
            
            # Add annotations to raw object
            if raw.annotations is not None and len(raw.annotations) > 0:
                raw.set_annotations(raw.annotations + trev_annotations)
            else:
                raw.set_annotations(trev_annotations)
            
            print(f"Added {len(trev_annotations)} annotations from TREV channel")
            
            # Display the annotations
            print("\nTREV annotations:")
            for i, ann in enumerate(trev_annotations):
                print(f"  {i+1}. Onset: {ann['onset']:.3f}s, "
                      f"Event ID: {events[i, 2]}, "
                      f"Description: {ann['description']}")
        else:
            print("No events found in TREV channel")
    except Exception as e:
        print(f"Error loading annotations from TREV channel: {e}")
else:
    print("\n'TREV' channel not found in the data.")


# Plot the data
raw.plot()

input("Press Enter to end the script...")